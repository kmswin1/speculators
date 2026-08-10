#!/usr/bin/env python3
"""Offline vLLM driver to measure speculative-decoding acceptance length for a
DSpark draft speculator against the A.X-K2 verifier.

Design
------
* Uses the **offline** ``vllm.LLM`` API (single node, TP8) -- self-contained,
  tears itself down on process exit, no lingering server.
* Acceptance metrics come from ``llm.get_metrics()`` (vLLM V1 in-memory
  Prometheus snapshot). The relevant counters are:
      vllm:spec_decode_num_drafts            (Counter)
      vllm:spec_decode_num_draft_tokens      (Counter)
      vllm:spec_decode_num_accepted_tokens   (Counter)
      vllm:spec_decode_num_accepted_tokens_per_pos  (Vector, one per position)
  Mean acceptance length (bonus token included, vLLM convention) is
      1 + num_accepted_tokens / num_drafts
  Per-position acceptance rate is
      accepted_per_pos[k] / num_drafts
* Run one mode per process (``--mode base`` or ``--mode spec``); the sbatch
  invokes both and computes the speedup. ``--mode both`` is available but loads
  the ~400GB verifier twice in one process, which is riskier -- prefer separate
  invocations.

Key run-time knobs come from env (see run_accept_test.sbatch):
    TARGET, CKPT_DRAFT, SPEC_METHOD, NUM_SPEC_TOKENS, TP, MAX_MODEL_LEN,
    GPU_MEM_UTIL, ENABLE_EP, ENFORCE_EAGER, MAX_TOKENS, ALLOW_DFLASH_FALLBACK

IMPORTANT (verified 2026-08-08 by inspecting the container squashfs):
    The target container vllm-v0.23.0-for-sktax.sqsh is vLLM 0.23.0+sktax and
    registers ONLY speculators methods {eagle3, peagle, dflash} -- it does NOT
    know "dspark". Loading the DSpark checkpoint with method="dspark" will raise
    a ValueError. See README_findings.md. Set SPEC_METHOD=dflash together with
    ALLOW_DFLASH_FALLBACK=1 to measure the DFlash *backbone only* (drops the
    Markov + confidence heads -> a LOWER-BOUND, NOT true DSpark) so you can still
    get a number on this container today.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
# Prompt set: Korean + English + a short reasoning prompt.
# --------------------------------------------------------------------------- #
BASE_PROMPTS: list[str] = [
    # --- Korean ---
    "대한민국의 수도는 어디이며, 그 도시의 대표적인 명소 세 곳을 알려줘.",
    "인공지능이 일상생활에 미치는 긍정적인 영향과 부정적인 영향을 각각 설명해줘.",
    "김치를 담그는 기본적인 과정을 순서대로 간단히 설명해줘.",
    "다음 문장을 정중한 표현으로 바꿔줘: '내일 회의 시간 좀 바꿔.'",
    "머신러닝과 딥러닝의 차이를 초등학생도 이해할 수 있게 설명해줘.",
    "서울에서 부산까지 여행 계획을 2박 3일 일정으로 짜줘.",
    # --- English ---
    "Explain the difference between TCP and UDP in a few sentences.",
    "Write a short professional email declining a meeting invitation.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "List five practical tips for improving focus while working from home.",
    "Describe how a hash map works and why lookups are O(1) on average.",
    "What are the main causes of inflation? Give a concise explanation.",
    # --- Short reasoning ---
    "A train leaves at 3:15 PM and arrives at 6:45 PM. How long is the trip? "
    "Think step by step and give the final answer.",
    "If a shirt costs $40 after a 20% discount, what was the original price? "
    "Show your reasoning.",
    "There are 5 red balls and 3 blue balls in a bag. If you draw 2 without "
    "replacement, what is the probability both are red? Reason step by step.",
    "Translate to Korean and then explain your translation choices: "
    "'The early bird catches the worm.'",
    # --- Code generation ---
    "Write a Python function that returns the nth Fibonacci number using "
    "memoization. Include a short docstring and a couple of example calls.",
    "Implement binary search in Python. Return the index of the target or -1 "
    "if not found, and briefly explain the time complexity.",
    "Write a Python class `Stack` with push, pop, peek, and is_empty methods, "
    "then show a short usage example.",
    "Write a SQL query to find the second-highest salary from an `Employees` "
    "table, and explain how it works.",
    "파이썬으로 주어진 문자열이 팰린드롬인지 확인하는 함수를 작성하고, "
    "대소문자와 공백을 무시하도록 처리한 뒤 테스트 예시도 보여줘.",
    # --- Long reasoning ---
    "A farmer has chickens and rabbits. There are 35 heads and 94 legs in "
    "total. How many chickens and how many rabbits are there? Think step by "
    "step, set up the equations explicitly, and solve them carefully.",
    "You have 8 identical-looking balls; one is slightly heavier. Using a "
    "balance scale only twice, how can you guarantee finding the heavy ball? "
    "Walk through every case in your reasoning before giving the answer.",
    "Prove step by step that the square root of 2 is irrational. Lay out the "
    "assumption, the algebra, and the contradiction explicitly.",
    "세 친구가 식당 계산서를 나눠 냈다. Alice가 40%, Bob이 35%를 냈고, Carol이 "
    "낸 나머지가 12,500원이었다. 전체 금액과 각자 낸 금액을 단계별로 추론해서 구해줘.",
]


def build_prompts(target_count: int) -> list[str]:
    prompts: list[str] = []
    while len(prompts) < target_count:
        prompts.extend(BASE_PROMPTS)
    return prompts[:target_count]


# --------------------------------------------------------------------------- #
# DFlash-compatibility view of a DSpark checkpoint (fallback path only).
# --------------------------------------------------------------------------- #
def make_dflash_view(ckpt_dir: str) -> str:
    """Create a temp dir that presents a DSpark checkpoint as a DFlash one.

    vLLM 0.23.0+sktax rejects speculators_model_type=="dspark". DSpark subclasses
    DFlash, so we rewrite ONLY config.json (speculators_model_type -> "dflash",
    architectures -> ["DFlashDraftModel"]) and symlink every other file. The
    extra markov_head.*/confidence_head.* tensors in model.safetensors are simply
    ignored by the DFlash model loader.

    UNVERIFIED: whether the container's DFlash weight loader tolerates the extra
    tensors without error, and whether the DFlash inference path produces sane
    output from DSpark-trained weights. Confirm at run time.
    """
    src = Path(ckpt_dir)
    cfg_path = src / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("speculators_model_type") == "dflash":
        return str(src)  # already dflash, nothing to do

    view = Path(tempfile.mkdtemp(prefix="dspark_as_dflash_"))
    for item in src.iterdir():
        if item.name == "config.json":
            continue
        (view / item.name).symlink_to(item.resolve())

    cfg["speculators_model_type"] = "dflash"
    cfg["architectures"] = ["DFlashDraftModel"]
    # DFlash speculator subtype marker some builds look for; harmless if unused.
    (view / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[fallback] wrote DFlash-view config to {view}", flush=True)
    return str(view)


# --------------------------------------------------------------------------- #
# Metrics extraction from llm.get_metrics()
# --------------------------------------------------------------------------- #
def extract_spec_metrics(metrics) -> dict:
    """Pull spec-decode acceptance stats out of a vLLM get_metrics() snapshot."""
    drafts = draft_tok = accepted = 0.0
    per_pos: list[float] = []
    for m in metrics:
        name = getattr(m, "name", "")
        if name == "vllm:spec_decode_num_drafts":
            drafts += float(getattr(m, "value", 0.0))
        elif name == "vllm:spec_decode_num_draft_tokens":
            draft_tok += float(getattr(m, "value", 0.0))
        elif name == "vllm:spec_decode_num_accepted_tokens":
            accepted += float(getattr(m, "value", 0.0))
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            vals = getattr(m, "values", None)
            if vals is not None:
                if not per_pos:
                    per_pos = [0.0] * len(vals)
                for i, v in enumerate(vals):
                    if i >= len(per_pos):
                        per_pos.extend([0.0] * (i - len(per_pos) + 1))
                    per_pos[i] += float(v)

    out = {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tok,
        "num_accepted_tokens": accepted,
        "acceptance_length": (1 + accepted / drafts) if drafts > 0 else 0.0,
        "per_position_acceptance": [
            (c / drafts if drafts > 0 else 0.0) for c in per_pos
        ],
    }
    return out


# --------------------------------------------------------------------------- #
# Core run
# --------------------------------------------------------------------------- #
def run(mode: str) -> dict:
    from vllm import LLM, SamplingParams  # imported after env is set

    target = os.environ.get(
        "TARGET", "/t1data/users/kmswin7/A.X-K2_fused_yarn2"
    )
    ckpt_draft = os.environ.get(
        "CKPT_DRAFT", "/t1data/users/kmswin7/speculators/dspark_ckpt_3n/0"
    )
    spec_method = os.environ.get("SPEC_METHOD", "dspark")
    num_spec = int(os.environ.get("NUM_SPEC_TOKENS", "5"))
    tp = int(os.environ.get("TP", "8"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    gpu_mem = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
    enable_ep = os.environ.get("ENABLE_EP", "1") == "1"
    enforce_eager = os.environ.get("ENFORCE_EAGER", "0") == "1"
    max_tokens = int(os.environ.get("MAX_TOKENS", "256"))
    num_prompts = int(os.environ.get("NUM_PROMPTS", "48"))
    allow_dflash_fallback = os.environ.get("ALLOW_DFLASH_FALLBACK", "0") == "1"

    prompts = build_prompts(num_prompts)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    common = dict(
        model=target,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_ep,
        disable_log_stats=False,  # REQUIRED: offline LLM defaults True → get_metrics() empty
    )
    if os.environ.get("DISABLE_ASYNC_SCHED", "0") == "1":
        # async scheduling's _prepare_input_ids fast-path requires a rectangular
        # draft tensor; disable it so the conf ragged list[list[int]] path flows.
        common["async_scheduling"] = False
        print("[cfg] async_scheduling=False (for conf ragged path)", flush=True)

    if mode == "spec":
        draft_path = ckpt_draft
        if spec_method == "dflash" and allow_dflash_fallback:
            draft_path = make_dflash_view(ckpt_draft)
        spec_cfg = {
            "model": draft_path,
            "method": spec_method,
            "num_speculative_tokens": num_spec,
        }
        print(f"[spec] speculative_config={json.dumps(spec_cfg)}", flush=True)
        llm = LLM(speculative_config=spec_cfg, **common)
    else:
        print("[base] no speculative_config (plain decoding)", flush=True)
        llm = LLM(**common)

    # Warmup (not timed) so CUDA graphs / caches are hot.
    llm.generate(prompts[:4], sampling, use_tqdm=False)

    batch1 = os.environ.get("BATCH1", "0") == "1"
    if batch1:
        # Single-stream latency regime: one request at a time (batch size 1).
        # This is where spec decode helps (GPU under-utilized -> drafting fills
        # idle FLOPs). tps here = single-stream decode throughput = 1000/(ms per token).
        outputs = []
        elapsed = 0.0
        for p in prompts:
            t0 = time.perf_counter()
            o = llm.generate([p], sampling, use_tqdm=False)
            elapsed += time.perf_counter() - t0
            outputs.extend(o)
    else:
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_out / elapsed if elapsed > 0 else 0.0
    ms_per_tok = (elapsed * 1000.0 / total_out) if total_out > 0 else 0.0
    print(f"MS_PER_TOKEN_{mode.upper()}={ms_per_tok:.3f}  (batch1={batch1})", flush=True)

    # Output fingerprint so base vs spec divergence is detectable across the two
    # separate processes (greedy spec decode MUST be lossless => identical output).
    import hashlib
    per_len = [len(o.outputs[0].token_ids) for o in outputs]
    concat = ",".join(str(t) for o in outputs for t in o.outputs[0].token_ids)
    fp = hashlib.md5(concat.encode()).hexdigest()[:12]
    preview = outputs[0].outputs[0].text[:160].replace("\n", " ")
    print(f"OUTPUT_FP_{mode.upper()}={fp}", flush=True)
    print(f"OUTPUT_LENS_{mode.upper()}={per_len}", flush=True)
    print(f"OUTPUT_PREVIEW_{mode.upper()}={preview}", flush=True)

    result = {
        "mode": mode,
        "num_prompts": len(prompts),
        "total_output_tokens": total_out,
        "elapsed_s": round(elapsed, 3),
        "output_tps": round(tps, 2),
    }

    if mode == "spec":
        try:
            spec = extract_spec_metrics(llm.get_metrics())
        except Exception as e:  # noqa: BLE001
            print(f"[warn] get_metrics() failed: {e}", flush=True)
            spec = {}
        result["spec"] = spec

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "spec", "both"],
                    default=os.environ.get("MODE", "spec"))
    args = ap.parse_args()

    modes = ["base", "spec"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        results[mode] = run(mode)

    print("\n================ RESULTS ================", flush=True)
    for mode, r in results.items():
        print(f"[{mode}] output_tps={r['output_tps']}  "
              f"tokens={r['total_output_tokens']}  elapsed={r['elapsed_s']}s")
        # machine-readable lines the sbatch greps
        print(f"OUTPUT_TPS_{mode.upper()}={r['output_tps']}")
        if mode == "spec" and r.get("spec"):
            spec = r["spec"]
            mal = spec.get("acceptance_length", 0.0)
            print(f"MEAN_ACCEPT_LEN={mal:.4f}")
            print(f"NUM_DRAFTS={spec.get('num_drafts', 0):.0f}")
            print(f"NUM_ACCEPTED_TOKENS={spec.get('num_accepted_tokens', 0):.0f}")
            print(f"NUM_DRAFT_TOKENS={spec.get('num_draft_tokens', 0):.0f}")
            pos = spec.get("per_position_acceptance", [])
            print("PER_POSITION_ACCEPTANCE=" + ",".join(f"{p:.4f}" for p in pos))
            for i, p in enumerate(pos):
                print(f"  accept_at_pos_{i}={p:.4f}")

    if args.mode == "both":
        b = results.get("base", {}).get("output_tps", 0.0)
        s = results.get("spec", {}).get("output_tps", 0.0)
        if b > 0:
            print(f"SPEEDUP_SPEC_OVER_BASE={s / b:.3f}")
    print("=========================================", flush=True)


if __name__ == "__main__":
    sys.exit(main())
