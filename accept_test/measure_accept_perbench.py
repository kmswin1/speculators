#!/usr/bin/env python3
"""Per-benchmark-task mean acceptance-length for the DSpark drafter on A.X-K2.

TWO-STAGE design (the baked vLLM container has NO ``datasets``/``lev``)
----------------------------------------------------------------------
Stage 1 -- CAPTURE (host, outside the container; needs ``datasets`` + ``lev``):
    python measure_accept_perbench.py --capture
  Uses lev's own task loaders + a CaptureEngine to collect, per task, the exact
  ``messages`` lists lev builds (system+user, correct chat shape), sampling
  N_PER_TASK prompts per task with lev's fixed seed 220302, and DUMPS them to
  ``PROMPTS_JSON`` (default accept_test/perbench_prompts.json) as
      {task_name: [ [ {"role":..,"content":..}, .. ], .. ], ..}
  This mode does NOT import vllm and does NOT build an engine. Run it under a
  Python 3.11+ venv that has datasets + lev on the path (lev/.venv works;
  speculators_venv is py3.10 and CANNOT import lev, which uses typing.Self).

Stage 2 -- RUN (inside the baked container; needs only vllm). LOADS
``PROMPTS_JSON`` (does NOT import lev/datasets), builds ONE engine per invocation
and measures each task's prompts at **batch=1 (single-stream latency)**:
    python measure_accept_perbench.py --mode base   # plain A.X-K2 -> base_tps
    python measure_accept_perbench.py --mode spec   # dspark      -> spec_tps + accept_len
    python measure_accept_perbench.py --combine      # merge -> per-task table
  BASE and SPEC are separate invocations (each loads/frees one ~400GB engine) so
  a single GPU node can do both sequentially without a double in-process load.
  SPEC accept_len comes from the cumulative spec-decode counter DELTA per task:

      mean_accept_len   = 1 + Δnum_accepted_tokens / Δnum_drafts
      per_pos_accept[k]  = Δnum_accepted_tokens_per_pos[k] / Δnum_drafts

  (counters are cumulative over the engine's life; acceptance length is a
  per-draft property => batch-invariant => batch=1 is a valid regime for it.)
  ``disable_log_stats=False`` is required or ``llm.get_metrics()`` is empty.
  --combine emits:  TASK | N | ACCEPT_LEN | BASE_TPS | SPEC_TPS | SPEEDUP.

Env knobs
---------
  CAPTURE:  PERBENCH_CAPTURE_ONLY, PROMPTS_JSON, N_PER_TASK, SEED, TASKS,
            CAPTURE_TIMEOUT_S
  RUN:      PROMPTS_JSON, TARGET, CKPT_DRAFT, CKPT_BASE, NUM_SPEC_TOKENS, TP,
            MAX_MODEL_LEN, GPU_MEM_UTIL, ENABLE_EP, ENFORCE_EAGER, SEED,
            TEMPERATURE, TOP_P, MAX_TOKENS (default 256), TASKS,
            RESULTS_BASE, RESULTS_SPEC
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


# --------------------------------------------------------------------------- #
# The 14-task lev suite (registry names) + a reasonable per-task max_tokens.
# The lev suite itself set NO max_tokens (server default, full reasoning length);
# we cap for bounded offline runtime. Acceptance length is largely stable across
# output length. Override all tasks at once with env MAX_TOKENS.
# --------------------------------------------------------------------------- #
SUITE_TASKS: list[str] = [
    "aa-lcr",
    "aa-omni-public",
    "aime26",
    "apex-shortlist",
    "apex2025",
    "click",
    "gpqa",
    "hle",
    "ifbench",
    "kmmlu-pro",
    "kmo26",
    "kobalt",
    "livecodebench",
    "scicode",
]

# per-task generation cap (tokens)
TASK_MAX_TOKENS: dict[str, int] = {
    "aa-lcr": 4096,
    "aa-omni-public": 2048,
    "aime26": 4096,
    "apex-shortlist": 4096,
    "apex2025": 4096,
    "click": 2048,
    "gpqa": 4096,
    "hle": 4096,
    "ifbench": 2048,
    "kmmlu-pro": 4096,
    "kmo26": 4096,
    "kobalt": 4096,
    "livecodebench": 4096,
    "scicode": 4096,
}
DEFAULT_MAX_TOKENS = 4096

DEFAULT_PROMPTS_JSON = "/t1data/users/kmswin7/speculators/accept_test/perbench_prompts.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_tasks() -> list[str]:
    tasks_env = os.environ.get("TASKS", "").strip()
    if tasks_env:
        return [t.strip() for t in tasks_env.split(",") if t.strip()]
    return list(SUITE_TASKS)


# =========================================================================== #
# STAGE 1: CAPTURE (imports lev + datasets; NO vllm)
# =========================================================================== #

def patch_gpt_judge() -> None:
    """Neutralise the OpenAI 'gpt' judge engine so judge tasks load offline.

    GPTEngine.set_endpoint() does a live /v1/models health check and raises
    without OPENAI_API_KEY; its generate_* would hit the network during scoring.
    Replace both with no-network stubs. Must run BEFORE load_task().
    """
    from lev.engines.gpt import GPTEngine
    from lev.types import Generation

    def _set_endpoint(self, endpoint: str = "", api_key: str | None = None):
        self.endpoint = endpoint or "https://dummy.invalid"
        self.api_key = api_key or "dummy"

    def _is_healthy(self) -> bool:
        return True

    async def _dummy_chat(self, request, gen_index: int = 0, jitter: float = 0.0):
        return Generation(
            request={}, generated_raw="", generated_answer="",
            input_tokens=0, output_tokens=0, finish_reason="stop",
        )

    async def _dummy_completion(self, request, gen_index: int = 0, jitter: float = 0.0):
        return Generation(
            request={}, generated_raw="", generated_answer="",
            input_tokens=0, output_tokens=0, finish_reason="stop",
        )

    GPTEngine.set_endpoint = _set_endpoint
    GPTEngine.is_healthy = _is_healthy
    GPTEngine.generate_chat_single = _dummy_chat
    GPTEngine.generate_completion_single = _dummy_completion
    log("[patch] GPTEngine judge neutralised (offline, no network).")


def make_capture_engine():
    from lev.engines.base import InferEngine
    from lev.types import Generation

    class CaptureEngine(InferEngine):
        """Records chat request messages; returns a benign empty Generation.

        No network, no queue manager. Implements every abstract InferEngine
        method plus no-op async context management.
        """

        def __init__(self):
            self.endpoint = "capture://local"
            self.model_name = ""
            self.api_key = None
            self.captured: list[list[dict]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        def set_endpoint(self, endpoint: str = "", api_key: str | None = None):
            self.endpoint = endpoint or "capture://local"
            self.api_key = api_key

        def is_healthy(self) -> bool:
            return True

        @staticmethod
        def _messages_to_dicts(messages) -> list[dict]:
            out: list[dict] = []
            for m in messages:
                # m is a lev.types.Message; content is str for all suite tasks.
                out.append({"role": m.role, "content": m.content})
            return out

        def _dummy(self) -> Generation:
            return Generation(
                request={}, generated_raw="", generated_answer="",
                input_tokens=0, output_tokens=0, finish_reason="stop",
            )

        async def generate_chat_single(self, request, gen_index: int = 0, jitter: float = 0.0):
            self.captured.append(self._messages_to_dicts(request.messages))
            return self._dummy()

        async def generate_completion_single(self, request, gen_index: int = 0, jitter: float = 0.0):
            self.captured.append([{"role": "user", "content": request.prompt}])
            return self._dummy()

        async def generate_completion_async(self, data, gen_per_sample: int = 1, desc: str = ""):
            outs = []
            for req in data:
                for _ in range(gen_per_sample):
                    outs.append(await self.generate_completion_single(req))
            return outs

        async def generate_chat_async(self, data, gen_per_sample: int = 1, desc: str = ""):
            outs = []
            for req in data:
                for _ in range(gen_per_sample):
                    outs.append(await self.generate_chat_single(req))
            return outs

    return CaptureEngine()


def capture_task_prompts(task_name: str, n: int, timeout_s: float) -> list[list[dict]]:
    """Return up to n chat conversations for a task via lev's own loader.

    gen_per_sample=1 + limit=n => samples_to_use(n) (lev default seed 220302)
    selects exactly the subsample our policy uses. Any scoring-side error/timeout
    still yields whatever was captured during the generation phase.
    """
    import asyncio

    from lev.tasks import load_task

    task = load_task(task_name, gen_per_sample=1)
    engine = make_capture_engine()

    async def _run():
        try:
            await asyncio.wait_for(task.evaluate_async(engine, limit=n), timeout=timeout_s)
        except asyncio.TimeoutError:
            log(f"[capture:{task_name}] evaluate_async timed out after {timeout_s}s "
                f"(prompts already captured: {len(engine.captured)})")
        except Exception as e:  # noqa: BLE001 -- scoring may raise; prompts already captured
            log(f"[capture:{task_name}] evaluate_async raised (non-fatal, "
                f"captured {len(engine.captured)}): {type(e).__name__}: {e}")

    asyncio.run(_run())
    return engine.captured[:n]


def capture_main() -> int:
    prompts_json = os.environ.get("PROMPTS_JSON", DEFAULT_PROMPTS_JSON)
    n_per_task = int(os.environ.get("N_PER_TASK", "20"))
    capture_timeout = float(os.environ.get("CAPTURE_TIMEOUT_S", "600"))
    tasks = resolve_tasks()

    log("=" * 70)
    log("STAGE 1: CAPTURE (lev + datasets; no vllm)")
    log(f"PROMPTS_JSON={prompts_json}")
    log(f"N_PER_TASK={n_per_task}  seed(lev)=220302  tasks={tasks}")
    log("=" * 70)

    patch_gpt_judge()

    out: dict[str, list[list[dict]]] = {}
    for name in tasks:
        t0 = time.perf_counter()
        try:
            convs = capture_task_prompts(name, n_per_task, capture_timeout)
        except Exception as e:  # noqa: BLE001 -- loader/dataset error -> skip task
            log(f"[capture:{name}] FAILED to load/capture ({type(e).__name__}: {e}); skipping.")
            traceback.print_exc()
            continue
        dt = time.perf_counter() - t0
        if not convs:
            log(f"[capture:{name}] captured 0 prompts ({dt:.1f}s); skipping.")
            continue
        out[name] = convs
        log(f"[capture:{name}] captured {len(convs)} prompts ({dt:.1f}s).")

    if not out:
        log("[fatal] no task produced prompts; nothing written.")
        return 1

    Path(prompts_json).parent.mkdir(parents=True, exist_ok=True)
    with open(prompts_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    total = sum(len(v) for v in out.values())
    log(f"\n[capture] wrote {len(out)} tasks / {total} prompts -> {prompts_json}")
    log("CAPTURE_DONE tasks=" + ",".join(out.keys()))
    return 0


# =========================================================================== #
# STAGE 2: RUN (imports vllm only; NO lev / datasets)
# =========================================================================== #

def snapshot_spec(llm) -> tuple[float, float, list[float]]:
    """Return (num_drafts, num_accepted_tokens, per_pos_accepted[]) cumulative."""
    drafts = accepted = 0.0
    per_pos: list[float] = []
    try:
        metrics = llm.get_metrics()
    except Exception as e:  # noqa: BLE001
        log(f"[warn] get_metrics() failed: {e}")
        return 0.0, 0.0, []
    for m in metrics:
        name = getattr(m, "name", "")
        if name == "vllm:spec_decode_num_drafts":
            drafts += float(getattr(m, "value", 0.0))
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
    return drafts, accepted, per_pos


def measure_task_batch1(llm, sampling, convs: list[list[dict]]) -> tuple[int, int, float]:
    """Run a task's conversations ONE AT A TIME (batch=1, single-stream latency).

    Returns (n_ok, total_output_tokens, total_elapsed_s). A prompt that errors
    (e.g. over-length) is skipped-and-logged; it never aborts the task.
    Acceptance length is batch-invariant, so batch=1 is a valid regime for it and
    is the single-stream latency regime we want tokens/s in.
    """
    n_ok = 0
    total_out = 0
    elapsed = 0.0
    for i, c in enumerate(convs):
        try:
            t0 = time.perf_counter()
            outs = llm.chat([c], sampling_params=sampling, use_tqdm=False)
            dt = time.perf_counter() - t0
            total_out += len(outs[0].outputs[0].token_ids)
            elapsed += dt
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            log(f"    [skip prompt {i}: {type(e).__name__}: {e}]")
    return n_ok, total_out, elapsed


def resolve_ckpt() -> str:
    explicit = os.environ.get("CKPT_DRAFT", "").strip()
    if explicit:
        return explicit
    base = Path(os.environ.get("CKPT_BASE", "/t1data/users/kmswin7/speculators/dspark_ckpt_3n"))
    int_dirs = []
    if base.is_dir():
        for p in base.iterdir():
            if p.is_dir() and p.name.isdigit():
                int_dirs.append((int(p.name), p))
    if not int_dirs:
        raise SystemExit(f"[fatal] no integer-named epoch dir under {base}; set CKPT_DRAFT explicitly.")
    int_dirs.sort()
    latest = int_dirs[-1][1]
    log(f"[ckpt] resolved latest/best draft checkpoint: {latest}")
    return str(latest)


def _default_results_path(mode: str) -> str:
    d = str(Path(os.environ.get("PROMPTS_JSON", DEFAULT_PROMPTS_JSON)).parent)
    return os.path.join(d, f"perbench_{mode}.json")


def _load_prompts() -> dict[str, list[list[dict]]]:
    prompts_json = os.environ.get("PROMPTS_JSON", DEFAULT_PROMPTS_JSON)
    if not Path(prompts_json).is_file():
        raise SystemExit(
            f"[fatal] prompts file not found: {prompts_json}\n"
            "        Run stage 1 first:  python measure_accept_perbench.py --capture "
            "(under lev/.venv, outside the container).")
    with open(prompts_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _ordered_tasks(task_prompts: dict) -> list[str]:
    filt = resolve_tasks()
    order = [t for t in SUITE_TASKS if t in task_prompts] + \
            [t for t in task_prompts if t not in SUITE_TASKS]
    if os.environ.get("TASKS", "").strip():
        return [t for t in order if t in filt]
    return order


def _build_engine(spec: bool):
    """Build ONE vLLM engine (base = no spec; spec = dspark). Imports vllm only."""
    from vllm import LLM

    target = os.environ.get("TARGET", "/t1data/users/kmswin7/A.X-K2-Preview-hf")
    tp = int(os.environ.get("TP", "8"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "32768"))
    gpu_mem = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
    enable_ep = os.environ.get("ENABLE_EP", "1") == "1"
    enforce_eager = os.environ.get("ENFORCE_EAGER", "0") == "1"
    seed = int(os.environ.get("SEED", "220302"))

    kw = dict(
        model=target,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_ep,
        disable_log_stats=False,  # REQUIRED for get_metrics() to be populated
        seed=seed,
    )
    if spec:
        ckpt = resolve_ckpt()
        num_spec = int(os.environ.get("NUM_SPEC_TOKENS", "5"))
        kw["speculative_config"] = {"model": ckpt, "method": "dspark", "num_speculative_tokens": num_spec}
        log(f"[engine:spec] verifier={target}")
        log(f"[engine:spec] speculative_config={kw['speculative_config']}")
    else:
        log(f"[engine:base] verifier={target} (no speculative_config)")
    log(f"[engine] TP={tp} max_model_len={max_model_len} gpu_mem={gpu_mem} "
        f"enforce_eager={enforce_eager} enable_ep={enable_ep} seed={seed}")
    return LLM(**kw)


def _sampling(name: str):
    from vllm import SamplingParams
    temperature = float(os.environ.get("TEMPERATURE", "0.0"))
    top_p = float(os.environ.get("TOP_P", "1.0"))
    override = os.environ.get("MAX_TOKENS", "").strip()
    mt = int(override) if override else 256  # single-stream latency regime -> modest cap
    seed = int(os.environ.get("SEED", "220302"))
    return SamplingParams(
        temperature=temperature,
        top_p=top_p if temperature > 0 else 1.0,
        max_tokens=mt,
        seed=seed,
    ), mt


def _warmup(llm) -> None:
    from vllm import SamplingParams
    try:
        llm.chat([[{"role": "user", "content": "안녕하세요"}]],
                 sampling_params=SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
    except Exception as e:  # noqa: BLE001
        log(f"[warn] warmup failed (continuing): {e}")


def run_engine(mode: str) -> int:
    """mode in {'base','spec'}: build one engine, batch=1 per-task, write results JSON."""
    assert mode in ("base", "spec")
    task_prompts = _load_prompts()
    tasks = _ordered_tasks(task_prompts)
    results_path = os.environ.get(f"RESULTS_{mode.upper()}", _default_results_path(mode))

    log("=" * 70)
    log(f"STAGE 2: RUN mode={mode} (vllm only; prompts from JSON; batch=1)")
    log(f"PROMPTS_JSON tasks={len(task_prompts)} prompts={sum(len(v) for v in task_prompts.values())}")
    log(f"RESULTS -> {results_path}")
    log(f"tasks={tasks}")
    log("=" * 70)

    llm = _build_engine(spec=(mode == "spec"))
    _warmup(llm)

    results: dict[str, dict] = {}
    for name in tasks:
        convs = task_prompts[name]
        try:
            sampling, mt = _sampling(name)
            if mode == "spec":
                d0, a0, p0 = snapshot_spec(llm)
            n_ok, out_tok, elapsed = measure_task_batch1(llm, sampling, convs)
            tps = (out_tok / elapsed) if elapsed > 0 else 0.0
            rec = {"n": n_ok, "out_tokens": out_tok, "elapsed_s": round(elapsed, 3),
                   "tps": round(tps, 3), "max_tokens": mt}
            if mode == "spec":
                d1, a1, p1 = snapshot_spec(llm)
                dd, da = d1 - d0, a1 - a0
                if dd > 0:
                    rec["accept_len"] = 1.0 + da / dd
                    L = max(len(p1), len(p0))
                    p0f = p0 + [0.0] * (L - len(p0))
                    p1f = p1 + [0.0] * (L - len(p1))
                    rec["per_pos"] = [(p1f[k] - p0f[k]) / dd for k in range(L)]
                    rec["d_drafts"] = dd
                    rec["d_accepted"] = da
                else:
                    log(f"[{name}] WARNING no drafts recorded (Δdrafts={dd}); accept_len unavailable.")
                    rec["accept_len"] = None
                    rec["per_pos"] = []
            results[name] = rec
            extra = ""
            if mode == "spec" and rec.get("accept_len") is not None:
                extra = f"  accept_len={rec['accept_len']:.4f}"
            log(f"[{mode}:{name}] N={n_ok}  tps={tps:.2f}  out_tok={out_tok}  "
                f"elapsed={elapsed:.1f}s  max_tokens={mt}{extra}")
        except Exception as e:  # noqa: BLE001 -- one task must not abort the run
            log(f"[{mode}:{name}] ERROR ({type(e).__name__}: {e}); skipping.")
            traceback.print_exc()
            continue

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    log(f"[{mode}] wrote {len(results)} task results -> {results_path}")
    log(f"RUN_DONE mode={mode}")
    return 0


def run_combine() -> int:
    """Read base + spec results JSONs and print the combined per-task table."""
    base_path = os.environ.get("RESULTS_BASE", _default_results_path("base"))
    spec_path = os.environ.get("RESULTS_SPEC", _default_results_path("spec"))
    base = json.load(open(base_path, encoding="utf-8")) if Path(base_path).is_file() else {}
    spec = json.load(open(spec_path, encoding="utf-8")) if Path(spec_path).is_file() else {}
    if not base:
        log(f"[warn] no base results at {base_path}")
    if not spec:
        log(f"[warn] no spec results at {spec_path}")

    order = [t for t in SUITE_TASKS if t in base or t in spec]
    order += [t for t in list(base) + list(spec) if t not in order]

    log("\n================ PER-BENCH: BASE vs SPEC (batch=1) ================")
    log(f"{'TASK':<16} {'N':>3}  {'ACCEPT_LEN':>10}  {'BASE_TPS':>9}  {'SPEC_TPS':>9}  {'SPEEDUP':>8}")
    log("-" * 68)
    speedups: list[float] = []
    accepts: list[float] = []
    for name in order:
        b = base.get(name, {})
        s = spec.get(name, {})
        n = s.get("n", b.get("n", 0))
        al = s.get("accept_len")
        bt = b.get("tps")
        st = s.get("tps")
        sp = (st / bt) if (bt and st and bt > 0) else None
        if al is not None:
            accepts.append(al)
        if sp is not None:
            speedups.append(sp)
        log(f"{name:<16} {n:>3}  "
            f"{(f'{al:.4f}' if al is not None else 'NA'):>10}  "
            f"{(f'{bt:.2f}' if bt is not None else 'NA'):>9}  "
            f"{(f'{st:.2f}' if st is not None else 'NA'):>9}  "
            f"{(f'{sp:.3f}' if sp is not None else 'NA'):>8}")
    log("-" * 68)
    mean_al = (sum(accepts) / len(accepts)) if accepts else None
    mean_sp = (sum(speedups) / len(speedups)) if speedups else None
    log(f"{'MEAN':<16} {'':>3}  "
        f"{(f'{mean_al:.4f}' if mean_al is not None else 'NA'):>10}  "
        f"{'':>9}  {'':>9}  "
        f"{(f'{mean_sp:.3f}' if mean_sp is not None else 'NA'):>8}")
    log("==================================================================")

    # machine-greppable
    for name in order:
        b = base.get(name, {})
        s = spec.get(name, {})
        al = s.get("accept_len")
        bt = b.get("tps")
        st = s.get("tps")
        sp = (st / bt) if (bt and st and bt > 0) else None
        log(f"PERBENCH_{name}_ACCEPTLEN=" + (f"{al:.4f}" if al is not None else "NA"))
        log(f"PERBENCH_{name}_BASETPS=" + (f"{bt:.3f}" if bt is not None else "NA"))
        log(f"PERBENCH_{name}_SPECTPS=" + (f"{st:.3f}" if st is not None else "NA"))
        log(f"PERBENCH_{name}_SPEEDUP=" + (f"{sp:.3f}" if sp is not None else "NA"))
        if s.get("per_pos"):
            log(f"PERBENCH_{name}_PER_POS=" + ",".join(f"{x:.4f}" for x in s["per_pos"]))
        log(f"PERBENCH_{name}_N={s.get('n', b.get('n', 0))}")
    if mean_al is not None:
        log(f"PERBENCH_MACRO_MEAN_ACCEPTLEN={mean_al:.4f}")
    if mean_sp is not None:
        log(f"PERBENCH_MACRO_MEAN_SPEEDUP={mean_sp:.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-benchmark DSpark base-vs-spec accept_len + throughput harness.")
    ap.add_argument("--capture", action="store_true",
                    help="Stage 1 (host): capture per-task prompts to PROMPTS_JSON (needs lev+datasets, no vllm).")
    ap.add_argument("--mode", choices=["base", "spec"], default=None,
                    help="Stage 2 (container): build ONE engine and measure batch=1 tps (+accept_len for spec).")
    ap.add_argument("--combine", action="store_true",
                    help="Stage 3: merge base+spec result JSONs into the final per-task table.")
    args = ap.parse_args()

    if args.capture or os.environ.get("PERBENCH_CAPTURE_ONLY", "") == "1":
        return capture_main()
    if args.combine:
        return run_combine()
    if args.mode:
        return run_engine(args.mode)
    # No flag: combine if results exist, else guide the caller.
    if Path(os.environ.get("RESULTS_BASE", _default_results_path("base"))).is_file() or \
       Path(os.environ.get("RESULTS_SPEC", _default_results_path("spec"))).is_file():
        return run_combine()
    log("[fatal] nothing to do. Use one of: --capture | --mode base | --mode spec | --combine")
    return 2


if __name__ == "__main__":
    sys.exit(main())
