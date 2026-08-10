# Per-benchmark-task acceptance length (DSpark drafter on A.X-K2)

Comprehensive per-task table for the trained DSpark speculator (final checkpoint
`dspark_ckpt_3n/3` = epoch3_step2256) vs the A.X-K2 verifier, over the SAME
14-task `lev` suite (`results/suite_fp8_256k`, `suite_nvfp4_256k`,
`suite_ptq_calib_256k`), sampling `N_PER_TASK` (default **30**) prompts per task.
For each task it reports **mean acceptance length**, **base tokens/s**, **spec
tokens/s** and **speedup**, all at **batch=1 (single-stream latency)**.

Files:
- `measure_accept_perbench.py` — driver with modes `--capture` (host) / `--mode base` / `--mode spec` / `--combine`.
- `run_perbench.sbatch` — orchestrates: host capture, then in one GPU job base engine → spec engine → combine.
- `perbench_prompts.json` — captured prompts (produced by stage 1; ready-to-use copy present).
- `perbench_base.json`, `perbench_spec.json` — per-task results written by stage 2 (base / spec).
- this README.

## Design (why staged + two engine loads)

The baked vLLM container (`vllm-dspark-axk2.sqsh`) has **no `datasets` / no `lev`**
(the first run failed with `ModuleNotFoundError: No module named 'datasets'`), and
GPU nodes are scarce (often only one free). So:

- **Stage 1 — CAPTURE (host, no container, no GPU).** `lev/.venv` + `lev` + `datasets`
  build the exact per-task prompts and dump them to `perbench_prompts.json`.
  Must use **`lev/.venv` (Python 3.14)** — `speculators_venv` is Python 3.10 and
  cannot import `lev` (it uses `typing.Self`, 3.11+).
- **Stage 2 — RUN (one baked-container GPU job).** Two engine loads back-to-back
  as **separate python processes** — `--mode base` (plain A.X-K2) then `--mode spec`
  (dspark) — so each ~400 GB engine loads and frees cleanly without a double
  in-process load. Then `--combine` merges the two result JSONs. **Imports only
  `vllm`** — never `lev`/`datasets` (verified with a hard import-blocker: all of
  base/spec/combine run with `lev`/`datasets` imports forbidden).

## The one command (run once training is cut)

```bash
sbatch /t1data/users/kmswin7/speculators/accept_test/run_perbench.sbatch
```
The sbatch runs stage 1 on the host (capture, if JSON missing / `SKIP_CAPTURE` unset),
then inside one `srun` container job runs `--mode base`, `--mode spec`, `--combine`.
To run the stages by hand:
```bash
# Stage 1 (host) — capture prompts (already done at N=30; JSON is present):
PYTHONPATH=/t1data/users/kmswin7/lev/src \
HF_HOME=/t1data/users/kmswin7/.cache/huggingface HF_HUB_OFFLINE=1 N_PER_TASK=30 \
  source /t1data/users/kmswin7/serving/.creds && \
  /t1data/users/kmswin7/lev/.venv/bin/python \
    /t1data/users/kmswin7/speculators/accept_test/measure_accept_perbench.py --capture
# Stage 2 (container): base engine, then spec engine, then combine (what the sbatch does).
#   python measure_accept_perbench.py --mode base
#   python measure_accept_perbench.py --mode spec
#   python measure_accept_perbench.py --combine
# Reuse an existing prompts JSON and skip stage 1:  SKIP_CAPTURE=1 sbatch run_perbench.sbatch
```

Common overrides:
```bash
# pin a node in the free 9-14 range
sbatch --nodelist=compute-gpu-st-distributed-ml-10 run_perbench.sbatch
# explicit draft checkpoint (default: highest-numbered epoch dir under dspark_ckpt_3n)
CKPT_DRAFT=/t1data/users/kmswin7/speculators/dspark_ckpt_3n/1 sbatch run_perbench.sbatch
# use the fresh skt/A.X-K2 instead of the Preview build
TARGET=/t1data/users/kmswin7/A.X-K2 sbatch run_perbench.sbatch
# one task only / a subset
TASKS=gpqa sbatch run_perbench.sbatch
# match the lev suite's sampling instead of greedy
TEMPERATURE=0.9 TOP_P=0.95 sbatch run_perbench.sbatch
```

Output: `accept_test/logs/<jobid>.log`. `--combine` prints the table

```
TASK | N | ACCEPT_LEN | BASE_TPS | SPEC_TPS | SPEEDUP        (+ a MEAN row)
```
(SPEEDUP = spec_tps / base_tps, both batch=1 single-stream tokens/s) and these
machine-greppable lines per task:
```
PERBENCH_<task>_ACCEPTLEN=<1 + Δaccepted/Δdrafts>
PERBENCH_<task>_BASETPS=<base tokens/s, batch=1>
PERBENCH_<task>_SPECTPS=<spec tokens/s, batch=1>
PERBENCH_<task>_SPEEDUP=<spec_tps/base_tps>
PERBENCH_<task>_PER_POS=<p0,p1,...>            # per-position acceptance
PERBENCH_<task>_N=<n prompts measured>
PERBENCH_MACRO_MEAN_ACCEPTLEN=<mean over tasks>
PERBENCH_MACRO_MEAN_SPEEDUP=<mean over tasks>
```

## `perbench_prompts.json` structure

```json
{
  "<task_name>": [
    [ {"role": "system", "content": "..."}, {"role": "user", "content": "..."} ],
    [ {"role": "user", "content": "..."} ],
    ...
  ],
  ...
}
```
Each task maps to its list of sampled conversations; each conversation is the
exact `messages` list lev builds (system+user for hle/kobalt/aa-omni-public,
user-only for the rest). Current file (regenerated at **N_PER_TASK=30**):
**14 tasks / 392 prompts** — 30 each except `apex2025`=12 and `kmo26`=20, whose
datasets have fewer rows (tasks with <30 capture all). Override the path with
`PROMPTS_JSON=`.

## The 14 lev tasks measured (registry name -> dataset -> prompt shape)

| # | lev task | HF dataset (loaded at task init) | prompt (messages) | gated |
|---|----------|----------------------------------|-------------------|-------|
| 1 | `aa-lcr` | `ArtificialAnalysis/AA-LCR` + zip `extracted_text/AA-LCR_extracted-text.zip` | user: multi-doc long-context QA | no |
| 2 | `aa-omni-public` | `ArtificialAnalysis/AA-Omniscience-Public` | system(domain/topic)+user | no |
| 3 | `aime26` | `math-ai/aime26` | user: boxed-answer math CoT | no |
| 4 | `apex-shortlist` | `MathArena/apex-shortlist` | user: boxed-answer math | no |
| 5 | `apex2025` | `MathArena/apex_2025` (**only 12 samples** → captures 12) | user: boxed-answer math | no |
| 6 | `click` | `EunsuKim/CLIcK` | user: Korean MC (정답: X) | no |
| 7 | `gpqa` | `Idavidrein/gpqa` cfg `gpqa_diamond` | user: MC (Answer: $LETTER) | **yes (HF_TOKEN + terms)** |
| 8 | `hle` | `cais/hle` (text-only rows: `image==""`) | system(format)+user | no |
| 9 | `ifbench` | `allenai/IFBench_test` | user: instruction following | no |
| 10 | `kmmlu-pro` | `LGAI-EXAONE/KMMLU-Pro` | user: Korean MC (정답:) | **yes (HF_TOKEN)** |
| 11 | `kmo26` | `skt/kmo26` | user: Korean math (**답**: boxed) | no |
| 12 | `kobalt` | `snunlp/KoBALT-700` split `raw` | system+user: Korean 10-way MC | no |
| 13 | `livecodebench` | `livecodebench/code_generation_lite` (`test6.jsonl`) | user: code generation | no |
| 14 | `scicode` | `skt/scicode-tests` + scicode data | user: multi-step scientific code | no |

Registry names match the suite output dir names exactly. (Not in this 14: the
vision tasks babyvision/charxiv/mmmu/zerobench, and mmlu-pro which appears in
some suite dirs but not the 256k suite set — add via `TASKS=` if wanted.)

## How prompts are sampled per task

The harness does **not** re-implement each task's formatting. It drives each
task's real `evaluate_async(engine, limit=N)` with a `CaptureEngine` that records
the `messages` lev builds (correct system prompt, per-task template, choices,
etc.) and returns an empty generation. Sampling uses lev's own
`Task.samples_to_use(limit=N)`, whose default seed is **220302** — the same fixed
seed as our HLE→500 / GPQA→100 subsample policy — so the 30 prompts are a
deterministic subset. If a task has < 30 samples, all are used (apex2025→12, kmo26→20).

Capture is **stage 1** (host, CPU only, before any GPU/engine). Judge-based
tasks (hle, aa-lcr, apex*, aa-omni-public) build an OpenAI "gpt" judge engine in
`__init__` that normally does a live network health check and needs
`OPENAI_API_KEY`; the harness monkeypatches `GPTEngine` to a no-network dummy so
they load and capture fully offline. The judge is only used during *scoring*,
which we don't need.

## How per-task acceptance is computed

In `--mode spec`, `llm.get_metrics()` spec-decode counters are cumulative over the
engine's life. For each task we snapshot before and after that task's prompts and
take the delta:
```
mean_accept_len   = 1 + Δ vllm:spec_decode_num_accepted_tokens / Δ vllm:spec_decode_num_drafts
per_pos_accept[k] = Δ vllm:spec_decode_num_accepted_tokens_per_pos[k] / Δ num_drafts
```
Acceptance length is a per-draft property → batch-invariant → measuring it at
batch=1 (which is what we want for single-stream tokens/s) is still valid.
`disable_log_stats=False` is set so the counters are populated (offline `LLM`
defaults it to `True`, which returns an empty snapshot).

**Throughput (both engines)** is per-task batch=1: each prompt runs alone via
`llm.chat([conv])`, timed individually; `tps = Σ output_tokens / Σ elapsed`.
`base_tps` from the plain engine, `spec_tps` from the dspark engine,
`speedup = spec_tps / base_tps`.

## Engine configuration (mirrors run_accept_test_baked.sbatch)

- baked container `vllm-dspark-axk2.sqsh` (dspark built in), `--container-mounts=/t1data:/t1data`, **no overlay**
- BASE load: plain A.X-K2 (no `speculative_config`). SPEC load: `speculative_config={model: <draft>, method: "dspark", num_speculative_tokens: 5}`.
- default draft = `dspark_ckpt_3n/3` (auto-resolved highest epoch dir; final = epoch3_step2256)
- TP8, `trust_remote_code`, `enable_expert_parallel`, `disable_log_stats=False`
- `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE` (A.X-K2 MLA+DSA verifier)
- default `MAX_MODEL_LEN=32768`, `GPU_MEM_UTIL=0.90`
- default sampling **greedy** (`TEMPERATURE=0.0`), **`MAX_TOKENS=256`** (single-stream latency regime; 14 tasks × ~30 prompts × 2 engines batch=1 is already long). Override with `MAX_TOKENS=`, `TEMPERATURE=`, `TOP_P=`.

## Datasets & network — only the HOST capture step touches HF

Datasets are needed **only in stage 1 (host)**. Every task calls
`load_dataset(...)` / `hf_hub_download` / `snapshot_download` at construction.
The capture step runs with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` against
`HF_HOME=/t1data/users/kmswin7/.cache/huggingface` — the cache the lev suite
populated. **All 14 datasets were verified present and captured cleanly offline
(392 prompts written at N=30)** including the gated gpqa + kmmlu-pro, whose `HF_TOKEN`
lives in `/t1data/users/kmswin7/serving/.creds` (sourced quietly by the sbatch,
never echoed). **Stage 2 in the container needs no datasets and no network** —
it only reads the JSON and the local model/checkpoint dirs.

If you move to a different node / clear the cache, pre-cache online first:
```bash
HF_HUB_OFFLINE=0 HF_HOME=/t1data/users/kmswin7/.cache/huggingface \
  source /t1data/users/kmswin7/serving/.creds && \
  PYTHONPATH=/t1data/users/kmswin7/lev/src \
  /t1data/users/kmswin7/lev/.venv/bin/python -c "
from lev.tasks import load_task
for t in ['aa-lcr','aa-omni-public','aime26','apex-shortlist','apex2025','click','gpqa','hle','ifbench','kmmlu-pro','kmo26','kobalt','livecodebench','scicode']:
    try: load_task(t, gen_per_sample=1); print('cached', t)
    except Exception as e: print('FAIL', t, e)"
```
(or set `HF_HUB_OFFLINE=0` on the sbatch if the run node itself has network).

## UNVERIFIED assumptions (could not run GPU / vLLM here)

1. **Engine build + `llm.chat` + `get_metrics()` deltas were NOT executed on real
   GPU** here. They reuse the exact, previously-working pattern from
   `measure_accept.py` (same counter names, `.values` for the per-pos vector,
   `disable_log_stats=False`). What *was* verified: (a) stage-1 capture for all 14
   tasks (392 prompts at N=30); (b) stage-2 `--mode base` / `--mode spec` /
   `--combine` end-to-end with a stubbed vLLM — batch=1 timing, the metric-delta
   math (`1 + Δacc/Δdrafts`, per-pos), and the combined table — under a hard
   import-blocker proving they import **only vllm**, never `lev`/`datasets`.
   The two ~400GB engine loads happen as separate processes; whether vLLM frees
   GPU memory fully between the base and spec python processes within one job is
   assumed (standard process-exit teardown) but not GPU-verified.
2. **Chat template fidelity**: lev sent raw `messages` to the OpenAI-compatible
   vLLM server (which applied A.X-K2's `chat_template.jinja`, `add_generation_prompt=True`,
   `skip_special_tokens=False`). This harness calls `llm.chat(messages, ...)`,
   which applies the same tokenizer chat template offline. Assumed equivalent; if
   A.X-K2's template has a non-defaulted `enable_thinking`-style variable that the
   server set differently, rendering could differ slightly.
3. **`method="dspark"` is available in the baked container** (`vllm-dspark-axk2.sqsh`,
   from-source, dspark baked). If the engine rejects `dspark`, this fails the same
   way `measure_accept.py` would — check the container, not this script.
4. **`MAX_MODEL_LEN=32768` may be too short for some `aa-lcr` / `hle` prompts.**
   Over-length prompts don't abort the task: batch=1 runs each prompt alone and
   skips only the offenders (logged), so the task's numbers come from the prompts
   that fit. Raise `MAX_MODEL_LEN` (the suite ran 256k) to include them.
5. **Sampling**: default is greedy (`TEMPERATURE=0.0`) for a deterministic
   headline acceptance-length number. The lev suite used `temperature=0.9,
   top_p=0.95`; acceptance length depends on the verifier's sampling, so set
   `TEMPERATURE=0.9 TOP_P=0.95` to reproduce the suite's regime.
6. **Draft checkpoint auto-resolution** picks the highest integer-named dir under
   `dspark_ckpt_3n` — now `3` (== `epoch3_step2256 -> 3`, the final checkpoint).
   Pass `CKPT_DRAFT` to override.
