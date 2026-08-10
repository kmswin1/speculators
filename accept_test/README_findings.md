# DSpark acceptance-length measurement — findings

Research date: 2026-08-08. Scope: how to load the freshly-trained DSpark draft +
A.X-K2 verifier in the sktax vLLM container and measure speculative-decoding
**acceptance length**. Nothing was executed on GPU; these are read-only findings
plus ready-to-run scripts (`measure_accept.py`, `run_accept_test.sbatch`).

---

## TL;DR

| Question | Answer |
|---|---|
| `method` string for DSpark | **`"dspark"`** (per speculators; `speculators_model_type` is copied verbatim into vLLM's `method`). |
| Convert/export step needed? | **No.** Speculators-trained checkpoints are already in speculators format (config.json carries `speculators_config`); vLLM consumes the checkpoint dir directly. `speculators.convert` is only for *external* eagle3/mtp/dflash checkpoints. |
| `num_speculative_tokens` | **5** (= `block_size`, because `sample_from_anchor=True`). The checkpoint's `speculators_config.proposal_methods[0].speculative_tokens` already encodes 5; a dspark-aware vLLM auto-infers it. |
| d2t/t2d shipping | **Baked into the checkpoint** `model.safetensors` as persistent buffers `t2d` / `d2t`. The `output_combined/*.npy` files are training-time source only; nothing extra needs to ship alongside for inference. |
| How acceptance length is surfaced | vLLM V1 Prometheus counters: `vllm:spec_decode_num_drafts`, `…_num_draft_tokens`, `…_num_accepted_tokens`, and the per-position vector `…_num_accepted_tokens_per_pos`. Offline: `llm.get_metrics()`. **Mean acceptance length = 1 + num_accepted_tokens / num_drafts.** |
| **BLOCKER** | The target container **does not support `method="dspark"`** — see below. |

---

## 1. How vLLM consumes a speculators checkpoint (and the exact invocation)

A speculators-trained checkpoint is directly runnable — no conversion. vLLM reads
`speculators_config` from `config.json`
(`vllm/transformers_utils/configs/speculators/base.py`), maps
`speculators_model_type` → `method`, and reads
`proposal_methods[0].speculative_tokens` → `num_speculative_tokens`.

The **intended** invocation (on a dspark-capable vLLM), per
`docs/user_guide/algorithms/dspark.md`:

```bash
# short form (auto-reads everything from config.json)
vllm serve /path/to/dspark_ckpt/0
# long form / explicit target override (what our offline driver uses):
vllm serve /t1data/users/kmswin7/A.X-K2_fused_yarn2 \
  --served-model-name A.X-K2 --trust-remote-code -tp 8 \
  --speculative-config '{"model":"/t1data/users/kmswin7/speculators/dspark_ckpt_3n/0","method":"dspark","num_speculative_tokens":5}'
```

The checkpoint's config.json (written by `save_pretrained`; schema from
`src/speculators/models/dspark/config.py` + inherited `dflash/config.py`) contains,
at minimum:

```
speculators_model_type = "dspark"
architectures          = ["DSparkSpeculator"]
draft_vocab_size       = 32768
block_size             = 5
sample_from_anchor     = true
markov_rank            = 256, markov_head_type = "vanilla"
enable_confidence_head = true, confidence_head_with_markov = true
aux_hidden_state_layer_ids = [2, 30, 58]      # == training --target-layer-ids
mask_token_id          = <int>
transformer_layer_config = { qwen3 …, vocab_size = 163840, hidden_size = 7168 }
speculators_config = { algorithm:"dspark",
                       proposal_methods:[{proposal_type:"greedy", speculative_tokens:5}],
                       default_proposal_method:"greedy",
                       verifier:{ name_or_path, architectures } }
```

---

## 2. Reduced draft vocab (d2t / t2d) at inference

`src/speculators/model.py::DraftVocabMixin._init_vocab` registers `t2d`
(bool, len = verifier vocab 163840) and `d2t` (long, len = draft vocab 32768) as
**buffers**. `register_buffer` defaults to `persistent=True`, so both are written
into `model.safetensors` when the checkpoint is saved. **Therefore the mapping
travels inside the checkpoint** — you do NOT need to copy
`output_combined/{d2t,t2d}.npy` next to the checkpoint for inference. (Those .npy
files are only consumed by the trainer, via `--data-path`, to build the buffers.)

`d2t` is an offset table: `target_id = draft_idx + d2t[draft_idx]`; `t2d` is a
boolean mask over the verifier vocab marking the 32768 kept tokens. A
dspark/dflash-aware vLLM applies this remap when projecting draft logits back to
the verifier vocabulary.

---

## 3. How acceptance length is measured (offline, preferred)

vLLM V1 exposes spec-decode stats as in-memory Prometheus metrics. The repo's own
`scripts/evaluate/perf_utils.py` already parses these (server `/metrics` path);
our driver uses the equivalent **offline** API `llm.get_metrics()`
(`vllm/entrypoints/llm.py`, returning `Counter`/`Vector` objects from
`vllm/v1/metrics/reader.py`). Metric names (confirmed present in the container's
`vllm/v1/spec_decode/metrics.py`):

```
vllm:spec_decode_num_drafts                    (Counter)
vllm:spec_decode_num_draft_tokens              (Counter)
vllm:spec_decode_num_accepted_tokens           (Counter)
vllm:spec_decode_num_accepted_tokens_per_pos   (Vector, one counter per position)
```

Derived (vLLM convention, bonus token included):

```
mean_acceptance_length      = 1 + num_accepted_tokens / num_drafts
per_position_acceptance[k]   = accepted_per_pos[k] / num_drafts
draft_acceptance_rate        = num_accepted_tokens / num_draft_tokens
```

`measure_accept.py` runs ~48 fixed prompts (Korean + English + short reasoning)
with greedy sampling, reads these counters, and prints
`MEAN_ACCEPT_LEN=<x>`, per-position rates, and tokens/s. `run_accept_test.sbatch`
runs baseline (no spec) then spec and reports the speedup. Offline `LLM()` was
chosen (over serve + `/metrics`) because it is self-contained and tears down on
process exit — no lingering server.

---

## 4. ⚠️ BLOCKER — the container does not know "dspark"

**Verified** by extracting files from `vllm-v0.23.0-for-sktax.sqsh`:

* Version: `vllm 0.23.0+sktax` (commit `g91df0fad4`).
* `vllm/transformers_utils/configs/speculators/algos.py` registers **only**
  `eagle3`, `peagle`, `dflash`. There is no `dspark` entry.
* `vllm/v1/spec_decode/` has `dflash.py` (`DFlashProposer`) but **no dspark
  proposer**; `vllm/model_executor/models/` has `qwen3_dflash.py` but no dspark
  model. `vllm/config/speculative.py` hard-codes the method set
  (`DFlashModelTypes = Literal["dflash"]`, etc.) with no dspark.
* The newer on-disk source trees `vllm-axk2-src` / `vllm-axk2-build` (Aug 7, newer
  than the Jul 22 container) *also* lack dspark — so a rebuild from those wouldn't
  add it either.
* Speculators ships **no vLLM plugin** (`pyproject.toml` has only a console-script
  entry point; nothing registers a model into `vllm.ModelRegistry`). So dspark
  support must come from vLLM itself.

**Consequence:** loading the checkpoint with `method="dspark"` raises
`ValueError: Expected one of: [eagle3, peagle, dflash]` (or, via the explicit
path, `NotImplementedError: Unsupported speculative method: 'dspark'`) at engine
init. **True DSpark acceptance cannot be measured on this container as-is.**

### Options

**(A) DFlash-backbone fallback — get a number today (LOWER BOUND, not true DSpark).**
DSpark subclasses DFlash; the Markov + confidence heads are purely additive.
`measure_accept.py::make_dflash_view()` writes a temp copy of the checkpoint dir
with `config.json` patched to `speculators_model_type="dflash"`,
`architectures=["DFlashDraftModel"]` and symlinks the rest (the config already has
the DFlash-required `aux_hidden_state_layer_ids` and `mask_token_id`). vLLM then
loads it through the working DFlash path; the extra `markov_head.*` /
`confidence_head.*` tensors are ignored. Run:

```bash
SPEC_METHOD=dflash ALLOW_DFLASH_FALLBACK=1 sbatch accept_test/run_accept_test.sbatch
```

This reports DFlash-only acceptance (no Markov/confidence contribution), so it is
a **floor** for what DSpark should achieve. *UNVERIFIED:* that the DFlash loader
tolerates the extra tensors without erroring, and that DSpark-trained weights
behave under the DFlash inference path — confirm at run time.

**(B) Real DSpark support — port it into vLLM.** Add a `register_speculator("dspark")`
updater (mirroring `update_dflash`), a `DSparkProposer` (subclass `DFlashProposer`
applying the Markov logit bias + confidence gating), and a dspark model, then
mount that patched vLLM into the container. This is a genuine upstream port and
was **not** authored here (cannot be verified without the upstream reference).
The DSpark paper/method is upstream in mainline vLLM per
`docs/user_guide/algorithms/dspark.md`; using a mainline vLLM build that already
registers `dspark` is the clean path if the container can be swapped.

---

## 5. Other gotchas

* **`trust_remote_code`** is required (A.X-K2 = custom `AXK2ForCausalLM`). Set in
  the driver and sbatch.
* **Attention backend.** Verifier uses `FLASHINFER_MLA_SPARSE` (DeepSeek-V3 MLA +
  DSA). `vllm/config/speculative.py` notes *"DFlash needs a non-causal-capable
  backend like FLASH_ATTN"* for the draft. Whether the draft submodel can use a
  different backend than the MLA-sparse target is **UNVERIFIED**; if engine init
  complains about the draft's attention, try `FLASH_ATTN` or unset
  `VLLM_ATTENTION_BACKEND`. `parallel_drafting=True` is auto-set for dflash.
* **`block_size` ↔ `num_speculative_tokens`.** `speculative_tokens = block_size`
  when `sample_from_anchor=True` (this checkpoint), else `block_size − 1`. Here
  block_size=5 → 5 speculative tokens.
* **aux layer indexing.** vLLM's `update_dflash` shifts:
  `dflash_config.target_layer_ids = [i−1 for i in aux_hidden_state_layer_ids]`
  (vLLM #40727). The checkpoint stores `aux_hidden_state_layer_ids=[2,30,58]`
  (matching training `--target-layer-ids 2 30 58`); vLLM derives `[1,29,57]`
  internally. No action needed, just be aware if debugging layer mismatches.
* **A.X-K2 256K concurrency crash** (from memory): irrelevant here — this is a
  single offline job with `max_model_len=8192`, `max_num_seqs` small.
* **Node range:** default `--nodelist=compute-gpu-st-distributed-ml-9`; keep to
  9/10/11 (12/13/14 busy training). Override with `sbatch --nodelist=…-10`.
* **Checkpoint dir:** default `dspark_ckpt_3n/0`. At research time no epoch dir
  existed yet (training in progress) — `config.json` schema above is derived from
  the code, not read from a saved file, so double-check field names against the
  actual saved `config.json` once epoch 0 lands.

---

## 6. One command to run (once a checkpoint exists)

DFlash-backbone fallback (works on THIS container today; lower-bound number):

```bash
CKPT_DRAFT=/t1data/users/kmswin7/speculators/dspark_ckpt_3n/0 \
SPEC_METHOD=dflash ALLOW_DFLASH_FALLBACK=1 \
sbatch /t1data/users/kmswin7/speculators/accept_test/run_accept_test.sbatch
```

True DSpark (only on a vLLM that registers the `dspark` method):

```bash
CKPT_DRAFT=/t1data/users/kmswin7/speculators/dspark_ckpt_3n/0 \
sbatch /t1data/users/kmswin7/speculators/accept_test/run_accept_test.sbatch
```

Read `MEAN_ACCEPT_LEN=`, `OUTPUT_TPS_BASE=`, `OUTPUT_TPS_SPEC=`,
`SPEEDUP_SPEC_OVER_BASE=`, and `PER_POSITION_ACCEPTANCE=` from the job log
(`accept_test/logs/<jobid>.log`).
