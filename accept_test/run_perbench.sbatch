#!/bin/bash
#SBATCH --job-name=dspark-perbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=03:00:00
#SBATCH --nodelist=compute-gpu-st-distributed-ml-9
#SBATCH --output=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#SBATCH --error=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#
# Per-benchmark-task mean acceptance-length for the trained DSpark drafter on
# A.X-K2, over the SAME 14-task lev suite we evaluated.
#
# TWO STAGES (the baked vLLM container has NO datasets/lev):
#   Stage 1 (host, NO container, NO GPU): lev/.venv captures the per-task
#           prompts to perbench_prompts.json.  lev needs datasets + py3.11+;
#           speculators_venv is py3.10 and cannot import lev (typing.Self).
#   Stage 2 (srun in baked container): reads the JSON and runs the vLLM
#           metric-delta measurement -- imports ONLY vllm, never datasets/lev.
#
# Override node:   sbatch --nodelist=compute-gpu-st-distributed-ml-10 run_perbench.sbatch
# Override draft:  CKPT_DRAFT=/path/to/epochN sbatch run_perbench.sbatch
# Override target: TARGET=/t1data/users/kmswin7/A.X-K2 sbatch run_perbench.sbatch
# One task only:   TASKS=gpqa sbatch run_perbench.sbatch
# Reuse existing prompts JSON (skip capture): SKIP_CAPTURE=1 sbatch run_perbench.sbatch
#
# NOTE: keep to the free node range 9-14 (see MEMORY: SLURM node range 9-14 only).

set -euo pipefail

IMG=/t1data/users/kmswin7/vllm-dspark-axk2.sqsh
HERE=/t1data/users/kmswin7/speculators/accept_test
# Host interpreter that has datasets + lev (py3.14). NOT speculators_venv (py3.10).
CAP_PY=/t1data/users/kmswin7/lev/.venv/bin/python
mkdir -p "$HERE/logs"

# ---- knobs (all overridable from the environment) ----
export TARGET="${TARGET:-/t1data/users/kmswin7/A.X-K2-Preview-hf}"
export CKPT_DRAFT="${CKPT_DRAFT:-}"                       # empty => auto-resolve highest epoch dir
export CKPT_BASE="${CKPT_BASE:-/t1data/users/kmswin7/speculators/dspark_ckpt_3n}"
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"           # = block_size (sample_from_anchor=True)
export TP="${TP:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"           # big enough for hle/aa-lcr long ctx; raise if truncating
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export ENABLE_EP="${ENABLE_EP:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export N_PER_TASK="${N_PER_TASK:-20}"
export SEED="${SEED:-220302}"
export TEMPERATURE="${TEMPERATURE:-0.0}"                 # 0=greedy (default). lev suite used 0.9/0.95.
export TOP_P="${TOP_P:-1.0}"
export MAX_TOKENS="${MAX_TOKENS:-}"                      # empty => per-task caps in the .py
export TASKS="${TASKS:-}"                                # empty => full 14-task suite
export CAPTURE_TIMEOUT_S="${CAPTURE_TIMEOUT_S:-600}"
export PROMPTS_JSON="${PROMPTS_JSON:-$HERE/perbench_prompts.json}"
export MODE="${MODE:-both}"
SKIP_CAPTURE="${SKIP_CAPTURE:-0}"

# ---- HF dataset cache (capture step loads HF datasets) ----
export HF_HOME="${HF_HOME:-/t1data/users/kmswin7/.cache/huggingface}"

echo "TARGET=$TARGET"
echo "CKPT_DRAFT=${CKPT_DRAFT:-<auto>}  CKPT_BASE=$CKPT_BASE"
echo "NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS  TP=$TP  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "N_PER_TASK=$N_PER_TASK  SEED=$SEED  TEMPERATURE=$TEMPERATURE  TOP_P=$TOP_P  TASKS=${TASKS:-<all>}"
echo "PROMPTS_JSON=$PROMPTS_JSON  HF_HOME=$HF_HOME"

# =====================================================================
# STAGE 1 -- CAPTURE on the host (no container, no GPU). Needs datasets+lev.
# =====================================================================
if [ "$SKIP_CAPTURE" != "1" ] || [ ! -f "$PROMPTS_JSON" ]; then
  echo "################## STAGE 1: CAPTURE (host, lev/.venv) ##################"
  # HF_TOKEN for gated gpqa/kmmlu-pro comes from .creds; sourced quietly, never echoed.
  [ -f /t1data/users/kmswin7/serving/.creds ] && source /t1data/users/kmswin7/serving/.creds >/dev/null 2>&1 || true
  env \
    HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" no_proxy='*' NO_PROXY='*' \
    ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
    PYTHONPATH=/t1data/users/kmswin7/lev/src:/t1data/users/kmswin7/speculators/src \
    PROMPTS_JSON="$PROMPTS_JSON" N_PER_TASK="$N_PER_TASK" SEED="$SEED" \
    TASKS="$TASKS" CAPTURE_TIMEOUT_S="$CAPTURE_TIMEOUT_S" \
    "$CAP_PY" "$HERE/measure_accept_perbench.py" --capture
else
  echo "################## STAGE 1 SKIPPED (reusing $PROMPTS_JSON) ##################"
fi

if [ ! -f "$PROMPTS_JSON" ]; then
  echo "[fatal] capture produced no prompts file ($PROMPTS_JSON); aborting before GPU run."
  exit 1
fi

# =====================================================================
# STAGE 2 -- RUN inside the baked container (vllm only, reads the JSON).
# =====================================================================
echo "################## STAGE 2: RUN (container, vllm only) ##################"
# Two engine loads in ONE job (node-scarce): BASE then SPEC, as separate python
# processes so each ~400GB engine loads and frees cleanly (no double in-process
# load). Then --combine merges into the per-task table. Each stays robust: a
# task that errors is skipped-and-logged inside the .py.
srun --container-image="$IMG" \
  --container-mounts="/t1data:/t1data" \
  bash -lc "
    set -x
    # ---- same engine env flags as run_accept_test_baked.sbatch ----
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    # A.X-K2 (DeepSeek-V3-style MLA + DSA) verifier attention backend.
    export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE
    export PYTHONDONTWRITEBYTECODE=1

    # speculators package importable (config classes for the dspark checkpoint
    # parse). lev is NOT needed here -- prompts come from the JSON.
    export PYTHONPATH=/t1data/users/kmswin7/speculators/src:\${PYTHONPATH:-}
    export PROMPTS_JSON='$PROMPTS_JSON'
    export TARGET='$TARGET' CKPT_DRAFT='${CKPT_DRAFT}' CKPT_BASE='$CKPT_BASE'
    export NUM_SPEC_TOKENS='$NUM_SPEC_TOKENS' TP='$TP' MAX_MODEL_LEN='$MAX_MODEL_LEN'
    export GPU_MEM_UTIL='$GPU_MEM_UTIL' ENABLE_EP='$ENABLE_EP' ENFORCE_EAGER='$ENFORCE_EAGER'
    export SEED='$SEED' TEMPERATURE='$TEMPERATURE' TOP_P='$TOP_P' MAX_TOKENS='$MAX_TOKENS'
    export TASKS='$TASKS'
    export RESULTS_BASE='$HERE/perbench_base.json' RESULTS_SPEC='$HERE/perbench_spec.json'

    cd $HERE
    MODE='${MODE:-both}'
    if [ \"\$MODE\" = base ] || [ \"\$MODE\" = both ]; then
      echo '######## BASE engine (plain A.X-K2, batch=1) ########'
      python3 measure_accept_perbench.py --mode base 2>&1 | tee /tmp/perbench_base.out
    fi
    if [ \"\$MODE\" = spec ] || [ \"\$MODE\" = both ]; then
      echo '######## SPEC engine (dspark, batch=1) ########'
      python3 measure_accept_perbench.py --mode spec 2>&1 | tee /tmp/perbench_spec.out
    fi
    if [ \"\$MODE\" = both ] || [ \"\$MODE\" = combine ]; then
      echo '######## COMBINE ########'
      python3 measure_accept_perbench.py --combine 2>&1 | tee /tmp/perbench.out
    fi

    echo '==================== GREP SUMMARY ===================='
    grep -E '^PERBENCH_' /tmp/perbench.out || true
    echo '======================================================'
  "

echo "DONE. Log: $HERE/logs/${SLURM_JOB_ID:-manual}.log"
echo "Per-task results JSON: $HERE/perbench_base.json  $HERE/perbench_spec.json"
