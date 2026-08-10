#!/bin/bash
#SBATCH --job-name=dspark-accept-baked
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --nodelist=compute-gpu-st-distributed-ml-9
#SBATCH --output=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#SBATCH --error=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#
# TRUE DSpark acceptance-length measurement (method="dspark", NO dflash fallback).
#
# Runs the sktax vLLM container with the DSpark overlay bind-mounted in
# (see ../vllm_overlay_dspark/MOUNTS.md), so the container gains method="dspark"
# support: a DSparkProposer that applies the Markov logit-bias while sampling
# the block left-to-right, on top of the unchanged DFlash path. A.X-K2
# (AXK2ForCausalLM) verifier support is untouched.
#
# OFFLINE, single node, TP8: baseline (no spec) then spec, prints
# MEAN_ACCEPT_LEN and tokens/s, then exits (no lingering server).
#
# Override node:   sbatch --nodelist=compute-gpu-st-distributed-ml-10 run_accept_test_dspark.sbatch
# Override draft:  CKPT_DRAFT=/path/to/epochN sbatch run_accept_test_dspark.sbatch
#
# NOTE: nodes 9,10,11 are the free range (12/13/14 busy training). Keep to 9-11.

set -euo pipefail

IMG=/t1data/users/kmswin7/vllm-dspark-axk2.sqsh
HERE=/t1data/users/kmswin7/speculators/accept_test
OVL=/t1data/users/kmswin7/speculators/vllm_overlay_dspark/files
DIST=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p "$HERE/logs"

# ---- knobs (all overridable from the environment) ----
export TARGET="${TARGET:-/t1data/users/kmswin7/A.X-K2_fused_yarn2}"
export CKPT_DRAFT="${CKPT_DRAFT:-/t1data/users/kmswin7/speculators/dspark_ckpt_3n/0}"
export SPEC_METHOD="${SPEC_METHOD:-dspark}"          # TRUE dspark (needs this overlay)
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"       # = block_size (sample_from_anchor=True)
export TP="${TP:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export ENABLE_EP="${ENABLE_EP:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export NUM_PROMPTS="${NUM_PROMPTS:-48}"
export BATCH1="${BATCH1:-0}"                          # 1 = single-stream latency regime
export ALLOW_DFLASH_FALLBACK=0                       # never fall back — true dspark only

# ---- DSpark overlay bind-mounts (host file -> container dist-packages file) ----
# Mirrors ../vllm_overlay_dspark/MOUNTS.md and the connector overlay pattern in
# train_colocated_5node.sbatch.
OVERLAY_MOUNTS="\
$OVL/vllm/transformers_utils/configs/speculators/algos.py:$DIST/transformers_utils/configs/speculators/algos.py,\
$OVL/vllm/config/speculative.py:$DIST/config/speculative.py,\
$OVL/vllm/v1/worker/gpu_model_runner.py:$DIST/v1/worker/gpu_model_runner.py,\
$OVL/vllm/v1/spec_decode/llm_base_proposer.py:$DIST/v1/spec_decode/llm_base_proposer.py,\
$OVL/vllm/v1/spec_decode/dflash.py:$DIST/v1/spec_decode/dflash.py,\
$OVL/vllm/v1/spec_decode/dspark.py:$DIST/v1/spec_decode/dspark.py,\
$OVL/vllm/model_executor/models/registry.py:$DIST/model_executor/models/registry.py,\
$OVL/vllm/model_executor/models/qwen3_dspark.py:$DIST/model_executor/models/qwen3_dspark.py"

echo "TARGET=$TARGET"
echo "CKPT_DRAFT=$CKPT_DRAFT"
echo "SPEC_METHOD=$SPEC_METHOD  NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS  TP=$TP"
echo "OVERLAY_MOUNTS=$OVERLAY_MOUNTS"

srun --container-image="$IMG" \
  --container-mounts="/t1data:/t1data,/t1data/users/kmswin7/speculators/vllm_overlay_dspark/mla_fix/model_arch_config_convertor.py:/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/model_arch_config_convertor.py" \
  bash -lc "
    set -x
    # ---- same env flags as serve_fp8_gen.sbatch ----
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    # A.X-K2 (DeepSeek-V3-style MLA + DSA) verifier attention backend.
    # UNVERIFIED: the DFlash/DSpark draft submodel may need a non-causal backend
    # (e.g. FLASH_ATTN); if engine init complains about the draft's attention
    # backend, try unsetting VLLM_ATTENTION_BACKEND or setting FLASH_ATTN.
    export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE
    # Don't try to write .pyc next to the read-only mounted overlay modules.
    export PYTHONDONTWRITEBYTECODE=1

    # speculators package importable (config classes for the checkpoint parse).
    export PYTHONPATH=/t1data/users/kmswin7/speculators/src:\${PYTHONPATH:-}

    cd $HERE

    echo '################## BASELINE (no spec) ##################'
    MODE=base python3 measure_accept.py --mode base 2>&1 | tee /tmp/base.out

    echo '################## SPEC (dspark) #######################'
    MODE=spec python3 measure_accept.py --mode spec 2>&1 | tee /tmp/spec.out

    echo '==================== SUMMARY ===================='
    BASE_TPS=\$(grep -m1 '^OUTPUT_TPS_BASE=' /tmp/base.out | cut -d= -f2 || echo 0)
    SPEC_TPS=\$(grep -m1 '^OUTPUT_TPS_SPEC=' /tmp/spec.out | cut -d= -f2 || echo 0)
    MAL=\$(grep -m1 '^MEAN_ACCEPT_LEN=' /tmp/spec.out | cut -d= -f2 || echo NA)
    echo \"MEAN_ACCEPT_LEN=\$MAL\"
    echo \"OUTPUT_TPS_BASE=\$BASE_TPS\"
    echo \"OUTPUT_TPS_SPEC=\$SPEC_TPS\"
    if [ \"\$BASE_TPS\" != \"0\" ] && [ -n \"\$SPEC_TPS\" ]; then
      python3 -c \"b=float('\$BASE_TPS'); s=float('\$SPEC_TPS'); print(f'SPEEDUP_SPEC_OVER_BASE={s/b:.3f}') if b>0 else print('SPEEDUP_SPEC_OVER_BASE=NA')\"
    fi
    grep '^PER_POSITION_ACCEPTANCE=' /tmp/spec.out || true
    echo '================================================='
  "

echo "DONE. Log: $HERE/logs/${SLURM_JOB_ID:-manual}.log"
