#!/bin/bash
#SBATCH --job-name=dspark-conf-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --nodelist=compute-gpu-st-distributed-ml-9
#SBATCH --output=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#SBATCH --error=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#
# Confidence-threshold SWEEP for DSpark dynamic draft length.
# threshold is read once at proposer __init__ and TP8 multiproc workers can't be
# updated at runtime, so each threshold = a FRESH engine load (sequential loop,
# one node held for the whole sweep). threshold 0.0 disables truncation == fixed-5.
#
# Override node:   sbatch --nodelist=compute-gpu-st-distributed-ml-10 run_conf_sweep.sbatch
# Override ckpt:   CKPT_DRAFT=/path/to/epochN sbatch run_conf_sweep.sbatch
# Override sweep:  THRESHOLDS="0.0 0.2 0.4 0.6" sbatch run_conf_sweep.sbatch
# Latency regime:  BATCH1=1 sbatch run_conf_sweep.sbatch   (default 0 = batched throughput)

set -euo pipefail

IMG=/t1data/users/kmswin7/vllm-dspark-axk2.sqsh
HERE=/t1data/users/kmswin7/speculators/accept_test
OVL_CONF=/t1data/users/kmswin7/speculators/vllm_overlay_dspark_conf/files
DIST=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p "$HERE/logs"

export TARGET="${TARGET:-/t1data/users/kmswin7/A.X-K2-Preview-hf}"
export CKPT_DRAFT="${CKPT_DRAFT:-/t1data/users/kmswin7/speculators/dspark_ckpt_3n/0}"
export SPEC_METHOD="${SPEC_METHOD:-dspark}"
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"
export TP="${TP:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export ENABLE_EP="${ENABLE_EP:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export NUM_PROMPTS="${NUM_PROMPTS:-128}"     # batched throughput regime
export BATCH1="${BATCH1:-0}"                 # 0 = batched (throughput), 1 = single-stream
export ALLOW_DFLASH_FALLBACK=0
export DSPARK_CONF_RECTANGULAR="${DSPARK_CONF_RECTANGULAR:-0}"
# threshold 0.0 == fixed-5 (truncation off). Sweep includes it as the baseline.
export THRESHOLDS="${THRESHOLDS:-0.0 0.1 0.2 0.3 0.5 0.7}"
export DISABLE_ASYNC_SCHED="${DISABLE_ASYNC_SCHED:-1}"

OVERLAY_MOUNTS="\
$OVL_CONF/vllm/v1/spec_decode/dspark.py:$DIST/v1/spec_decode/dspark.py,\
$OVL_CONF/vllm/model_executor/models/qwen3_dspark.py:$DIST/model_executor/models/qwen3_dspark.py"

echo "SWEEP TARGET=$TARGET CKPT_DRAFT=$CKPT_DRAFT"
echo "THRESHOLDS=$THRESHOLDS  NUM_PROMPTS=$NUM_PROMPTS BATCH1=$BATCH1 MAX_TOKENS=$MAX_TOKENS"

srun --container-image="$IMG" \
  --container-mounts="/t1data:/t1data,$OVERLAY_MOUNTS" \
  bash -lc "
    set -x
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE
    export PYTHONDONTWRITEBYTECODE=1
    export DSPARK_CONF_RECTANGULAR=$DSPARK_CONF_RECTANGULAR
    export PYTHONPATH=/t1data/users/kmswin7/speculators/src:\${PYTHONPATH:-}
    export DISABLE_ASYNC_SCHED=$DISABLE_ASYNC_SCHED
    cd $HERE
    SUM=/tmp/sweep_summary.tsv
    echo -e 'threshold\tMEAN_ACCEPT_LEN\tOUTPUT_TPS_SPEC\tPER_POSITION' > \$SUM
    for t in $THRESHOLDS; do
      echo \"################## THRESHOLD=\$t ##################\"
      out=/tmp/sweep_\$t.out
      DSPARK_CONF_THRESHOLD=\$t MODE=spec python3 measure_accept.py --mode spec 2>&1 | tee \$out
      al=\$(grep -m1 '^MEAN_ACCEPT_LEN=' \$out | cut -d= -f2 || echo NA)
      tps=\$(grep -m1 '^OUTPUT_TPS_SPEC=' \$out | cut -d= -f2 || echo NA)
      pp=\$(grep -m1 '^PER_POSITION_ACCEPTANCE=' \$out | cut -d= -f2 || echo NA)
      echo -e \"\$t\t\$al\t\$tps\t\$pp\" >> \$SUM
    done
    echo '==================== SWEEP SUMMARY ===================='
    cat \$SUM
    echo '======================================================'
  "
