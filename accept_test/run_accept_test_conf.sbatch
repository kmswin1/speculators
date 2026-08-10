#!/bin/bash
#SBATCH --job-name=dspark-accept-conf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --nodelist=compute-gpu-st-distributed-ml-9
#SBATCH --output=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#SBATCH --error=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#
# A/B test: DSpark CONFIDENCE variant (confidence-based dynamic draft length).
#
# Copy of run_accept_test_baked.sbatch, but bind-mounts the _conf overlay's TWO
# changed files (v1/spec_decode/dspark.py, model_executor/models/qwen3_dspark.py)
# over the baked container's dist-packages, and sets DSPARK_CONF_THRESHOLD. The
# other 6 overlay files come from the validated fixed-5 overlay, so this is a
# clean A/B vs. run_accept_test_baked.sbatch (which mounts the same 8 targets,
# all from fixed-5). See ../vllm_overlay_dspark_conf/MOUNTS.md.
#
#   A (fixed-5):   sbatch run_accept_test_baked.sbatch
#   B (conf):      DSPARK_CONF_THRESHOLD=0.3 sbatch run_accept_test_conf.sbatch
#
# Both use the baked container /t1data/users/kmswin7/vllm-dspark-axk2.sqsh and
# verifier A.X-K2-Preview-hf. Confidence truncation returns ragged per-request
# drafts by default (true variable length -> real verifier savings). Set
# DSPARK_CONF_RECTANGULAR=1 for the no-savings sentinel-padding fallback.
#
# NOTE: nodes 9-14 only (1-8 shared). 9,10,11 are the free range.

set -euo pipefail

IMG=/t1data/users/kmswin7/vllm-dspark-axk2.sqsh
HERE=/t1data/users/kmswin7/speculators/accept_test
# The baked container already has the validated fixed-5 DSpark overlay baked in
# (run_accept_test_baked.sbatch mounts only /t1data). For the A/B we override
# ONLY the 2 conf-changed files on top of the baked dist-packages.
OVL_CONF=/t1data/users/kmswin7/speculators/vllm_overlay_dspark_conf/files # conf (2 files)
DIST=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p "$HERE/logs"

# ---- knobs (all overridable from the environment) ----
export TARGET="${TARGET:-/t1data/users/kmswin7/A.X-K2-Preview-hf}"
export CKPT_DRAFT="${CKPT_DRAFT:-/t1data/users/kmswin7/speculators/dspark_ckpt_3n/0}"
export SPEC_METHOD="${SPEC_METHOD:-dspark}"          # TRUE dspark (needs this overlay)
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"       # MAX block; conf may emit fewer
export TP="${TP:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export ENABLE_EP="${ENABLE_EP:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export NUM_PROMPTS="${NUM_PROMPTS:-48}"
export BATCH1="${BATCH1:-0}"                          # 1 = single-stream latency regime
export ALLOW_DFLASH_FALLBACK=0                       # never fall back — true dspark only

# ---- confidence knobs (the _conf variant) ----
export DSPARK_CONF_THRESHOLD="${DSPARK_CONF_THRESHOLD:-0.3}"
export DSPARK_CONF_RECTANGULAR="${DSPARK_CONF_RECTANGULAR:-0}"
export DISABLE_ASYNC_SCHED="${DISABLE_ASYNC_SCHED:-0}"

# ---- overlay bind-mounts: ONLY the 2 conf-changed files over the baked image ----
# (fixed-5's other 6 files are already baked into vllm-dspark-axk2.sqsh.)
OVERLAY_MOUNTS="\
$OVL_CONF/vllm/v1/spec_decode/dspark.py:$DIST/v1/spec_decode/dspark.py,\
$OVL_CONF/vllm/model_executor/models/qwen3_dspark.py:$DIST/model_executor/models/qwen3_dspark.py"

echo "TARGET=$TARGET"
echo "CKPT_DRAFT=$CKPT_DRAFT"
echo "SPEC_METHOD=$SPEC_METHOD  NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS  TP=$TP"
echo "DSPARK_CONF_THRESHOLD=$DSPARK_CONF_THRESHOLD  DSPARK_CONF_RECTANGULAR=$DSPARK_CONF_RECTANGULAR"
echo "OVERLAY_MOUNTS=$OVERLAY_MOUNTS"

srun --container-image="$IMG" \
  --container-mounts="/t1data:/t1data,$OVERLAY_MOUNTS" \
  bash -lc "
    set -x
    # ---- same env flags as run_accept_test_baked.sbatch ----
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    # A.X-K2 (DeepSeek-V3-style MLA + DSA) verifier attention backend.
    export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE
    # Don't try to write .pyc next to the read-only mounted overlay modules.
    export PYTHONDONTWRITEBYTECODE=1
    # Confidence knobs must reach the drafter process inside the container.
    export DSPARK_CONF_THRESHOLD=$DSPARK_CONF_THRESHOLD
    export DSPARK_CONF_RECTANGULAR=$DSPARK_CONF_RECTANGULAR
    export DISABLE_ASYNC_SCHED=${DISABLE_ASYNC_SCHED:-0}

    # speculators package importable (config classes for the checkpoint parse).
    export PYTHONPATH=/t1data/users/kmswin7/speculators/src:\${PYTHONPATH:-}

    cd $HERE

    echo '################## BASELINE (no spec) ##################'
    MODE=base python3 measure_accept.py --mode base 2>&1 | tee /tmp/base.out

    echo '################## SPEC (dspark + confidence) ##########'
    MODE=spec python3 measure_accept.py --mode spec 2>&1 | tee /tmp/spec.out

    echo '==================== SUMMARY ===================='
    BASE_TPS=\$(grep -m1 '^OUTPUT_TPS_BASE=' /tmp/base.out | cut -d= -f2 || echo 0)
    SPEC_TPS=\$(grep -m1 '^OUTPUT_TPS_SPEC=' /tmp/spec.out | cut -d= -f2 || echo 0)
    MAL=\$(grep -m1 '^MEAN_ACCEPT_LEN=' /tmp/spec.out | cut -d= -f2 || echo NA)
    echo \"DSPARK_CONF_THRESHOLD=$DSPARK_CONF_THRESHOLD\"
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
