#!/bin/bash
#SBATCH --job-name=dspark-accept
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --nodelist=compute-gpu-st-distributed-ml-9
#SBATCH --output=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#SBATCH --error=/t1data/users/kmswin7/speculators/accept_test/logs/%j.log
#
# Measure speculative-decoding acceptance length of a DSpark draft against the
# A.X-K2 verifier, OFFLINE (single node, TP8), in the sktax vLLM container.
# Runs baseline (no spec) then spec, prints MEAN_ACCEPT_LEN and tokens/s for
# both, then exits. No lingering server.
#
# Override node:   sbatch --nodelist=compute-gpu-st-distributed-ml-10 run_accept_test.sbatch
# Override draft:  CKPT_DRAFT=/path/to/epochN sbatch run_accept_test.sbatch
#
# NOTE: nodes 9,10,11 are the free range (12/13/14 busy training). Keep to 9-11.
#
# !!!! READ README_findings.md FIRST !!!!
# The container (vllm 0.23.0+sktax) does NOT support method="dspark". With the
# default SPEC_METHOD=dspark this job will fail at engine init with a
# "Expected one of: [eagle3, peagle, dflash]" / "Unsupported speculative method"
# error. To get a number on THIS container today, run with the DFlash fallback:
#     SPEC_METHOD=dflash ALLOW_DFLASH_FALLBACK=1 sbatch run_accept_test.sbatch
# That measures the DFlash BACKBONE ONLY (drops DSpark's Markov + confidence
# heads) -> a lower bound, NOT true DSpark acceptance. For true DSpark you need a
# vLLM that registers the dspark method (see README_findings.md).

set -euo pipefail

IMG=/t1data/users/kmswin7/vllm-v0.23.0-for-sktax.sqsh
HERE=/t1data/users/kmswin7/speculators/accept_test
mkdir -p "$HERE/logs"

# ---- knobs (all overridable from the environment) ----
export TARGET="${TARGET:-/t1data/users/kmswin7/A.X-K2_fused_yarn2}"
export CKPT_DRAFT="${CKPT_DRAFT:-/t1data/users/kmswin7/speculators/dspark_ckpt_3n/0}"
export SPEC_METHOD="${SPEC_METHOD:-dspark}"          # dspark (needs capable vLLM) | dflash (fallback)
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"       # = block_size (sample_from_anchor=True)
export TP="${TP:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export ENABLE_EP="${ENABLE_EP:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export NUM_PROMPTS="${NUM_PROMPTS:-48}"
export ALLOW_DFLASH_FALLBACK="${ALLOW_DFLASH_FALLBACK:-0}"

echo "TARGET=$TARGET"
echo "CKPT_DRAFT=$CKPT_DRAFT"
echo "SPEC_METHOD=$SPEC_METHOD  NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS  TP=$TP"

srun --container-image="$IMG" --container-mounts=/t1data:/t1data \
  bash -lc "
    set -x
    # ---- same env flags as serve_fp8_gen.sbatch ----
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    # A.X-K2 (DeepSeek-V3-style MLA + DSA) verifier attention backend.
    # UNVERIFIED: the DFlash/DSpark draft submodel may require a non-causal
    # backend (e.g. FLASH_ATTN); if engine init complains about the draft's
    # attention backend, try unsetting VLLM_ATTENTION_BACKEND or setting it to
    # FLASH_ATTN for the draft. Confirm at run time.
    export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE

    # Make the speculators package importable (config classes are only needed
    # by vLLM if it must parse the speculators checkpoint; harmless otherwise).
    export PYTHONPATH=/t1data/users/kmswin7/speculators/src:\${PYTHONPATH:-}

    cd $HERE

    echo '################## BASELINE (no spec) ##################'
    MODE=base python3 measure_accept.py --mode base 2>&1 | tee /tmp/base.out

    echo '################## SPEC ($SPEC_METHOD) #################'
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
