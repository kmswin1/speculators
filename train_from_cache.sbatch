#!/bin/bash
#SBATCH --job-name=axk2-train-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --output=/t1data/users/kmswin7/serving/logs/spec/%j.traincache.log
#SBATCH --error=/t1data/users/kmswin7/serving/logs/spec/%j.traincache.log
#
# Train DSpark speculator for FP8 A.X-K2 from a STATIC hidden-state cache.
# NO live verifier: --on-missing skip means samples without a cached HS file are
# skipped and the OpenAI client is never even constructed (see data.py _setup_client,
# only called inside _maybe_generate_hs on the "generate" path). => cannot crash on
# the DSA sample_tokens NFS-flock deadlock. Deterministic loss.
# Env: SPEC_TYPE (default dspark)  SAVE_PATH  MAXSTEPS  EXTRA
set -x
cd /t1data/users/kmswin7/speculators
source speculators_venv/bin/activate
export no_proxy='*' NO_PROXY='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VERIFIER=/t1data/users/kmswin7/A.X-K2_fused_yarn2
DATA=/t1data/users/kmswin7/speculators/output_combined
HS=$DATA/hidden_states
SAVE_PATH=${SAVE_PATH:-/t1data/users/kmswin7/speculators/dspark_ckpt}
mkdir -p "$SAVE_PATH/logs"

torchrun --standalone --nproc_per_node 8 scripts/train.py \
  --speculator-type "${SPEC_TYPE:-dspark}" \
  --draft-arch qwen3 --num-layers 3 \
  --verifier-name-or-path "$VERIFIER" --trust-remote-code \
  --data-path "$DATA" \
  --total-seq-len 8192 \
  --target-layer-ids 58 59 60 \
  --block-size 5 --markov-rank 256 --sliding-window 128 \
  --vllm-endpoint "http://localhost:1/v1" \
  --hidden-states-backend file \
  --hidden-states-path "$HS" \
  --on-missing skip \
  --epochs "${EPOCHS:-10}" --save-best \
  --save-path "$SAVE_PATH" \
  --logger tensorboard --log-dir "$SAVE_PATH/logs" \
  ${MAXSTEPS:+--max-steps $MAXSTEPS} \
  ${EXTRA:-}
echo "TRAIN_CACHE_DONE rc=$? $(date '+%F %T')"
