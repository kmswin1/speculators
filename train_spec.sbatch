#!/bin/bash
#SBATCH --job-name=axk2-spec-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --output=/t1data/users/kmswin7/serving/logs/spec/%j.train.log
#SBATCH --error=/t1data/users/kmswin7/serving/logs/spec/%j.train.log
#
# Train a DFlash or DSpark speculator for FP8 A.X-K2 (full K2 vocab, no pruning).
# Reads hidden states from a shared-FS dir written by the HS verifier (kv connector),
# and triggers on-demand generation via the verifier's OpenAI endpoint.
# Env: SPEC_TYPE={dflash,dspark}  VERIF_EP  HS_DIR  SAVE_PATH  EXTRA (extra args)
set -x
cd /t1data/users/kmswin7/speculators
source speculators_venv/bin/activate
export no_proxy='*' NO_PROXY='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VERIFIER=/t1data/users/kmswin7/A.X-K2_fused_yarn2

torchrun --standalone --nproc_per_node 8 scripts/train.py \
  --speculator-type "${SPEC_TYPE:?}" \
  --draft-arch qwen3 --num-layers 3 \
  --verifier-name-or-path "$VERIFIER" --trust-remote-code \
  --data-path "${DATA_OVERRIDE:-/t1data/users/kmswin7/speculators/output}" \
  --total-seq-len 8192 \
  --target-layer-ids 58 59 60 \
  --block-size 5 --markov-rank 256 --sliding-window 128 \
  --vllm-endpoint "${VERIF_EP:?}" \
  --hidden-states-backend file \
  --hidden-states-path "${HS_DIR:?}" \
  --on-missing generate --on-generate delete \
  --epochs 10 --save-best \
  --save-path "${SAVE_PATH:?}" \
  --logger tensorboard --log-dir "${SAVE_PATH}/logs" \
  ${EXTRA:-}
echo "SPEC_TRAIN_DONE rc=$? $(date '+%F %T')"
