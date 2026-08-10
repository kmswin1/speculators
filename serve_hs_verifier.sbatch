#!/bin/bash
#SBATCH --job-name=axk2-hs-verif
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=2-00:00:00
#SBATCH --output=/t1data/users/kmswin7/serving/logs/spec/%j.hsverif.log
#SBATCH --error=/t1data/users/kmswin7/serving/logs/spec/%j.hsverif.log
#
# Hidden-state extraction verifier for speculator training.
# Serves FP8 A.X-K2 with:
#   --speculative_config (method=extract_hidden_states, eagle_aux_hidden_state_layer_ids)
#   --kv_transfer_config (ExampleHiddenStatesConnector, kv_producer -> shared storage on /t1data)
# HS_DIR must be on shared FS (/t1data) so the training node can read it.
CKPT=/t1data/users/kmswin7/A.X-K2_fused_yarn2
IMG=/t1data/users/kmswin7/vllm-v0.23.0-for-sktax.sqsh
PORT=8202
HS_DIR=${HS_DIR:-/t1data/users/kmswin7/speculators/hs_shared}
LAYERS=${LAYERS:-"[58, 59, 60]"}
mkdir -p "$HS_DIR" /t1data/users/kmswin7/serving/logs/spec
SPEC_CFG="{\"method\": \"extract_hidden_states\", \"num_speculative_tokens\": 1, \"draft_model_config\": {\"hf_config\": {\"eagle_aux_hidden_state_layer_ids\": $LAYERS}}}"
KV_CFG="{\"kv_connector\": \"ExampleHiddenStatesConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"shared_storage_path\": \"$HS_DIR\"}}"

srun --container-image="$IMG" --container-mounts=/t1data:/t1data \
  bash -lc "
    set -x
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    vllm serve $CKPT \
      --served-model-name /t1data/users/kmswin7/A.X-K2_fused_yarn2 \
      --host 0.0.0.0 --port $PORT \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --trust-remote-code \
      --speculative_config '$SPEC_CFG' \
      --kv_transfer_config '$KV_CFG' \
      --no-enable-chunked-prefill \
      --max-model-len 8448 \
      --max-num-seqs ${MAXSEQ:-4} \
      --gpu-memory-utilization 0.90 &
    SPID=\$!
    for i in \$(seq 1 900); do
      curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { echo HS_VERIFIER_READY; break; }
      kill -0 \$SPID 2>/dev/null || { echo HS_VERIFIER_DIED; break; }
      sleep 5
    done
    echo HS_SERVE_ALIVE
    wait \$SPID
  "
