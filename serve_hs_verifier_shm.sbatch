#!/bin/bash
#SBATCH --job-name=axk2-hsv-shm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=2-00:00:00
#SBATCH --output=/t1data/users/kmswin7/serving/logs/spec/%j.hsvshm.log
#SBATCH --error=/t1data/users/kmswin7/serving/logs/spec/%j.hsvshm.log
# Persistent HS-extract verifier: /dev/shm connector + single-writer overlay patch.
# Trainer attaches separately via srun --overlap (cheap restarts, no reload).
CKPT=/t1data/users/kmswin7/A.X-K2_fused_yarn2
IMG=/t1data/users/kmswin7/vllm-v0.23.0-for-sktax.sqsh
PORT=8202; SHM=/dev/shm/hs_col
OVERLAY=/t1data/users/kmswin7/speculators/vllm_overlay/example_hidden_states_connector.py
CONNPATH=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
LAYERS="[58, 59, 60]"
SPEC_CFG="{\"method\": \"extract_hidden_states\", \"num_speculative_tokens\": 1, \"draft_model_config\": {\"hf_config\": {\"eagle_aux_hidden_state_layer_ids\": $LAYERS}}}"
KV_CFG="{\"kv_connector\": \"ExampleHiddenStatesConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"shared_storage_path\": \"$SHM\"}}"
srun --container-image="$IMG" --container-mounts=/t1data:/t1data,/dev/shm:/dev/shm,$OVERLAY:$CONNPATH \
  bash -lc "
    set -x; rm -rf $SHM; mkdir -p $SHM
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    vllm serve $CKPT --served-model-name $CKPT --host 0.0.0.0 --port $PORT \
      --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
      --speculative_config '$SPEC_CFG' --kv_transfer_config '$KV_CFG' \
      --no-enable-chunked-prefill --max-model-len 8448 \
      --max-num-seqs ${MAXSEQ:-8} --gpu-memory-utilization ${MEMUTIL:-0.72} &
    SPID=\$!
    for i in \$(seq 1 1000); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo HSV_SHM_READY; break; }; kill -0 \$SPID 2>/dev/null || { echo HSV_DIED; exit 1; }; sleep 5; done
    echo HSV_SERVE_ALIVE; wait \$SPID
  "
