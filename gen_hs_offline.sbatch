#!/bin/bash
#SBATCH --job-name=axk2-hs-offline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --output=/t1data/users/kmswin7/serving/logs/spec/%j.hsoffline.log
#SBATCH --error=/t1data/users/kmswin7/serving/logs/spec/%j.hsoffline.log
#
# Offline hidden-state generation for one shard (RANK of WORLD_SIZE).
# Brings up its OWN HS verifier (extract_hidden_states + connector, layers 58/59/60)
# then runs data_generation_offline.py at low concurrency (crash-safe) to cache HS to disk.
# Env: RANK WORLD_SIZE CONC PREP OUT_HS  (defaults below)
set -x
SPEC=/t1data/users/kmswin7/speculators
IMG=/t1data/users/kmswin7/vllm-v0.23.0-for-sktax.sqsh
CKPT=/t1data/users/kmswin7/A.X-K2_fused_yarn2
PORT=8202
RANK=${RANK:-0}; WORLD_SIZE=${WORLD_SIZE:-1}; CONC=${CONC:-4}
PREP=${PREP:-$SPEC/output_combined}
OUT_HS=${OUT_HS:-$SPEC/hs_offline}
HS_DIR=/tmp/hs_conn_$RANK   # connector scratch (node-local, ephemeral)
LAYERS="[58, 59, 60]"
SPEC_CFG="{\"method\": \"extract_hidden_states\", \"num_speculative_tokens\": 1, \"draft_model_config\": {\"hf_config\": {\"eagle_aux_hidden_state_layer_ids\": $LAYERS}}}"
KV_CFG="{\"kv_connector\": \"ExampleHiddenStatesConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"shared_storage_path\": \"$HS_DIR\"}}"
mkdir -p "$OUT_HS" "$HS_DIR"

srun --container-image="$IMG" --container-mounts=/t1data:/t1data \
  bash -lc "
    set -x
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    vllm serve $CKPT --served-model-name $CKPT --host 0.0.0.0 --port $PORT \
      --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
      --speculative_config '$SPEC_CFG' --kv_transfer_config '$KV_CFG' \
      --no-enable-chunked-prefill --max-model-len 8448 --max-num-seqs ${CONC} --gpu-memory-utilization 0.90 &
    SPID=\$!
    for i in \$(seq 1 900); do curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { echo HS_VERIFIER_READY; break; }; kill -0 \$SPID 2>/dev/null || { echo VERIFIER_DIED; exit 1; }; sleep 5; done
    cd $SPEC
    ./speculators_venv/bin/python scripts/data_generation_offline.py \
      --model $CKPT --endpoint http://localhost:$PORT/v1 \
      --preprocessed-data $PREP --output $OUT_HS \
      --concurrency $CONC --request-timeout 1800 --max-retries 6 \
      ${MAXS:+--max-samples $MAXS} \
      --world-size $WORLD_SIZE --rank $RANK
    echo \"HS_OFFLINE_DONE rank=$RANK rc=\$?\"
    kill \$SPID 2>/dev/null
  "
