#!/bin/bash
#SBATCH --job-name=axk2-colo-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=2-00:00:00
#SBATCH --output=/t1data/users/kmswin7/serving/logs/spec/%j.colo.log
#SBATCH --error=/t1data/users/kmswin7/serving/logs/spec/%j.colo.log
#
# CO-LOCATED online DSpark training for FP8 A.X-K2 on ONE node.
#   verifier (vLLM, extract_hidden_states) + trainer (torchrun) share the SAME node.
#   HS connector writes to /dev/shm (2TB tmpfs, RAM) -> local flock, NO Lustre-flock
#   hot-path stall -> NO sample_tokens deadlock. Trainer reads /dev/shm locally.
#   --on-generate delete -> each HS deleted right after use -> ZERO disk accumulation.
#   => online, no NFS, no 2TB cache. max-num-seqs can be raised freely (batch was
#      never the crash cause).
# Env: SPEC_TYPE(dspark) MAXSEQ MEMUTIL MAXSTEPS EPOCHS SAVE_PATH EXTRA
set -x
SPEC=/t1data/users/kmswin7/speculators
IMG=/t1data/users/kmswin7/vllm-v0.23.0-for-sktax.sqsh
CKPT=/t1data/users/kmswin7/A.X-K2_fused_yarn2
PORT=8202
SHM=/dev/shm/hs_col
SAVE_PATH=${SAVE_PATH:-$SPEC/dspark_ckpt}
LAYERS="[2, 30, 58, 60]"
SPEC_CFG="{\"method\": \"extract_hidden_states\", \"num_speculative_tokens\": 1, \"draft_model_config\": {\"hf_config\": {\"eagle_aux_hidden_state_layer_ids\": $LAYERS}}}"
KV_CFG="{\"kv_connector\": \"ExampleHiddenStatesConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"shared_storage_path\": \"$SHM\"}}"

OVERLAY=/t1data/users/kmswin7/speculators/vllm_overlay/example_hidden_states_connector.py
CONNPATH=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
PY310=/t1data/users/kmswin7/py310_include
srun --container-image="$IMG" --container-mounts=/t1data:/t1data,/dev/shm:/dev/shm,$OVERLAY:$CONNPATH,$PY310/python3.10:/usr/include/python3.10,$PY310/x86_64-python3.10:/usr/include/x86_64-linux-gnu/python3.10 \
  bash -lc "
    set -x
    rm -rf $SHM; mkdir -p $SHM $SAVE_PATH/logs
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export no_proxy='*' NO_PROXY='*'
    # --- verifier (background) ---
    vllm serve $CKPT --served-model-name $CKPT --host 0.0.0.0 --port $PORT \
      --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
      --speculative_config '$SPEC_CFG' --kv_transfer_config '$KV_CFG' \
      --no-enable-chunked-prefill --no-enable-prefix-caching --max-model-len 8448 \
      --max-num-seqs ${MAXSEQ:-8} --gpu-memory-utilization ${MEMUTIL:-0.72} &
    SPID=\$!
    for i in \$(seq 1 900); do
      curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { echo COLO_VERIFIER_READY; break; }
      kill -0 \$SPID 2>/dev/null || { echo COLO_VERIFIER_DIED; exit 1; }
      sleep 5
    done
    # --- trainer (foreground, same node, reads /dev/shm) ---
    cd $SPEC && source speculators_venv/bin/activate
    # container loopback (localhost/127.0.0.1) mis-routes -> use node hostname (regen's proven way);
    # kill proxy + force eager (inductor dynamic-reshape crash on the draft)
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    NODE=\$(hostname)
    export no_proxy=\"\$NODE,127.0.0.1,localhost\" NO_PROXY=\"\$NODE,127.0.0.1,localhost\"
    export
    torchrun --standalone --nproc_per_node 8 scripts/train.py \
      --speculator-type ${SPEC_TYPE:-dspark} \
      --draft-arch qwen3 --num-layers 3 --draft-vocab-size 32768 \
      --verifier-name-or-path $CKPT --trust-remote-code \
      --data-path $SPEC/output_combined \
      --total-seq-len 8192 \
      --target-layer-ids 2 30 58 \
      --block-size 5 --markov-rank 256 --sliding-window 128 \
      --vllm-endpoint http://\$NODE:$PORT/v1 \
      --hidden-states-backend file --hidden-states-path $SHM \
      --on-missing generate --on-generate delete \
      --epochs ${EPOCHS:-10} --save-best \
      --save-path $SAVE_PATH \
      --logger '' --log-dir $SAVE_PATH/logs \
      --request-timeout 1800 --max-retries 8 \
      ${MAXSTEPS:+--max-steps \$MAXSTEPS} \
      ${EXTRA:-}
    echo \"COLO_TRAIN_DONE rc=\$? \$(date '+%F %T')\"
    kill \$SPID 2>/dev/null
  "
