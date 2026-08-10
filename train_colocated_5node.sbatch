#!/bin/bash
#SBATCH --job-name=axk2-colo5
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=2-00:00:00
#SBATCH --output=/t1data/users/kmswin7/serving/logs/spec/%j.colo5.log
#SBATCH --error=/t1data/users/kmswin7/serving/logs/spec/%j.colo5.log
#
# N-NODE co-located DSpark training (multi-node DDP on the tiny draft) — carries the
# full verified single-node recipe (train_colocated.sbatch) to multi-node.
#   Each node: 1 vLLM verifier(TP8, extract_hidden_states, /dev/shm connector, patched)
#              + trainer ranks. Each rank hits ITS OWN node's verifier (node hostname)
#              and reads ITS OWN node-local /dev/shm -> HS never crosses nodes.
#   DistributedSampler shards data across all ranks; DDP syncs only the small draft grads.
# Env: MAXSEQ MEMUTIL MAXSTEPS EPOCHS SAVE_PATH
set -x
SPEC=/t1data/users/kmswin7/speculators
IMG=/t1data/users/kmswin7/vllm-v0.23.0-for-sktax.sqsh
CKPT=/t1data/users/kmswin7/A.X-K2_fused_yarn2
PORT=8202
SHM=/dev/shm/hs_col
SAVE_PATH=${SAVE_PATH:-$SPEC/dspark_ckpt_5n}
RDZV_DIR=$SPEC/rdzv_$SLURM_JOB_ID
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
mkdir -p "$RDZV_DIR" "$SAVE_PATH/logs"
LAYERS="[2, 30, 58, 60]"    # verifier saves fc-aux [2,30,58] + verifier_last 60 (off-by-one)
SPEC_CFG="{\"method\": \"extract_hidden_states\", \"num_speculative_tokens\": 1, \"draft_model_config\": {\"hf_config\": {\"eagle_aux_hidden_state_layer_ids\": $LAYERS}}}"
KV_CFG="{\"kv_connector\": \"ExampleHiddenStatesConnector\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"shared_storage_path\": \"$SHM\"}}"
OVERLAY=$SPEC/vllm_overlay/example_hidden_states_connector.py
CONNPATH=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
PY310=/t1data/users/kmswin7/py310_include

srun --container-image="$IMG" \
  --container-mounts=/t1data:/t1data,/dev/shm:/dev/shm,$OVERLAY:$CONNPATH,$PY310/python3.10:/usr/include/python3.10,$PY310/x86_64-python3.10:/usr/include/x86_64-linux-gnu/python3.10 \
  bash -lc "
    set -x
    NID=\$SLURM_NODEID; NN=\$SLURM_NNODES
    rm -rf $SHM; mkdir -p $SHM
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_READY_TIMEOUT_S=60000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
    export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    # --- per-node verifier (background) ---
    vllm serve $CKPT --served-model-name $CKPT --host 0.0.0.0 --port $PORT \
      --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
      --speculative_config '$SPEC_CFG' --kv_transfer_config '$KV_CFG' \
      --no-enable-chunked-prefill --no-enable-prefix-caching --max-model-len 8448 \
      --max-num-seqs ${MAXSEQ:-8} --gpu-memory-utilization ${MEMUTIL:-0.60} &
    SPID=\$!
    for i in \$(seq 1 1000); do curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && break; kill -0 \$SPID 2>/dev/null || { echo COLO5_VERIFIER_DIED node=\$NID; exit 1; }; sleep 5; done
    echo COLO5_VERIFIER_READY node=\$NID
    # --- cross-node barrier: all verifiers healthy before torchrun rendezvous ---
    touch $RDZV_DIR/ready_\$NID
    for i in \$(seq 1 1200); do [ \$(ls $RDZV_DIR/ready_* 2>/dev/null | wc -l) -ge \$NN ] && break; sleep 5; done
    echo COLO5_ALL_READY node=\$NID count=\$(ls $RDZV_DIR/ready_* 2>/dev/null | wc -l)
    # --- multi-node DDP trainer; each rank uses its OWN node's verifier + local /dev/shm ---
    cd $SPEC && source speculators_venv/bin/activate
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    NODE=\$(hostname)
    export no_proxy=\"\$NODE,127.0.0.1,localhost\" NO_PROXY=\"\$NODE,127.0.0.1,localhost\" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=enp71s0
    torchrun --nnodes=\$NN --node-rank=\$NID --nproc_per_node=8 \
      --master_addr=$MASTER --master_port=29511 \
      scripts/train.py \
      --speculator-type dspark --draft-arch qwen3 --num-layers 3 --draft-vocab-size 32768 \
      --verifier-name-or-path $CKPT --trust-remote-code \
      --data-path $SPEC/output_combined --total-seq-len 8192 \
      --target-layer-ids 2 30 58 --block-size 5 --markov-rank 256 --sliding-window 128 \
      --vllm-endpoint http://\$NODE:$PORT/v1 \
      --hidden-states-backend file --hidden-states-path $SHM \
      --on-missing generate --on-generate delete \
      --epochs ${EPOCHS:-5} --checkpoint-freq ${CKPTFREQ:-0.25} --save-path $SAVE_PATH \
      --logger '' --log-dir $SAVE_PATH/logs \
      --request-timeout 1800 --max-retries 8 \
      ${MAXSTEPS:+--max-steps \$MAXSTEPS}
    echo \"COLO5_TRAIN_DONE node=\$NID rc=\$? \$(date '+%F %T')\"
    kill \$SPID 2>/dev/null
  "
rm -rf "$RDZV_DIR"
