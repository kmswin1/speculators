#!/bin/bash
# verifier(3409) ready 대기 후 trainer를 같은 노드에 srun --overlap으로 붙여 실행.
# 127.0.0.1 endpoint + /dev/shm 캐시 + single-writer 패치된 verifier 사용.
export PATH=$PATH:/opt/slurm/bin
SPEC=/t1data/users/kmswin7/speculators
IMG=/t1data/users/kmswin7/vllm-v0.23.0-for-sktax.sqsh
CKPT=/t1data/users/kmswin7/A.X-K2_fused_yarn2
VERIF=$(cat /t1data/users/kmswin7/.claude/jobs/ff73fc1e/tmp/verif_jid.txt)
VL=$(ls -t /t1data/users/kmswin7/serving/logs/spec/*.hsvshm.log | head -1)
echo "[overlap] verifier job=$VERIF log=$VL — ready 대기"
for i in $(seq 1 200); do
  grep -aq "HSV_SHM_READY" "$VL" 2>/dev/null && { echo "[overlap] verifier READY"; break; }
  squeue -h -j "$VERIF" 2>/dev/null | grep -q . || { echo "[overlap] verifier 잡 사라짐"; exit 1; }
  sleep 15
done
TL=$SPEC/serving_trainer_overlap.log; : > $TL 2>/dev/null || TL=/t1data/users/kmswin7/serving/logs/spec/trainer_overlap.log
echo "[overlap] trainer 시작 → $TL"
srun --overlap --jobid=$VERIF --ntasks=1 --gpus=8 --cpus-per-task=48 \
  --container-image="$IMG" --container-mounts=/t1data:/t1data,/dev/shm:/dev/shm,/t1data/users/kmswin7/py310_include/python3.10:/usr/include/python3.10,/t1data/users/kmswin7/py310_include/x86_64-python3.10:/usr/include/x86_64-linux-gnu/python3.10 \
  bash -lc "
    cd $SPEC && source speculators_venv/bin/activate
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_COMPILE_THREADS=1
    torchrun --standalone --nproc_per_node 8 scripts/train.py \
      --speculator-type dspark --draft-arch qwen3 --num-layers 3 --draft-vocab-size 32768 \
      --verifier-name-or-path $CKPT --trust-remote-code \
      --data-path $SPEC/output_combined --total-seq-len 8192 \
      --target-layer-ids 58 59 60 --block-size 5 --markov-rank 256 --sliding-window 128 \
      --vllm-endpoint http://compute-gpu-st-distributed-ml-10:8202/v1 \
      --hidden-states-backend file --hidden-states-path /dev/shm/hs_col \
      --on-missing generate --on-generate delete \
      --epochs 10 --save-best --save-path $SPEC/dspark_ckpt \
      --logger tensorboard --log-dir $SPEC/dspark_ckpt/logs \
      --request-timeout 1800 --max-retries 8 --max-steps ${MAXSTEPS:-60}
    echo TRAINER_OVERLAP_DONE rc=\$?
  " > "$TL" 2>&1
echo "[overlap] trainer 종료"
