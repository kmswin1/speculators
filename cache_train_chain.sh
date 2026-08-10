#!/bin/bash
# 캐시가 THRESH개 쌓이면 node11 정적캐시 학습 자동 launch + loss 감시.
export PATH=$PATH:/opt/slurm/bin no_proxy='*' NO_PROXY='*'
SPEC=/t1data/users/kmswin7/speculators
HS=$SPEC/output_combined/hidden_states
THRESH=${THRESH:-150}
log(){ echo "[chain $(date '+%F %T')] $*"; }
log "캐시 $THRESH개 대기 시작"
for i in $(seq 1 720); do
  n=$(ls "$HS"/hs_*.safetensors 2>/dev/null | wc -l)
  [ $((i % 6)) -eq 0 ] && log "캐시 $n / $THRESH"
  [ "$n" -ge "$THRESH" ] && { log "캐시 $n개 도달 — 학습 launch"; break; }
  sleep 30
done
JID=$(EPOCHS=10 MAXSTEPS=60 SAVE_PATH=$SPEC/dspark_ckpt \
      sbatch --parsable --nodelist=compute-gpu-st-distributed-ml-11 \
      $SPEC/train_from_cache.sbatch 2>/dev/null)
log "train-from-cache job=$JID 제출 (node11, 정적캐시, MAXSTEPS=60)"
TL=$SPEC/../serving/logs/spec/$JID.traincache.log
for i in $(seq 1 480); do
  [ -f "$TL" ] || { sleep 15; continue; }
  if tr '\r' '\n' < "$TL" 2>/dev/null | grep -aqE "loss[=: ]"; then
    log "★★★ LOSS 등장 ★★★"
    tr '\r' '\n' < "$TL" | grep -aiE "loss|step|epoch" | tail -8
    echo LOSS_APPEARED; break
  fi
  if tr '\r' '\n' < "$TL" 2>/dev/null | grep -aqE "Traceback|Error|CUDA out|TRAIN_CACHE_DONE"; then
    log "학습 종료/에러 감지"; tr '\r' '\n' < "$TL" | tail -20; echo TRAIN_ENDED; break
  fi
  sleep 20
done
log "chain 종료"
