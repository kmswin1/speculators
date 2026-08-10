#!/bin/bash
# node10 offline HS 캐시 빌드 (자동재시작+resume). 목표 CACHE_N 샘플 캐시.
export PATH=$PATH:/opt/slurm/bin no_proxy='*' NO_PROXY='*'
SPEC=/t1data/users/kmswin7/speculators
OUT_HS=$SPEC/output_combined/hidden_states
CACHE_N=${CACHE_N:-800}
log(){ echo "[gen-sup $(date '+%F %T')] $*"; }
for attempt in $(seq 1 20); do
  have=$(ls "$OUT_HS"/hs_*.safetensors 2>/dev/null | wc -l)
  log "attempt $attempt: 현재 캐시 $have / 목표 $CACHE_N"
  [ "$have" -ge "$CACHE_N" ] && { log "목표 달성 ($have) — 종료"; echo GEN_CACHE_DONE; break; }
  JID=$(RANK=0 WORLD_SIZE=1 CONC=24 MAXS=$CACHE_N OUT_HS=$OUT_HS \
        sbatch --parsable --nodelist=compute-gpu-st-distributed-ml-10 \
        --job-name=hsgen-cache $SPEC/gen_hs_offline.sbatch 2>/dev/null)
  log "offline gen job=$JID 제출 (verifier+gen 콜로케이트, resume)"
  # 이 잡이 끝날 때까지 대기 (verifier hang으로 죽으면 반환 → 다음 attempt가 resume)
  while squeue -h -j "$JID" 2>/dev/null | grep -q .; do sleep 30; done
  log "job=$JID 종료 — resume 재시도"
done
