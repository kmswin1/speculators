#!/bin/bash
# DSpark 학습 로그 깔끔 뷰어. 사용: bash watchloss.sh [jobid] [tail_n]
# ANSI 색코드 + progress bar(\r) 제거하고 loss/accept 지표만 보여줌.
JID=${1:-3819}
N=${2:-40}
LOG=/t1data/users/kmswin7/serving/logs/spec/${JID}.colo5.log
clean(){ sed 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\x1b\][0-9]*;[^\a]*\a//g; s/\r/\n/g'; }
if [ "$3" = "-f" ] || [ "$N" = "-f" ]; then
  # 실시간 follow (loss/accept 라인만)
  tail -F "$LOG" | clean | grep --line-buffered -aoE "train/(loss|accept_len|accept_rate|position_0_acc)=[0-9.]+" | paste - - - -
else
  echo "=== 최근 loss/accept 추이 (job $JID) ==="
  # 로그가 크므로(수백 MB) 끝부분 30MB만 훑어서 즉시 출력
  tail -c 30M "$LOG" | clean | grep -aoE "train/loss=[0-9.]+|train/accept_len=[0-9.]+|train/accept_rate=[0-9.]+|train/position_0_acc=[0-9.]+" | paste - - - - | tail -"$N"
  echo "=== 현재 스텝 ==="
  tail -c 30M "$LOG" | clean | grep -aoE "Epoch [0-9]+ +[0-9]+% +.*[0-9,]+/(3,007|336)" | tail -1
fi
