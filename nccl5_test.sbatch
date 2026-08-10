#!/bin/bash
#SBATCH --job-name=nccl5
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=/t1data/users/kmswin7/serving/logs/spec/%j.nccl5.log
#SBATCH --error=/t1data/users/kmswin7/serving/logs/spec/%j.nccl5.log
set -x
SPEC=/t1data/users/kmswin7/speculators
IMG=/t1data/users/kmswin7/vllm-v0.23.0-for-sktax.sqsh
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
srun --container-image="$IMG" --container-mounts=/t1data:/t1data bash -lc "
  export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=enp71s0 NCCL_DEBUG=WARN NCCL_P2P_DISABLE=0
  cd $SPEC && source speculators_venv/bin/activate
  echo \"NODE=\$(hostname) NODEID=\$SLURM_NODEID NN=\$SLURM_NNODES MASTER=$MASTER\"
  torchrun --nnodes=\$SLURM_NNODES --node_rank=\$SLURM_NODEID --nproc_per_node=8 \
    --master_addr=$MASTER --master_port=29511 \
    /t1data/users/kmswin7/.claude/jobs/ff73fc1e/tmp/nccl_test.py
  echo \"NCCL5_DONE node=\$SLURM_NODEID rc=\$?\"
"
