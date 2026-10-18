#!/usr/bin/env bash
# Slurm array job, one task per dataset: sbatch scripts/slurm_train.sh  (or --array=3 for HetData #3)
# Adapt the account/partition lines and the environment activation to your cluster.
#SBATCH --job-name=g2lmnet
#SBATCH --array=1-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/g2lmnet_%A_%a.out
##SBATCH --account=<your_account>
##SBATCH --partition=<gpu_partition>
set -euo pipefail
# conda activate g2lmnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # needed by deterministic CUDA kernels (HetData #3)
N=${SLURM_ARRAY_TASK_ID:-1}
DATA_DIR=${DATA_DIR:-data/hetdata}
WORK_DIR=${WORK_DIR:-work_dirs}
mkdir -p logs
python tools/train.py --config configs/hetdata${N}.yaml --data_dir "$DATA_DIR" --work_dir "$WORK_DIR/hetdata${N}"
