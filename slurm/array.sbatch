#!/bin/bash
# Run a sharded script as a SLURM job array (every shard skips the questions that are already done).
#
#   sbatch slurm/array.sbatch scripts/dump_attention.py --task coco --model qwen3-vl-2b \
#       --questions data/coco/questions.jsonl --out results/dumps
#
# The scripts read SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT as --shard / --nshards. Adjust the account,
# resources and array size below. The paper's runs used CPU nodes (float32); add --gres=gpu:1 and --device cuda for
# a GPU. Download models and data beforehand (scripts/download_models.py) if compute nodes have no internet.
#SBATCH --job-name=wherebits
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" HF_HUB_OFFLINE=1
python "$@"
