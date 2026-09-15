#!/bin/bash
#SBATCH --job-name=bb_encode
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=bb_encode_%A_%a.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
OUT=${OUT:-$DATA/results/bayesbind}
BAYESBIND=${BAYESBIND:-$DATA/datasets/BayesBindV1.5}
BIGBIND=${BIGBIND:-$DATA/datasets/BigBindV1.5}

LAYERS=(vae raw)
LATENT=${LAYERS[$SLURM_ARRAY_TASK_ID]}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
python src/screening/encode_bayesbind.py \
    --bayesbind_root "$BAYESBIND" \
    --bigbind_root   "$BIGBIND" \
    --out_dir        "$OUT" \
    --latent         "$LATENT" \
    --workers        "${SLURM_CPUS_PER_TASK}" \
    --device         cuda
