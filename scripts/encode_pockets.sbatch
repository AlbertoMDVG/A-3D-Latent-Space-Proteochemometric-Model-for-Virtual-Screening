#!/bin/bash
#SBATCH --job-name=poc_encode
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=poc_encode_%j.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
OUT=${OUT:-$DATA/results/pockets}
export BIGBIND=${BIGBIND:-$DATA/datasets/BigBindV1.5}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
python src/pockets_encoding/encode_bigbind_pockets.py \
    --out_dir "$OUT" \
    --layers vae raw \
    --device cuda
