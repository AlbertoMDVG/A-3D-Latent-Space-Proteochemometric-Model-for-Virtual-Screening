#!/bin/bash
#SBATCH --job-name=lig_encode
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=lig_encode_%j.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
OUT=${OUT:-$DATA/results/ligands}
export BIGBIND=${BIGBIND:-$DATA/datasets/BigBindV1.5}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
# --workers picks up SLURM_CPUS_PER_TASK on its own.
python src/ligands_encoding/encode_bigbind_ligands.py \
    --out_dir "$OUT" \
    --layers vae raw \
    --device cuda
