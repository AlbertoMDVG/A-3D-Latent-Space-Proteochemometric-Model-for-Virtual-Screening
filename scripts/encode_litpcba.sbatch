#!/bin/bash
#SBATCH --job-name=lp_encode
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=lp_encode_%A_%a.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
OUT=${OUT:-$DATA/results/litpcba}
LITPCBA=${LITPCBA:-$DATA/datasets/lit-pcba/full}
LITPCBA_SI=${LITPCBA_SI:-$DATA/datasets/lit-pcba/ave}

LAYERS=(vae raw)
LATENT=${LAYERS[$SLURM_ARRAY_TASK_ID]}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
python src/screening/encode_litpcba.py \
    --litpcba_root "$LITPCBA" \
    --si_dir       "$LITPCBA_SI" \
    --pocket_dir   "$OUT/pockets" \
    --out_dir      "$OUT/$LATENT" \
    --latent       "$LATENT" \
    --workers      "${SLURM_CPUS_PER_TASK}" \
    --device       cuda
