#!/bin/bash
#SBATCH --job-name=vs_score
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=vs_score_%A_%a.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
LIG_DIR=${LIG_DIR:-$DATA/results/ligands}
POC_DIR=${POC_DIR:-$DATA/results/pockets}
OUT=${OUT:-$DATA/results/screening}
ACTIVITIES=${ACTIVITIES:-$DATA/datasets/BigBindV1.5}
SEED=${SEED:-0}

LATENTS=(vae raw)
LATENT=${LATENTS[$SLURM_ARRAY_TASK_ID]}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
python src/screening/score_models.py \
    --activities "$ACTIVITIES" \
    --ligands    "$LIG_DIR/ligands_$LATENT.parquet" \
    --pockets    "$POC_DIR/pockets_$LATENT.parquet" \
    --bundles    "$OUT/${LATENT}_*/models/*.joblib" \
    --splits     val test \
    --seed       "$SEED" \
    --out        "$OUT/scores_$LATENT.parquet"
