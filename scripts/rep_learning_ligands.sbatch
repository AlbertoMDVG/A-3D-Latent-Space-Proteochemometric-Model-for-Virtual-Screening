#!/bin/bash
#SBATCH --job-name=lig_rep
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=lig_rep_%A_%a.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
OUT=${OUT:-$DATA/results/ligands}
SIZES=${SIZES:-"1000 5000 10000 50000"}

LAYERS=(vae raw)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
python src/ligands_encoding/rep_learning_ligands.py \
    --encodings "$OUT/ligands_$LAYER.parquet" \
    --features  "$OUT/ligand_features.parquet" \
    --out       "$OUT/rep_learning_$LAYER.csv" \
    --sizes $SIZES
