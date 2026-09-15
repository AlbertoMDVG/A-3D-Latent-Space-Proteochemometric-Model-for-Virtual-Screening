#!/bin/bash
#SBATCH --job-name=vs_train
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=vs_train_%A_%a.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
LIG_DIR=${LIG_DIR:-$DATA/results/ligands}
POC_DIR=${POC_DIR:-$DATA/results/pockets}
OUT=${OUT:-$DATA/results/screening}
ACTIVITIES=${ACTIVITIES:-$DATA/datasets/BigBindV1.5}
SECONDS_PER_ARM=${SECONDS_PER_ARM:-1800}

COMBOS=(vae:lgbm vae:rank vae:lrl2 raw:lgbm raw:rank raw:lrl2)
COMBO=${COMBOS[$SLURM_ARRAY_TASK_ID]}
LATENT=${COMBO%%:*}
CONFIG=${COMBO##*:}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
python -c "import flaml" || { echo "flaml not installed in $CONDA_ENV"; exit 1; }

echo "[vs] latent=$LATENT config=$CONFIG"
python src/screening/train_flaml.py \
    --activities "$ACTIVITIES" \
    --ligands    "$LIG_DIR/ligands_$LATENT.parquet" \
    --pockets    "$POC_DIR/pockets_$LATENT.parquet" \
    --out_dir    "$OUT/${LATENT}_${CONFIG}" \
    --latent     "$LATENT" \
    --config     "$CONFIG" \
    --seconds    "$SECONDS_PER_ARM" \
    --n_jobs     "${SLURM_CPUS_PER_TASK}"
