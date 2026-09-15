#!/bin/bash
#SBATCH --job-name=bm_score
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=bm_score_%A_%a.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
BUNDLES=${BUNDLES:-$DATA/results/screening/vae_lgbm/models}
BB_CACHE=${BB_CACHE:-$DATA/results/bayesbind}
LP_CACHE=${LP_CACHE:-$DATA/results/litpcba/vae}
OUT=${OUT:-$DATA/results/benchmark_scores}

BENCHMARKS=(bayesbind litpcba)
BENCHMARK=${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}
CACHES=("$BB_CACHE" "$LP_CACHE")
CACHE=${CACHES[$SLURM_ARRAY_TASK_ID]}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
python src/screening/score_benchmarks.py \
    --benchmark "$BENCHMARK" \
    --cache_dir "$CACHE" \
    --bundles   "$BUNDLES" \
    --out       "$OUT/${BENCHMARK}_scores.parquet"
