#!/bin/bash
#SBATCH --job-name=bm_smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=bm_smoke_%j.out

set -euo pipefail

DATA=${DATA:-/data/stat-cadd/$USER}
REPO=${REPO:-$DATA/projects/A-3D-Latent-Space-Proteochemometric-Model-for-Virtual-Screening}
export OUT=${OUT:-$DATA/scratch/bm_smoke}
BAYESBIND=${BAYESBIND:-$DATA/datasets/BayesBindV1.5}
BIGBIND=${BIGBIND:-$DATA/datasets/BigBindV1.5}
LITPCBA=${LITPCBA:-$DATA/datasets/lit-pcba/full}
LITPCBA_SI=${LITPCBA_SI:-$DATA/datasets/lit-pcba/ave}
BUNDLES=${BUNDLES:-$DATA/results/screening/vae_lgbm/models}

N_TARGETS=${N_TARGETS:-2}
N_INACTIVES=${N_INACTIVES:-2000}
LP_TARGET=${LP_TARGET:-ALDH1}

module load Anaconda3/2025.06-1
conda activate ${CONDA_ENV:-$DATA/envs/MolFLAE}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
rm -rf "$OUT/bb" "$OUT/lp"
mkdir -p "$OUT"

echo "### 1/4 encode_bayesbind: $N_TARGETS targets, val split"
python src/screening/encode_bayesbind.py \
    --bayesbind_root "$BAYESBIND" --bigbind_root "$BIGBIND" \
    --out_dir "$OUT/bb" --latent vae --splits val --limit "$N_TARGETS" \
    --workers "${SLURM_CPUS_PER_TASK}" --device cuda

echo "### 2/4 score_benchmarks: bayesbind"
python src/screening/score_benchmarks.py \
    --benchmark bayesbind --cache_dir "$OUT/bb" --bundles "$BUNDLES" \
    --out "$OUT/bayesbind_scores.parquet"

echo "### 3/4 encode_litpcba: $LP_TARGET, $N_INACTIVES inactives"
python src/screening/encode_litpcba.py \
    --litpcba_root "$LITPCBA" --si_dir "$LITPCBA_SI" \
    --out_dir "$OUT/lp" --latent vae --targets "$LP_TARGET" \
    --max_inactives "$N_INACTIVES" \
    --workers "${SLURM_CPUS_PER_TASK}" --device cuda

echo "### 4/4 score_benchmarks: litpcba"
python src/screening/score_benchmarks.py \
    --benchmark litpcba --cache_dir "$OUT/lp" --bundles "$BUNDLES" \
    --out "$OUT/litpcba_scores.parquet"

echo "### summary"
python - <<'PYEOF'
import os, pandas as pd, pathlib
out = pathlib.Path(os.environ["OUT"])
for name in ("bayesbind", "litpcba"):
    f = out / f"{name}_scores.parquet"
    d = pd.read_parquet(f)
    print(f"{name}: {len(d):,} rows | arms {sorted(d.bundle.unique())} | "
          f"targets {d.target.nunique()} | score range "
          f"[{d.score.min():.4f}, {d.score.max():.4f}] | nan {d.score.isna().sum()}")
PYEOF
echo "### smoke OK"
