#!/usr/bin/env python
"""
Regress each ligand property on the MolFLAE latents, over a ladder of sample sizes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # must be imported before sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from encode_lig_utils import DESCRIPTORS, log

ALPHAS = np.logspace(-3, 3, 13)


def load_tables(encodings: Path, features: Path):
    """Inner-join latents and descriptors on lig_key. Returns (frame, z_h cols, z_x cols)."""
    enc = pd.read_parquet(encodings)
    fea = pd.read_csv(features) if features.suffix == ".csv" else pd.read_parquet(features)
    df = enc.merge(fea.drop(columns=["lig_file"]), on="lig_key")
    zh = sorted(c for c in df.columns if c.startswith("z_h_"))
    zx = sorted(c for c in df.columns if c.startswith("z_x_"))
    log(f"{len(df):,} ligands joined | {len(zh)} z_h + {len(zx)} z_x columns")
    return df, zh, zx


def regress_property(df, zh, zx, prop, size, cv=5, seed=0, save_dir=None, layer="") -> list:
    """One property at one sample size: 2 models x 3 feature sets = 6 rows."""
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
        "hgb": make_pipeline(StandardScaler(),
                             HistGradientBoostingRegressor(max_iter=150, random_state=seed)),
    }
    y_all = df[f"desc_{prop}"].to_numpy(float)
    idx = np.flatnonzero(np.isfinite(y_all))
    if size < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, size, replace=False))
    y = y_all[idx]

    rows = []
    for fs_name, cols in (("z_h", zh), ("z_x", zx), ("z_h+z_x", zh + zx)):
        X = df.iloc[idx][cols].to_numpy(np.float32)
        for mname, model in models.items():
            t0 = time.time()
            folds = cross_val_score(model, X, y, cv=KFold(cv, shuffle=False),
                                    scoring="r2", n_jobs=cv)
            rows.append({
                "run": f"{layer}|n{len(idx)}|{prop}|{fs_name}|{mname}",
                "layer": layer, "property": prop, "features": fs_name, "model": mname,
                "n": len(idx), "n_features": X.shape[1], "cv": cv,
                "r2_mean": folds.mean(), "r2_std": folds.std(),
                "r2_min": folds.min(), "r2_max": folds.max(),
                "seconds": round(time.time() - t0, 1),
                **{f"r2_fold_{i}": s for i, s in enumerate(folds)},
            })
            if save_dir:
                import joblib
                model.fit(X, y)
                joblib.dump(model, Path(save_dir) /
                            f"{layer}_{prop}_{fs_name}_{mname}_n{len(idx)}.joblib")
    return rows


def rep_by_size(df, zh, zx, sizes, properties, cv=5, seed=0, save_dir=None, layer=""):
    """Every property at every sample size."""
    rows = []
    for size in sizes:
        for prop in properties:
            t0 = time.time()
            got = regress_property(df, zh, zx, prop, size, cv, seed, save_dir, layer)
            rows += got
            best = max(got, key=lambda r: r["r2_mean"])
            log(f"n={size:,} {prop:18} {time.time() - t0:5.0f}s | best R2 "
                f"{best['r2_mean']:.3f} ({best['model']} on {best['features']})")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encodings", type=Path, required=True, help="ligands_<layer>.parquet")
    ap.add_argument("--features", type=Path, required=True, help="ligand_features.parquet/csv")
    ap.add_argument("--sizes", type=int, nargs="+", required=True,
                    help="sample sizes, e.g. --sizes 1000 5000 10000 50000")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--properties", nargs="+", default=list(DESCRIPTORS))
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_models", action="store_true",
                    help="also write a fitted model per cell (off: only the scores are kept)")
    args = ap.parse_args()

    layer = args.encodings.stem.replace("ligands_", "")
    out = args.out or args.encodings.parent / f"rep_learning_{layer}.csv"
    save_dir = out.parent / f"models_{layer}" if args.save_models else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    log(f"layer {layer} | sizes {args.sizes} | {len(args.properties)} properties | cv {args.cv}")
    df, zh, zx = load_tables(args.encodings, args.features)

    t0 = time.time()
    frame = rep_by_size(df, zh, zx, args.sizes, args.properties, args.cv, args.seed,
                        save_dir, layer)
    frame.to_csv(out, index=False)
    log(f"wrote {out} ({len(frame)} rows) in {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
