#!/usr/bin/env python
"""
Score saved bundles on any BigBind split, without retraining.

One row per (bundle, split, row):

    latent, config, variant   which arm produced the score
    split                     train / val / test
    pocket, lig_key           what was scored
    row_id                    position within that split
    active                    the label
    score                     the model's score
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from train_flaml import VARIANTS, assemble, build_features, log, scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activities", type=Path, required=True)
    ap.add_argument("--ligands", type=Path, required=True)
    ap.add_argument("--pockets", type=Path, required=True)
    ap.add_argument("--bundles", required=True, help="glob for the .joblib bundles")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    choices=("train", "val", "test"))
    ap.add_argument("--seed", type=int, default=0,
                    help="must match the training run, or the conformer draw differs")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.bundles))
    if not paths:
        raise SystemExit(f"no bundles matched {args.bundles}")
    log(f"{len(paths)} bundles, splits {args.splits}")

    rows, parts = build_features(args)      
    y = rows.active.to_numpy()
    idx = {k: np.where(rows.split.to_numpy() == k)[0] for k in ("train", "val", "test")}

    import joblib
    out = []
    for path in paths:
        b = joblib.load(path)
        variant = b["variant"]
        if any(parts[n] is None for n in VARIANTS[variant]):
            log(f"skipping {path.name}: missing feature block for {variant}")
            continue

        X = assemble(parts, variant)
        if X.shape[1] != len(b["columns"]):
            raise SystemExit(f"{path.name}: trained on {len(b['columns'])} features, rebuilt "
                             f"{X.shape[1]}. These are not the parquets it saw.")

        for split in args.splits:
            Z = X[idx[split]]                 
            b["scaler"].transform(Z, copy=False)
            s = scores(b["estimator"], Z, b["columns"])
            sub = rows.iloc[idx[split]]
            out.append(pd.DataFrame({
                "latent": b["latent"], "config": b.get("config", ""), "variant": variant,
                "split": split,
                "pocket": sub.pocket.to_numpy(), "lig_key": sub.lig_key.to_numpy(),
                "row_id": np.arange(len(idx[split]), dtype=np.int32),
                "active": sub.active.to_numpy(bool),
                "score": np.asarray(s, dtype=np.float32)}))
            del Z
        del X
        log(f"  {path.name}: {sum(len(idx[s]) for s in args.splits):,} rows")

    if not out:
        raise SystemExit("nothing scored")
    frame = pd.concat(out, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False, compression="zstd")
    log(f"wrote {args.out}: {len(frame):,} rows over "
        f"{frame.groupby(['latent', 'config', 'variant']).ngroups} arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
