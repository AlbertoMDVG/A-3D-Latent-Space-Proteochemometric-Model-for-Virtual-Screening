#!/usr/bin/env python
"""
Score saved bundles against an encoded benchmark cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_flaml import VARIANTS, scores                            

ARMS = ("vae_lgbm_B", "vae_lgbm_ligand_only")


def log(msg):
    print(f"[score] {msg}", flush=True)


def build_matrix(bundle, lig_zh: np.ndarray, poc_row: pd.Series) -> np.ndarray:
    parts = [lig_zh]
    for name in VARIANTS[bundle["variant"]][1:]:
        prefix = "a_" if name == "A" else "b_"
        cols = sorted([c for c in poc_row.index if str(c).startswith(prefix)],
                      key=lambda c: int(str(c).rsplit("_", 1)[1]))
        block = poc_row[cols].to_numpy(dtype=np.float32)
        parts.append(np.repeat(block[None, :], len(lig_zh), axis=0))
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def load_cache(cache: Path, latent: str, benchmark: str):
    lig = pd.read_parquet(cache / f"ligands_{latent}.parquet")
    zh = sorted([c for c in lig.columns if c.startswith("z_h_")],
                key=lambda c: int(c.rsplit("_", 1)[1]))
    key = "lig_key" if benchmark == "bayesbind" else "smiles"
    if benchmark == "bayesbind" and key not in lig.columns:
        lig[key] = lig.lig_file.str.rsplit("/", n=1).str[-1]
    poc = pd.read_parquet(cache / f"pockets_{latent}.parquet")
    idx = ["split", "target"] if benchmark == "bayesbind" else ["target"]
    return lig.set_index(key)[zh].astype(np.float32), poc.set_index(idx), key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=("bayesbind", "litpcba"))
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, required=True, help="dir holding the .joblib files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()

    member = pd.read_parquet(args.cache_dir / "membership.parquet")
    group = ["split", "target"] if args.benchmark == "bayesbind" else ["target"]
    cached, dumps = {}, []

    for arm in args.arms:
        bp = args.bundles / f"{arm}.joblib"
        if not bp.is_file():
            raise SystemExit(f"no bundle {bp}")
        b = joblib.load(bp)
        lat = b["latent"]
        if lat not in cached:
            cached[lat] = load_cache(args.cache_dir, lat, args.benchmark)
        lig_tab, poc_tab, key = cached[lat]
        log(f"{arm} ({b['variant']}, {len(b['columns'])} features)")

        for gk, grp in member.groupby(group, sort=False):
            if gk not in poc_tab.index:
                continue
            poc_row = poc_tab.loc[gk]
            if isinstance(poc_row, pd.DataFrame):
                poc_row = poc_row.iloc[0]
            if key not in grp.columns:
                grp = grp.assign(**{key: grp.lig_file.str.rsplit("/", n=1).str[-1]})
            g = grp[grp[key].isin(lig_tab.index)]
            if g.empty:
                continue

            s = np.empty(len(g), dtype=np.float32)
            for a in range(0, len(g), args.chunk):
                sl = slice(a, a + args.chunk)
                X = build_matrix(b, lig_tab.loc[g[key].iloc[sl]].to_numpy(np.float32), poc_row)
                if X.shape[1] != len(b["columns"]):
                    raise SystemExit(f"{arm}: built {X.shape[1]} features, model wants "
                                     f"{len(b['columns'])} -- wrong cache latent")
                b["scaler"].transform(X, copy=False)
                s[sl] = scores(b["estimator"], X, b["columns"])

            rec = {"bundle": arm, "target": g.target.iloc[0], key: g[key].to_numpy(),
                   "score": s}
            if args.benchmark == "bayesbind":
                rec["split"] = g.split.iloc[0]
                rec["role"] = g.role.to_numpy()
            else:
                rec["active"] = g.active.to_numpy(bool)
            dumps.append(pd.DataFrame(rec))

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(dumps, ignore_index=True)
    frame.to_parquet(out, index=False, compression="zstd")
    log(f"wrote {out}: {len(frame):,} rows over {frame.bundle.nunique()} arms, "
        f"{frame.target.nunique()} targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
