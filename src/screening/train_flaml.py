#!/usr/bin/env python
"""Train the virtual screening models on ligand + pocket latents.
Four arms per run, each a different feature block:

    ligand_only   ligand latent alone (the control)
    A             + whole-pocket latent
    B             + pooled per-residue latent
    A+B           + both

Three configs, run separately: lgbm (classification), rank (LambdaRank), lrl2 (linear).
Twelve arms per latent, 24 in total once vae and raw are both done.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

VARIANTS = {"ligand_only": ("L",), "A": ("L", "A"), "B": ("L", "B"), "A+B": ("L", "A", "B")}
CONFIGS = {
    "lgbm": {"task": "classification", "estimator": "lgbm", "metric": "roc_auc"},
    "rank": {"task": "rank", "estimator": "lgbm", "metric": "ndcg@20"},
    "lrl2": {"task": "classification", "estimator": "lrl2", "metric": "roc_auc"},
}
MAX_QUERY_ROWS = 10_000   


def log(msg):
    print(f"[train] {msg}", flush=True)


def peak_rss_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        return 0.0


def load_activities(csv_dir: Path) -> pd.DataFrame:
    cols = ["lig_file", "active", "pocket"]
    out = []
    for split in ("train", "val", "test"):
        d = pd.read_csv(csv_dir / f"activities_sna_1_{split}.csv", usecols=cols)
        d["split"] = split
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["lig_key"] = d.lig_file.str.rsplit("/", n=1).str[-1]
    d["active"] = d.active.astype(bool)
    return d


def numeric_cols(frame, prefix):
    cols = [c for c in frame.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def build_features(args):
    rows = load_activities(args.activities)
    n0 = len(rows)

    pock = pd.read_parquet(args.pockets)
    pock["pocket"] = pock.family
    lig = pd.read_parquet(args.ligands).drop_duplicates("lig_key")

    rows = rows[rows.pocket.isin(set(pock.pocket)) & rows.lig_key.isin(set(lig.lig_key))]
    rows = rows.reset_index(drop=True)
    log(f"{n0:,} activity rows -> {len(rows):,} with both a ligand latent and a pocket")
    if len(rows) == 0:
        raise SystemExit("no row has both a ligand latent and a pocket -- check the parquets")

    rng = np.random.default_rng(args.seed)
    pock = pock.reset_index(drop=True)
    by_target = {t: np.asarray(ix) for t, ix in pock.groupby("pocket").indices.items()}
    sizes = rows.pocket.map(lambda t: len(by_target[t])).to_numpy()
    pick = (rng.random(len(rows)) * sizes).astype(int)
    ri = np.array([by_target[t][k] for t, k in zip(rows.pocket, pick)], dtype=np.int64)

    zh = numeric_cols(lig, "z_h_")
    a_cols, b_cols = numeric_cols(pock, "a_"), numeric_cols(pock, "b_")
    lig_ix = {k: i for i, k in enumerate(lig.lig_key)}
    parts = {
        "L": lig[zh].to_numpy(np.float32)[rows.lig_key.map(lig_ix).to_numpy()],
        "A": pock[a_cols].to_numpy(np.float32)[ri] if a_cols else None,
        "B": pock[b_cols].to_numpy(np.float32)[ri] if b_cols else None,
    }
    log(f"widths: ligand {parts['L'].shape[1]}, A {len(a_cols)}, B {len(b_cols)}")
    log(f"conformers: {len(set(ri)):,} structures over {rows.pocket.nunique():,} targets")
    return rows, parts


def assemble(parts, variant):
    names = VARIANTS[variant]
    return parts[names[0]] if len(names) == 1 else np.hstack([parts[n] for n in names])


def split_groups(counts, limit=MAX_QUERY_ROWS):
    out = []
    for c in np.asarray(counts, dtype=np.int64):
        if c <= limit:
            out.append(int(c))
        else:
            k = int(np.ceil(c / limit))
            base, rem = divmod(int(c), k)
            out += [base + 1] * rem + [base] * (k - rem)
    return np.asarray(out, dtype=np.int64)


def fit(X, y, groups, cfg, seconds, seed, n_jobs):
    from flaml import AutoML

    names = [f"f{i}" for i in range(X.shape[1])]
    extra = {"estimator_list": [cfg["estimator"]]}

    if cfg["task"] == "rank":
        order = np.argsort(groups, kind="stable")
        X, y = X[order], np.asarray(y)[order]
        _, counts = np.unique(groups[order], return_counts=True)
        extra["groups"] = split_groups(counts)
    else:
        extra["split_type"] = "group"
        extra["groups"] = np.asarray(groups)

    if cfg["estimator"] == "lrl2":
        extra["custom_hp"] = {"lrl2": {"max_iter": {"domain": 1000}}}

    model = AutoML()
    model.fit(pd.DataFrame(X, columns=names, copy=False), np.asarray(y).astype(int),
              task=cfg["task"], time_budget=int(seconds), metric=cfg["metric"],
              seed=seed, n_jobs=n_jobs, verbose=1, early_stop=True, **extra)
    return model, names


def scores(model, X, names):
    for attr in ("predict_proba", "decision_function", "predict"):
        fn = getattr(model, attr, None)
        if fn is None:
            continue
        try:
            p = np.asarray(fn(pd.DataFrame(X, columns=names, copy=False)))
        except Exception:
            continue
        if attr == "predict_proba" and p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.ravel().astype(float)
    raise TypeError("no usable scoring method")


def per_target_auroc(y, s, targets):
    vals = []
    for t in np.unique(targets):
        m = targets == t
        if 0 < y[m].sum() < m.sum():
            vals.append(roc_auc_score(y[m], s[m]))
    vals = np.asarray(vals)
    return vals, (float(vals.mean()), float(np.median(vals))) if len(vals) else (np.nan, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activities", type=Path, required=True, help="dir with activities_sna_1_*.csv")
    ap.add_argument("--ligands", type=Path, required=True, help="ligand parquet, z_h_* columns")
    ap.add_argument("--pockets", type=Path, required=True, help="pocket parquet, a_*/b_* columns")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--config", default="lgbm", choices=list(CONFIGS))
    ap.add_argument("--latent", default="vae", choices=("vae", "raw"),
                    help="which latent the parquets hold; stored in the bundle so the "
                         "scoring scripts know which cache to pair it with")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seconds", type=int, default=1800, help="FLAML budget per arm")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    out_dir = args.out_dir.resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    log(f"config={args.config} task={cfg['task']} estimator={cfg['estimator']}")

    rows, parts = build_features(args)
    y = rows.active.to_numpy()
    targets = rows.pocket.to_numpy()
    idx = {k: np.where(rows.split.to_numpy() == k)[0] for k in ("train", "val", "test")}
    for k, v in idx.items():
        log(f"  {k:5} {len(v):>9,} rows  {rows.pocket[v].nunique():>5} targets  "
            f"{100 * y[v].mean():.1f}% active")

    results, all_scores = [], []
    for variant in args.variants:
        if any(parts[n] is None for n in VARIANTS[variant]):
            log(f"skipping {variant}: missing feature block")
            continue

        X = assemble(parts, variant)
        log(f"=== {variant} | {X.shape[1]} features ===")
        t0 = time.time()

        sc = StandardScaler().fit(X[idx["train"]])
        Xtr = X[idx["train"]]
        sc.transform(Xtr, copy=False)
        model, names = fit(Xtr, y[idx["train"]], targets[idx["train"]],
                           cfg, args.seconds, args.seed, args.n_jobs)
        del Xtr

        Xte = X[idx["test"]]
        sc.transform(Xte, copy=False)
        s_test = scores(model, Xte, names)
        del Xte

        _, (auroc_mean, auroc_median) = per_target_auroc(y[idx["test"]], s_test,
                                                         targets[idx["test"]])
        pooled = roc_auc_score(y[idx["test"]], s_test)
        dt = time.time() - t0

        results.append({"latent": args.latent, "config": args.config, "variant": variant,
                        "n_features": X.shape[1],
                        "auroc_per_target_mean": auroc_mean,
                        "auroc_per_target_median": auroc_median,
                        "auroc_pooled": pooled,
                        "best_estimator": model.best_estimator,
                        "best_loss": float(model.best_loss),
                        "n_trials": len(getattr(model, "config_history", []) or []),
                        "seconds": round(dt, 1),
                        "peak_rss_gb": peak_rss_gb()})
        log(f"  AUROC per-target {auroc_mean:.3f} | pooled {pooled:.3f} | "
            f"{dt:.0f}s | {results[-1]['n_trials']} trials | {results[-1]['peak_rss_gb']} GB")

        te = rows.iloc[idx["test"]]
        all_scores.append(pd.DataFrame({
            "latent": args.latent, "config": args.config, "variant": variant,
            "pocket": te.pocket.to_numpy(), "lig_key": te.lig_key.to_numpy(),
            "row_id": np.arange(len(idx["test"]), dtype=np.int32),
            "active": te.active.to_numpy(bool),
            "score": np.asarray(s_test, dtype=np.float32)}))

        import joblib
        joblib.dump({"estimator": model, "scaler": sc, "columns": names,
                     "latent": args.latent, "variant": variant, "config": args.config,
                     "n_features": X.shape[1]},
                    out_dir / "models" / f"{args.latent}_{args.config}_{variant}.joblib")
        del X
        pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
        pd.concat(all_scores, ignore_index=True).to_parquet(
            out_dir / "test_scores.parquet", index=False, compression="zstd")

    frame = pd.DataFrame(results)
    print("\n" + frame.to_string(index=False))
    (out_dir / "summary.json").write_text(json.dumps({
        "latent": args.latent,
        "config": args.config,
        "ligands": str(args.ligands),
        "pockets": str(args.pockets),
        "seconds_per_arm": args.seconds,
        "n_rows": len(rows),
        "n_test_rows": int(len(idx["test"])),
        "results": results,
    }, indent=2), encoding="utf-8")
    log(f"wrote results.csv, test_scores.parquet and summary.json in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
