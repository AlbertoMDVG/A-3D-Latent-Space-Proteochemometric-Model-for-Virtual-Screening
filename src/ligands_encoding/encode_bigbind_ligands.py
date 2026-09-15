#!/usr/bin/env python
"""Encode every BigBind ligand with the frozen MolFLAE encoder, once per latent layer.

    python src/ligands_encoding/encode_bigbind_ligands.py --layers vae raw
    python src/ligands_encoding/encode_bigbind_ligands.py --layers vae --limit 200

Ligands are listed and featurised once, then handed to one encoder per layer. Everything
lands in --out_dir (default encode_ligands_output/):

    ligands_<layer>.parquet   lig_key, lig_file, z_h_*, z_x_* (45 sorted node distances)
    ligands_<layer>.txt       the lig_key of every ligand in that parquet
    summary_<layer>.json      counts, timings, settings
    encode_<layer>.log        the log for that layer
    ligand_features.parquet   RDKit descriptors per molecule, keyed by lig_key
    errors.csv                every ligand that could not be featurised, with a reason
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from encode_lig_utils import (
    LAYERS, MOLFLAE_CKPT, MOLFLAE_CONFIG, REPO_ROOT, MolFLAEEncoder, featurise_ligands,
    generate_features, log,
)

BIGBIND = Path(os.environ.get("BIGBIND", REPO_ROOT / "data" / "BigBindV1.5"))
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


def ligand_tasks(root: Path, limit: int = None) -> list:
    """Unique (lig_file, absolute sdf path) over the whole of activities_all.csv."""
    files = pd.read_csv(root / "activities_all.csv", usecols=["lig_file"])["lig_file"]
    files = files.drop_duplicates()
    if limit:
        files = files.head(limit)
    return [(f, str(root / f)) for f in files]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bigbind_root", type=Path, default=BIGBIND)
    ap.add_argument("--out_dir", type=Path, default=Path("encode_ligands_output"))
    ap.add_argument("--layers", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    ap.add_argument("--ckpt", type=Path, default=MOLFLAE_CKPT)
    ap.add_argument("--config", type=Path, default=MOLFLAE_CONFIG)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--batch_size", type=int, default=512, help="molecules per forward pass")
    ap.add_argument("--workers", type=int, default=WORKERS, help="featurisation processes")
    ap.add_argument("--limit", type=int, default=None, help="only the first N ligands")
    ap.add_argument("--features", default="parquet", choices=("parquet", "csv", "none"),
                    help="format for the RDKit descriptor table, or none to skip it")
    args = ap.parse_args()

    root = args.bigbind_root.resolve()
    if not (root / "activities_all.csv").is_file():
        raise SystemExit(f"no activities_all.csv in {root} -- pass --bigbind_root or set $BIGBIND")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"bigbind {root}")
    log(f"out_dir {out_dir}")

    # --- list and featurise once, shared by every layer ------------------------------------
    tasks = ligand_tasks(root, args.limit)
    t0 = time.time()
    records, rejected = featurise_ligands(tasks, workers=args.workers)
    featurise_s = time.time() - t0
    log(f"featurised {len(records):,} of {len(tasks):,} ligands in {featurise_s:.0f}s "
        f"({args.workers} workers)")

    if rejected:
        pd.DataFrame(rejected)[["lig_key", "lig_file", "error"]].to_csv(
            out_dir / "errors.csv", index=False)
        counts = pd.Series([r["error"].split(":")[0] for r in rejected]).value_counts()
        log(f"rejected {len(rejected):,} -> errors.csv: " +
            ", ".join(f"{n} {reason}" for reason, n in counts.items()))
    if not records:
        raise SystemExit("nothing to encode")

    if args.features != "none":
        features = out_dir / f"ligand_features.{args.features}"
        frame = generate_features(tasks, features, workers=args.workers)
        log(f"{len(frame):,} rows of RDKit descriptors -> {features.name}")

    # --- one encoder per layer -------------------------------------------------------------
    for layer in args.layers:
        log.open(out_dir / f"encode_{layer}.log")
        log(f"=== layer {layer} | {len(records):,} ligands ===")
        encoder = MolFLAEEncoder(layer=layer, ckpt=args.ckpt, config=args.config,
                                 device=args.device, batch_size=args.batch_size)
        encoder.load()
        t0 = time.time()
        parquet = encoder.encode(records, out_dir / f"ligands_{layer}.parquet")
        dt = time.time() - t0

        (out_dir / f"ligands_{layer}.txt").write_text(
            "\n".join(r["lig_key"] for r in records) + "\n", encoding="utf-8")
        names = pq.ParquetFile(parquet).schema.names
        n_zh = sum(c.startswith("z_h_") for c in names)
        n_zx = sum(c.startswith("z_x_") for c in names)
        (out_dir / f"summary_{layer}.json").write_text(json.dumps({
            "layer": layer,
            "n_encoded": len(records),
            "n_rejected": len(rejected),
            "n_z_h": n_zh,
            "n_z_x": n_zx,
            "seconds": round(dt, 1),
            "device": encoder.device,
            "batch_size": args.batch_size,
            "bigbind_root": str(root),
            "ckpt": str(args.ckpt),
        }, indent=2), encoding="utf-8")

        log(f"wrote {parquet.name}: {len(records):,} rows, {n_zh} z_h + {n_zx} z_x columns")
        log(f"done in {dt:.0f}s ({len(records) / max(dt, 1e-9):.0f} mol/s) on {encoder.device}")
        log.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
