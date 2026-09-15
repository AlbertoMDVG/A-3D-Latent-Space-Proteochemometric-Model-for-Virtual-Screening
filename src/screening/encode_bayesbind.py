#!/usr/bin/env python
"""
Encode the BayesBind benchmark: its ligand union and one pocket per target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ligands_encoding"))
sys.path.insert(0, str(HERE.parent / "pockets_encoding"))

from encode_lig_utils import (MOLFLAE_CKPT, MOLFLAE_CONFIG, MolFLAEEncoder,
                              featurise_ligands, _width)
from encode_bigbind_pockets import MIN_RESIDUE_ATOMS, encode_sets, read_pocket_pdb


SPLITS = ("val", "test")
ACT_CUTOFF = 5.0      # analysis/bayesbind_analysis.py:18
MAX_LIGANDS = 1000    # cfg.baseline_max_ligands, applied in baselines/collate.py:25


def log(msg):
    print(f"[bb] {msg}", flush=True)


def discover_targets(root: Path, splits=SPLITS) -> list:
    out = []
    for s in splits:
        if not (root / s).is_dir():
            continue
        for t in sorted(p for p in (root / s).iterdir() if (p / "actives.csv").exists()):
            out.append({"split": s, "target": t.name, "dir": t})
    return out


def read_target_ligands(tdir: Path, pchembl_cutoff=ACT_CUTOFF, max_ligands=MAX_LIGANDS) -> tuple:
    act = pd.read_csv(tdir / "actives.csv")
    rnd = pd.read_csv(tdir / "random.csv")
    if max_ligands:
        act, rnd = act[:max_ligands], rnd[:max_ligands]
    if pchembl_cutoff is not None and "pchembl_value" in act.columns:
        act = act[pd.to_numeric(act["pchembl_value"], errors="coerce") >= pchembl_cutoff]
    elif "active" in act.columns:
        act = act[act["active"].astype(bool)]
    keep = ["lig_file", "lig_smiles"]
    return act[keep].drop_duplicates("lig_file"), rnd[keep].drop_duplicates("lig_file")


def read_ml_targets(path: Path) -> set:
    return {ln.strip() for ln in Path(path).read_text().splitlines()
            if ln.startswith("    ") and ln.strip()}


def pocket_row(enc, pdb: Path, batch_size: int, min_res_atoms: int = None) -> dict:
    if min_res_atoms is None:
        min_res_atoms = MIN_RESIDUE_ATOMS
    p = read_pocket_pdb(pdb)
    if len(p["z"]) == 0:
        return {}
    a = encode_sets(enc, [(p["z"], p["xyz"])], 1)[0].reshape(-1).astype(np.float32)
    res = [(p["z"][m], p["xyz"][m]) for r in np.unique(p["res_id"])
           for m in [p["res_id"] == r] if m.sum() >= min_res_atoms]
    if not res:
        return {}
    per = encode_sets(enc, res, batch_size).reshape(len(res), -1)
    b = np.concatenate([per.mean(0), per.std(0)]).astype(np.float32)
    row = {f"a_{j:0{_width(len(a))}d}": v for j, v in enumerate(a)}
    row.update({f"b_{j:0{_width(len(b))}d}": v for j, v in enumerate(b)})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bayesbind_root", type=Path, required=True)
    ap.add_argument("--bigbind_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--latent", default="vae", choices=("vae", "raw"))
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--pchembl_cutoff", type=float, default=ACT_CUTOFF)
    ap.add_argument("--max_ligands", type=int, default=MAX_LIGANDS)
    ap.add_argument("--min_res_atoms", type=int, default=None,
                    help="default: the encoder's own MIN_RESIDUE_ATOMS")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N targets")
    args = ap.parse_args()

    targets = discover_targets(args.bayesbind_root, args.splits.split(","))
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        raise SystemExit(f"no targets under {args.bayesbind_root} -- expected "
                         f"<root>/{{val,test}}/<TARGET>/actives.csv")
    cache = args.out_dir
    cache.mkdir(parents=True, exist_ok=True)
    log(f"{len(targets)} targets: " +
        ", ".join(f"{s}={sum(1 for t in targets if t['split'] == s)}" for s in SPLITS))

    rows = []
    for t in targets:
        for role, frame in zip(("active", "random"),
                               read_target_ligands(t["dir"], args.pchembl_cutoff,
                                                   args.max_ligands)):
            f = frame.copy()
            f["split"], f["target"], f["role"] = t["split"], t["target"], role
            rows.append(f)
    member = pd.concat(rows, ignore_index=True)
    member.to_parquet(cache / "membership.parquet")
    uniq = member.drop_duplicates("lig_file").lig_file.tolist()
    log(f"{len(member):,} target-ligand pairs -> {len(uniq):,} distinct molecules")

    records, rejected = featurise_ligands(
        [(f, str(args.bigbind_root / f)) for f in uniq], workers=args.workers)
    log(f"{len(records):,} featurised, {len(rejected):,} rejected")
    if rejected:
        pd.DataFrame(rejected)[["lig_key", "lig_file", "error"]].to_csv(
            cache / "rejected.csv", index=False)
    if not records:
        raise SystemExit("no molecule survived featurisation -- is --bigbind_root correct?")

    enc = MolFLAEEncoder(layer=args.latent, ckpt=args.ckpt or MOLFLAE_CKPT,
                         config=args.config or MOLFLAE_CONFIG,
                         device=args.device, batch_size=args.batch_size)
    enc.load()
    zh, _ = enc._forward(records)
    zh = zh.reshape(len(records), -1)
    lig = pd.DataFrame(zh, columns=[f"z_h_{i:0{_width(zh.shape[1])}d}"
                                    for i in range(zh.shape[1])])
    lig.insert(0, "lig_file", [r["lig_file"] for r in records])
    lig.insert(1, "lig_key", [r["lig_key"] for r in records])
    lig.to_parquet(cache / f"ligands_{args.latent}.parquet")
    log(f"ligands -> ligands_{args.latent}.parquet ({zh.shape[1]} dims)")

    prs = []
    for i, t in enumerate(targets, 1):
        row = pocket_row(enc, t["dir"] / "pocket.pdb", args.batch_size, args.min_res_atoms)
        if not row:
            log(f"  {t['target']}: no supported heavy atoms, skipped")
            continue
        prs.append({"split": t["split"], "target": t["target"], **row})
        log(f"  pocket {i}/{len(targets)} {t['target']}")
    pd.DataFrame(prs).to_parquet(cache / f"pockets_{args.latent}.parquet")
    log(f"pockets -> pockets_{args.latent}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
