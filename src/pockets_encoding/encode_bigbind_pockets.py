#!/usr/bin/env python
"""
Encode every BigBind pocket with the frozen MolFLAE encoder.

Three pocket representations, all from the same frozen encoder:
    A     the whole pocket encoded as one molecule           
    B     each residue encoded on its own, then pooled      
    A+B   both blocks together                               

Outputs:

    pockets_<layer>.parquet   pocket_key, family, a_*, b_*, aa_*, and a few counts
    pockets_<layer>.txt       the pocket_key of every pocket in that parquet
    summary_<layer>.json      counts, timings, settings
    encode_<layer>.log        the log for that layer
    errors.csv                every pocket that could not be encoded, with a reason
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ligands_encoding"))
from encode_lig_utils import (
    ATOM_TYPE_INDEX, LAYERS, MOLFLAE_CKPT, MOLFLAE_CONFIG, REPO_ROOT,
    MolFLAEEncoder, log,
)

BIGBIND = Path(os.environ.get("BIGBIND", REPO_ROOT / "data" / "BigBindV1.5"))
AA_ORDER = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

ELEMENT_TO_Z = {
    "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "CL": 17, "BR": 35, "I": 53,
    "SE": 34, "H": 1, "D": 1, "NA": 11, "MG": 12, "K": 19, "CA": 20, "MN": 25,
    "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30, "CD": 48, "HG": 80,
}

MIN_RESIDUE_ATOMS = 3 


def read_pocket_pdb(path) -> dict:
    """
    Parse a pocket .pdb into the heavy atoms MolFLAE can encode.
    """
    from Bio.PDB import PDBParser

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         
        structure = PDBParser(QUIET=True).get_structure("pocket", str(path))

    z, xyz, res_id, res_name = [], [], [], []
    dropped = {"hydrogen": 0, "hetatm": 0, "unsupported": 0, "unknown": 0}
    next_res = 0

    for residue in structure.get_residues():
        is_het = residue.get_id()[0].strip() != "" 
        used = False
        for atom in residue.get_atoms():
            if is_het:
                dropped["hetatm"] += 1
                continue
            zi = ELEMENT_TO_Z.get((atom.element or "").strip().upper())
            if zi is None:
                dropped["unknown"] += 1
            elif zi == 1:
                dropped["hydrogen"] += 1
            elif zi not in ATOM_TYPE_INDEX:
                dropped["unsupported"] += 1
            else:
                z.append(zi)
                xyz.append(atom.coord)
                res_id.append(next_res)
                res_name.append(residue.get_resname().strip().upper())
                used = True
        if used:
            next_res += 1

    return {
        "z": np.asarray(z, dtype=int),
        "xyz": np.asarray(xyz, dtype=float).reshape(-1, 3),
        "res_id": np.asarray(res_id, dtype=int),
        "res_name": np.asarray(res_name, dtype=object),
        "dropped": dropped,
    }


def residue_counts(pocket: dict) -> dict:
    """
    How many of each of the 20 standard amino acids the pocket contains.
    """
    res_id, res_name = pocket["res_id"], pocket["res_name"]
    if len(res_id) == 0:
        return {a: 0 for a in AA_ORDER}
    _, first = np.unique(res_id, return_index=True)
    names = [str(n).upper() for n in res_name[np.sort(first)]]
    return {a: int(names.count(a)) for a in AA_ORDER}


def pocket_tasks(root: Path, limit: int = None) -> list:
    tasks = []
    for family_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for pdb in sorted(family_dir.glob("*_rec_pocket.pdb")):
            tasks.append((f"{family_dir.name}/{pdb.name}", family_dir.name, str(pdb)))
    return tasks[:limit] if limit else tasks


def encode_sets(encoder: MolFLAEEncoder, atom_sets: list, batch_size: int) -> np.ndarray:
    """
    Encode [(z, xyz), ...] and return one (10, d) latent per set, in input order.
    """
    out = []
    for start in range(0, len(atom_sets), max(1, batch_size)):
        records = [{"atom_idx": [ATOM_TYPE_INDEX[int(v)] for v in z],
                    "coords": np.asarray(xyz, dtype=float).reshape(-1, 3).tolist()}
                   for z, xyz in atom_sets[start:start + max(1, batch_size)]]
        zh, _ = encoder._forward(records)
        out.append(zh)
    return np.concatenate(out)


def encode_pockets(encoder: MolFLAEEncoder, tasks: list, batch_size: int = 32):
    """Encode every pocket into variants A and B. Returns (rows, rejected).

    One pocket needs two kinds of forward pass:
      A  a single graph holding every heavy atom in the pocket
      B  one graph per residue, then mean and std over the per-residue latents
    """
    rows, rejected = [], []
    for n, (key, family, path) in enumerate(tasks, 1):
        try:
            pocket = read_pocket_pdb(path)
            if len(pocket["z"]) == 0:
                raise ValueError("no encodable heavy atoms")

            # variant B: every residue with enough heavy atoms, encoded on its own
            residues = [(pocket["z"][m], pocket["xyz"][m])
                        for m in (pocket["res_id"] == r for r in np.unique(pocket["res_id"]))
                        if m.sum() >= MIN_RESIDUE_ATOMS]
            if not residues:
                raise ValueError(f"no residue had >= {MIN_RESIDUE_ATOMS} heavy atoms")
            per_residue = encode_sets(encoder, residues, batch_size)
            flat = np.stack([z.reshape(-1) for z in per_residue])        # (n_res, 10*d)
            b = np.concatenate([flat.mean(0), flat.std(0)]).astype(np.float32)

            # variant A: the whole pocket as one graph
            a = encode_sets(encoder, [(pocket["z"], pocket["xyz"])], 1)[0]
            a = a.reshape(-1).astype(np.float32)

            row = {"pocket_key": key, "family": family,
                   "n_atoms": len(pocket["z"]),
                   "n_dropped": int(sum(pocket["dropped"].values())),
                   "n_res_encoded": len(residues),
                   "n_res_total": int(len(np.unique(pocket["res_id"])))}
            row.update({f"a_{i:0{_width(len(a))}d}": v for i, v in enumerate(a)})
            row.update({f"b_{i:0{_width(len(b))}d}": v for i, v in enumerate(b)})
            row.update({f"aa_{k}": v for k, v in residue_counts(pocket).items()})
            rows.append(row)
        except Exception as exc:
            rejected.append({"pocket_key": key, "family": family,
                             "error": f"{type(exc).__name__}: {exc}"})
        if n % 500 == 0:
            log(f"  encoded {n:,} / {len(tasks):,}")
    return rows, rejected


def _width(n_cols: int) -> int:
    return max(2, len(str(n_cols - 1)))


def variant_columns(frame: pd.DataFrame, variant: str) -> list:
    """The feature columns for 'A', 'B' or 'A+B'"""
    a = sorted(c for c in frame.columns if c.startswith("a_"))
    b = sorted(c for c in frame.columns if c.startswith("b_"))
    return {"A": a, "B": b, "A+B": a + b}[variant.upper()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bigbind_root", type=Path, default=BIGBIND)
    ap.add_argument("--out_dir", type=Path, default=Path("encode_pockets_output"))
    ap.add_argument("--layers", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    ap.add_argument("--ckpt", type=Path, default=MOLFLAE_CKPT)
    ap.add_argument("--config", type=Path, default=MOLFLAE_CONFIG)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--batch_size", type=int, default=32, help="residues per forward pass")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pockets")
    args = ap.parse_args()

    root = args.bigbind_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"no such directory {root} -- pass --bigbind_root or set $BIGBIND")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = pocket_tasks(root, args.limit)
    if not tasks:
        raise SystemExit(f"no *_rec_pocket.pdb under {root}")
    log(f"bigbind {root}")
    log(f"out_dir {out_dir}")
    log(f"{len(tasks):,} pockets in {len({t[1] for t in tasks}):,} families")

    for layer in args.layers:
        log.open(out_dir / f"encode_{layer}.log")
        log(f"=== layer {layer} | {len(tasks):,} pockets ===")
        encoder = MolFLAEEncoder(layer=layer, ckpt=args.ckpt, config=args.config,
                                 device=args.device, batch_size=args.batch_size)
        encoder.load()
        t0 = time.time()
        rows, rejected = encode_pockets(encoder, tasks, args.batch_size)
        dt = time.time() - t0
        if not rows:
            raise SystemExit("nothing encoded")

        frame = pd.DataFrame(rows)
        parquet = out_dir / f"pockets_{layer}.parquet"
        frame.to_parquet(parquet, index=False, compression="zstd")
        (out_dir / f"pockets_{layer}.txt").write_text(
            "\n".join(frame.pocket_key) + "\n", encoding="utf-8")
        if rejected:
            pd.DataFrame(rejected).to_csv(out_dir / "errors.csv", index=False)
            log(f"rejected {len(rejected):,} -> errors.csv")

        n_a, n_b = len(variant_columns(frame, "A")), len(variant_columns(frame, "B"))
        (out_dir / f"summary_{layer}.json").write_text(json.dumps({
            "layer": layer,
            "n_encoded": len(frame),
            "n_rejected": len(rejected),
            "n_a": n_a, "n_b": n_b, "n_a_plus_b": n_a + n_b,
            "seconds": round(dt, 1),
            "device": encoder.device,
            "batch_size": args.batch_size,
            "bigbind_root": str(root),
            "ckpt": str(args.ckpt),
        }, indent=2), encoding="utf-8")

        log(f"wrote {parquet.name}: {len(frame):,} rows, {n_a} a_ + {n_b} b_ + 20 aa_ columns")
        log(f"done in {dt:.0f}s ({len(frame) / max(dt, 1e-9):.1f} pockets/s) on {encoder.device}")
        log.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
