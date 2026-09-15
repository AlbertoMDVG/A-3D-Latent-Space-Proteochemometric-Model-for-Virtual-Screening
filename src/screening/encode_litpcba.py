#!/usr/bin/env python
"""
Encode the LIT-PCBA biochemical targets
"""
from __future__ import annotations

import argparse
import contextlib
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ligands_encoding"))
sys.path.insert(0, str(HERE.parent / "pockets_encoding"))

from encode_lig_utils import (MolFLAEEncoder, MOLFLAE_CKPT, MOLFLAE_CONFIG,
                              ATOM_TYPE_INDEX, _run, _width)
from encode_bigbind_pockets import (MIN_RESIDUE_ATOMS, encode_sets,
                                    read_pocket_pdb)
from litpcba_pockets import SI_PDB, build_pocket_pdb

BIOCHEMICAL = ("ALDH1", "FEN1", "GBA", "IDH1", "KAT2A", "PKM2", "VDR")
EMBED_TRIES = 10          # num_embed_tries
UFF_ITERS = 500           # AllChem.UFFOptimizeMolecule(mol, 500)
EMBED_TIMEOUT_S = 20      # `with timeout(20)` around the whole embed loop


def log(msg):
    print(f"[lp] {msg}", flush=True)


class _EmbedTimeout(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise _EmbedTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def read_smi(path: Path, role: str, target: str) -> pd.DataFrame:
    """One LIT-PCBA .smi -> (smiles, cid, role, target). Two whitespace columns, no header."""
    rows = [(f[0], f[1] if len(f) > 1 else "")
            for f in (ln.split() for ln in Path(path).read_text().splitlines()) if f]
    d = pd.DataFrame(rows, columns=["smiles", "cid"])
    d["role"], d["target"] = role, target
    return d


def read_membership(root: Path, targets, max_inactives: int = 0) -> pd.DataFrame:
    frames = []
    for t in targets:
        d = root / t
        if not d.is_dir():
            raise SystemExit(f"no LIT-PCBA target directory {d}")
        act = read_smi(d / "actives.smi", "active", t)
        inact = read_smi(d / "inactives.smi", "inactive", t)
        if max_inactives:
            inact = inact[:max_inactives]
        inact = inact[~inact.smiles.isin(set(act.smiles))]
        frames.append(pd.concat([act, inact], ignore_index=True))
    m = pd.concat(frames, ignore_index=True)
    m["active"] = m.role == "active"
    return m


def featurise_smiles(task):
    RDLogger.DisableLog("rdApp.*")
    smiles, seed = task
    rec = {"lig_key": smiles, "lig_file": smiles, "ok": False, "error": ""}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rec["error"] = "unparseable_smiles"
        return rec
    bad = sorted({a.GetAtomicNum() for a in mol.GetAtoms()
                  if a.GetAtomicNum() not in ATOM_TYPE_INDEX})
    if bad:
        rec["error"] = f"unsupported_element:{bad}"
        return rec

    molh = Chem.AddHs(mol)
    try:
        with _time_limit(EMBED_TIMEOUT_S):
            for i in range(EMBED_TRIES):
                if AllChem.EmbedMolecule(molh, randomSeed=seed + i) == 0:
                    break
            else:
                rec["error"] = "embed_failed"
                return rec
            try:
                AllChem.UFFOptimizeMolecule(molh, UFF_ITERS)
            except RuntimeError:
                rec["error"] = "uff_failed"
                return rec
    except _EmbedTimeout:
        rec["error"] = "embed_timeout"
        return rec

    mol = Chem.RemoveHs(molh)
    rec["atom_idx"] = [ATOM_TYPE_INDEX[a.GetAtomicNum()] for a in mol.GetAtoms()]
    rec["coords"] = mol.GetConformer().GetPositions().tolist()
    rec["ok"] = True
    return rec


def pocket_pdbs(args) -> dict:
    out = {}
    pdir = args.pocket_dir or (args.out_dir / "pockets")
    pdir.mkdir(parents=True, exist_ok=True)
    for t in args.targets:
        found = sorted(pdir.glob(f"{t}_*_pocket.pdb"))
        if found:
            out[t] = found[0]
            continue
        d = args.si_dir / t
        prot = next(d.glob("*protein*.mol2"), None)
        lig = next(d.glob("*ligand*.mol2"), None)
        if prot is None or lig is None:
            log(f"  {t}: no mol2 pair under {d}, skipped")
            continue
        text, n = build_pocket_pdb(prot, lig, args.cutoff)
        p = pdir / f"{t}_{SI_PDB.get(t, 'pocket')}_pocket.pdb"
        p.write_text(text)
        out[t] = p
        log(f"  {t}: {n} residues -> {p.name}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--litpcba_root", type=Path, required=True, help="dir of <TARGET>/*.smi")
    ap.add_argument("--si_dir", type=Path, help="dir of <TARGET>/*.mol2 for pocket carving")
    ap.add_argument("--pocket_dir", type=Path, help="reuse pocket pdbs instead of carving")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", default=list(BIOCHEMICAL))
    ap.add_argument("--latent", default="vae", choices=("vae", "raw"))
    ap.add_argument("--cutoff", type=float, default=5.0, help="pocket radius, angstrom")
    ap.add_argument("--min_res_atoms", type=int, default=MIN_RESIDUE_ATOMS)
    ap.add_argument("--ckpt", type=Path, default=MOLFLAE_CKPT)
    ap.add_argument("--config", type=Path, default=MOLFLAE_CONFIG)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_inactives", type=int, default=0,
                    help="smoke runs only: cap inactives per target (0 = all)")
    args = ap.parse_args()

    cache = args.out_dir
    cache.mkdir(parents=True, exist_ok=True)
    member = read_membership(args.litpcba_root, args.targets, args.max_inactives)
    member.to_parquet(cache / "membership.parquet")
    uniq = member.drop_duplicates("smiles").smiles.tolist()
    log(f"{len(member):,} target-molecule pairs -> {len(uniq):,} distinct molecules")

    recs = _run(featurise_smiles, [(s, args.seed) for s in uniq], args.workers)
    ok = [r for r in recs if r["ok"]]
    bad = [r for r in recs if not r["ok"]]
    log(f"{len(ok):,} featurised, {len(bad):,} rejected")
    if bad:
        pd.DataFrame(bad)[["lig_key", "error"]].to_csv(cache / "rejected.csv", index=False)
    if not ok:
        raise SystemExit("no molecule survived featurisation")

    enc = MolFLAEEncoder(layer=args.latent, ckpt=args.ckpt, config=args.config,
                         device=args.device, batch_size=args.batch_size)
    enc.load()
    zh, _ = enc._forward(ok)
    zh = zh.reshape(len(ok), -1)
    lig = pd.DataFrame(zh, columns=[f"z_h_{i:0{_width(zh.shape[1])}d}"
                                    for i in range(zh.shape[1])])
    lig.insert(0, "smiles", [r["lig_key"] for r in ok])
    lig.to_parquet(cache / f"ligands_{args.latent}.parquet")
    log(f"ligands -> ligands_{args.latent}.parquet ({zh.shape[1]} dims)")

    prs = []
    for t, pdb in pocket_pdbs(args).items():
        p = read_pocket_pdb(pdb)
        a = encode_sets(enc, [(p["z"], p["xyz"])], 1)[0].reshape(-1).astype(np.float32)
        res = [(p["z"][m], p["xyz"][m]) for r in np.unique(p["res_id"])
               for m in [p["res_id"] == r] if m.sum() >= args.min_res_atoms]
        if not res:
            log(f"  {t}: no residue had >= {args.min_res_atoms} heavy atoms, skipped")
            continue
        per = encode_sets(enc, res, args.batch_size).reshape(len(res), -1)
        b = np.concatenate([per.mean(0), per.std(0)]).astype(np.float32)
        row = {"target": t, "pdb": SI_PDB.get(t, "")}
        row.update({f"a_{j:0{_width(len(a))}d}": v for j, v in enumerate(a)})
        row.update({f"b_{j:0{_width(len(b))}d}": v for j, v in enumerate(b)})
        prs.append(row)
        log(f"  pocket {t} ({pdb.name})")
    pd.DataFrame(prs).to_parquet(cache / f"pockets_{args.latent}.parquet")
    log(f"pockets -> pockets_{args.latent}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
