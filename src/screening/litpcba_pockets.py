#!/usr/bin/env python
"""
Turn a LIT-PCBA into the pocket the encoder expects.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["SI_PDB", "STANDARD_AA", "read_mol2_atoms", "build_pocket_pdb"]
SI_PDB = {
    "GBA": "2xwd", "ALDH1": "4x4l", "ADRB2": "4lde", "ESR1_ant": "2r6w", "ESR1_ago": "2qse",
    "FEN1": "5fv7", "KAT2A": "5mlj", "PKM2": "5x1v", "MTORC1": "4dri", "IDH1": "5de1",
    "VDR": "3a2j", "PPARG": "3b1m", "MAPK1": "4qp9", "OPRK1": "6b73", "TP53": "4agq",
}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN", "MSE",
}

_SUBST = re.compile(r"^([A-Za-z]+)(\d+)([A-Za-z]?)$")


def read_mol2_atoms(path) -> dict:
    names, elems, xyz, sid, rname = [], [], [], [], []
    in_block = False
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("@<TRIPOS>"):
            in_block = line.strip() == "@<TRIPOS>ATOM"
            continue
        if not in_block:
            continue
        f = line.split()
        if len(f) < 6:
            continue
        try:
            x, y, z = float(f[2]), float(f[3]), float(f[4])
        except ValueError:
            continue
        names.append(f[1][:4])
        elems.append(f[5].split(".")[0].upper())
        xyz.append((x, y, z))
        sid.append(int(f[6]) if len(f) > 6 and f[6].lstrip("-").isdigit() else 1)
        m = _SUBST.match(f[7]) if len(f) > 7 else None
        rname.append(m.group(1).upper()[:3] if m else "LIG")

    return {"name": np.array(names, dtype=object),
            "element": np.array(elems, dtype=object),
            "xyz": np.asarray(xyz, dtype=float).reshape(-1, 3),
            "subst_id": np.asarray(sid, dtype=int),
            "res_name": np.array(rname, dtype=object)}


def _pdb_line(serial: int, name: str, res_name: str, res_seq: int, xyz, element: str) -> str:
    rec = "ATOM  " if res_name in STANDARD_AA else "HETATM"
    nm = name if len(element) == 2 else f" {name}"
    return (f"{rec}{serial:5d} {nm:<4.4s}{res_name:>4.4s} A{res_seq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"  1.00  0.00          {element:>2.2s}\n")


def build_pocket_pdb(protein_mol2, ligand_mol2, cutoff: float = 5.0) -> tuple:
    prot = read_mol2_atoms(protein_mol2)
    lig = read_mol2_atoms(ligand_mol2)
    if len(prot["xyz"]) == 0:
        raise SystemExit(f"no ATOM records parsed from {protein_mol2}")
    if len(lig["xyz"]) == 0:
        raise SystemExit(f"no ATOM records parsed from {ligand_mol2}")

    lig_heavy = lig["xyz"][lig["element"] != "H"]
    keep_atom = prot["element"] != "H"
    if lig_heavy.size == 0 or not keep_atom.any():
        raise SystemExit(f"{protein_mol2}: nothing left after dropping hydrogens")
    xyz = prot["xyz"]
    near = np.zeros(len(xyz), dtype=bool)
    for a in range(0, len(xyz), 20_000):
        blk = xyz[a:a + 20_000]
        d2 = ((blk[:, None, :] - lig_heavy[None, :, :]) ** 2).sum(-1)
        near[a:a + 20_000] = d2.min(1) <= cutoff * cutoff
    near &= keep_atom

    keep_res = set(prot["subst_id"][near].tolist())
    if not keep_res:
        raise SystemExit(f"{protein_mol2}: no residue within {cutoff} A of the ligand")
    sel = np.isin(prot["subst_id"], list(keep_res)) & keep_atom

    lines, serial = [], 1
    for new_seq, rid in enumerate(sorted(keep_res), start=1):
        m = sel & (prot["subst_id"] == rid)
        for i in np.flatnonzero(m):
            lines.append(_pdb_line(serial, prot["name"][i], prot["res_name"][i], new_seq,
                                   prot["xyz"][i], prot["element"][i]))
            serial += 1
    lines.append("END\n")

    info = {"n_res": len(keep_res), "n_atoms": int(sel.sum()),
            "n_lig_atoms": int(len(lig_heavy)), "cutoff": cutoff,
            "n_protein_atoms": int(len(xyz))}
    return "".join(lines), info
