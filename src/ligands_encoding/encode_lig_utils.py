"""
MolFLAE ligand encoding: SDF featurisation, RDKit descriptors, and the encoder classes.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from scipy.spatial.distance import pdist

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "vendor").is_dir())
MOLFLAE_DIR = REPO_ROOT / "vendor" / "MolFLAE" / "Latent_Experiments"
MOLFLAE_CONFIG = MOLFLAE_DIR / "config.yaml"
MOLFLAE_CKPT = REPO_ROOT / "models" / "molflae" / "model-epoch=24-val_loss=3.40.ckpt"

# MolFLAE's fixed heavy-atom vocabulary
ATOM_TYPE_INDEX = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
N_ATOM_TYPES = 9
N_GLOBAL_NODES = 10
WRITE_CHUNK = 20_000                    

# Where to stop the encoder
LAYERS = ("vae", "raw")

DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "QED": QED.qed,
}


class _Log:
    """print() to stdout, and to the current layer's .log file when one is open."""

    file = None

    def open(self, path):
        self.close()
        self.file = open(path, "w", encoding="utf-8")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def __call__(self, msg):
        line = f"[INFO] {msg}"
        print(line, flush=True)
        if self.file:
            print(line, file=self.file, flush=True)


log = _Log()


def _run(fn, tasks, workers):
    """Map fn over tasks, in a process pool when asked for more than one worker."""
    if workers <= 1:
        return [fn(t) for t in tasks]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap(fn, tasks, chunksize=64))


def read_mol(path):
    """First molecule in an SDF, heavy atoms only. Returns (mol, reason)."""
    if not os.path.exists(path):
        return None, "missing_sdf"
    mol = next((m for m in Chem.SDMolSupplier(path, removeHs=True, sanitize=True)
                if m is not None), None)
    if mol is None:
        return None, "unreadable_sdf"
    if mol.GetNumConformers() == 0:
        return None, "no_conformer"
    return mol, ""


def featurise_one(task: tuple) -> dict:
    """(lig_file, sdf path) -> encoder input"""
    lig_file, path = task
    rec = {"lig_file": lig_file, "lig_key": os.path.basename(lig_file)}

    mol, reason = read_mol(path)
    if mol is None:
        return {**rec, "error": reason}

    atomic = [a.GetAtomicNum() for a in mol.GetAtoms()]
    bad = sorted(set(atomic) - set(ATOM_TYPE_INDEX))
    if bad:
        return {**rec, "error": f"unsupported_element:{bad}"}

    rec["atom_idx"] = [ATOM_TYPE_INDEX[z] for z in atomic]
    rec["coords"] = mol.GetConformer().GetPositions().tolist()
    return rec


def featurise_ligands(tasks: list, workers: int = 1) -> tuple[list, list]:
    """Featurise every task. Returns (usable records, rejected records)."""
    out = _run(featurise_one, tasks, workers)
    return [r for r in out if "error" not in r], [r for r in out if "error" in r]


def descriptors_one(task: tuple) -> dict:
    """(lig_file, sdf path) -> one row of desc_* values, or an 'error'."""
    lig_file, path = task
    rec = {"lig_key": os.path.basename(lig_file), "lig_file": lig_file}

    mol, reason = read_mol(path)
    if mol is None:
        return {**rec, "error": reason}
    try:
        rec.update({f"desc_{name}": float(fn(mol)) for name, fn in DESCRIPTORS.items()})
    except Exception as exc:
        return {**rec, "error": f"descriptor_failed:{type(exc).__name__}"}
    return rec


def generate_features(tasks: list, out_path=None, workers: int = 1) -> pd.DataFrame:
    """
    RDKit descriptors per molecule, keyed by lig_key. Needs no model.
    """
    frame = pd.DataFrame([r for r in _run(descriptors_one, tasks, workers) if "error" not in r])
    if out_path is not None:
        out_path = Path(out_path)
        if out_path.suffix == ".csv":
            frame.to_csv(out_path, index=False)
        else:
            frame.to_parquet(out_path, index=False, compression="zstd")
    return frame


def postprocess_Zx(zx: np.ndarray) -> np.ndarray:
    """
    (n, 10, 3) -> (n, 45) sorted pairwise node distances, decreasing.
    Zx is rotation-invariant
    """
    return np.stack([np.sort(pdist(z))[::-1] for z in zx])


def _width(n_cols: int) -> int:
    return max(2, len(str(n_cols - 1)))


def latent_frame(records: list, zh: np.ndarray, zx: np.ndarray) -> pd.DataFrame:
    """One encoded chunk -> DataFrame of lig_key, lig_file, z_h_* (flat), z_x_* (45 sorted)."""
    n = len(records)
    zh, zx = zh.reshape(n, -1), postprocess_Zx(zx)
    wh, wx = _width(zh.shape[1]), _width(zx.shape[1])
    cols = {
        "lig_key": [r["lig_key"] for r in records],
        "lig_file": [r["lig_file"] for r in records],
    }
    cols.update({f"z_h_{j:0{wh}d}": zh[:, j] for j in range(zh.shape[1])})
    cols.update({f"z_x_{j:0{wx}d}": zx[:, j] for j in range(zx.shape[1])})
    return pd.DataFrame(cols)


def _knn_graph_pure(x, k, batch=None, loop=False, flow="source_to_target", **_):
    import torch

    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    rows, cols = [], []
    for b in torch.unique(batch):
        idx = torch.where(batch == b)[0]
        if len(idx) < 2:
            continue
        d = torch.cdist(x[idx], x[idx])
        d.fill_diagonal_(float("inf"))                     
        nbr = d.topk(min(k, len(idx) - 1), largest=False).indices
        query = torch.arange(len(idx), device=x.device).repeat_interleave(nbr.shape[1])
        src, dst = nbr.reshape(-1), query                 
        if flow != "source_to_target":
            src, dst = dst, src
        rows.append(idx[src])
        cols.append(idx[dst])
    if not rows:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def _import_molflae(molflae_dir=MOLFLAE_DIR):
    """Put the vendored tree on sys.path and import its config/model."""
    if str(molflae_dir) not in sys.path:
        sys.path.insert(0, str(molflae_dir))
    try:
        import torch_cluster
        backend = "torch_cluster"
    except ImportError:
        import torch_geometric.nn as gnn
        import torch_geometric.nn.pool as pool
        gnn.knn_graph = pool.knn_graph = _knn_graph_pure
        backend = "pure-torch fallback"
    from config.config import load_config
    from model.train_loop import TrainLoop, center_pos
    return load_config, TrainLoop, center_pos, backend


class Encoder(ABC):
    """Turns featurised ligands into a parquet of latent columns."""

    def __init__(self, layer: str, device: str = "auto", batch_size: int = 512):
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
        self.layer = layer
        self.device = device
        self.batch_size = batch_size

    @abstractmethod
    def encode(self, records: list, out_path) -> Path:
        """Encode `records` (from featurise_ligands) and write `out_path`."""


class MolFLAEEncoder(Encoder):
    """
    The frozen MolFLAE encoder, batched over molecules.
    """

    def __init__(self, layer: str, ckpt=MOLFLAE_CKPT, config=MOLFLAE_CONFIG,
                 device: str = "auto", batch_size: int = 512):
        super().__init__(layer, device, batch_size)
        self.ckpt = Path(ckpt)
        self.config = Path(config)
        self._model = None
        self._center_pos = None

    def load(self):
        """Load the checkpoint once and resolve the device."""
        if self._model is not None:
            return self._model
        import torch

        if not self.ckpt.is_file():
            raise SystemExit(f"no MolFLAE checkpoint at {self.ckpt} (67 MB, untracked)")
        load_config, TrainLoop, center_pos, backend = _import_molflae()

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            log("no GPU visible -- using cpu")
            self.device = "cpu"
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        with contextlib.redirect_stdout(io.StringIO()):   
            model = TrainLoop(load_config(str(self.config)))
        state = torch.load(str(self.ckpt), map_location=self.device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        model.to(self.device).eval()

        log(f"model on {self.device} | knn: {backend}")
        self._model, self._center_pos = model, center_pos
        return model

    def _forward(self, batch: list):
        """One batch of records -> (Zh, Zx) as (B, 10, d) and (B, 10, 3) arrays."""
        import torch
        import torch.nn.functional as F

        model = self.load()
        idx = [torch.tensor(r["atom_idx"], dtype=torch.long) for r in batch]
        batch_vec = torch.cat([torch.full((len(i),), n, dtype=torch.long)
                               for n, i in enumerate(idx)]).to(self.device)
        h = F.one_hot(torch.cat(idx), N_ATOM_TYPES).float().to(self.device)
        x = torch.cat([torch.tensor(r["coords"], dtype=torch.float32)
                       for r in batch]).to(self.device)
        x, _ = self._center_pos(x, batch_vec, mode=True)

        with torch.no_grad():
            if self.layer == "raw":
                zh, zx, _ = model.encoder(h, x, batch_vec)
            else:
                zh, zx, _, _, _ = model.encode(h, x, batch_vec, deterministic=True)

        b = len(batch)
        return (zh.reshape(b, N_GLOBAL_NODES, zh.shape[-1]).cpu().numpy(),
                zx.reshape(b, N_GLOBAL_NODES, 3).cpu().numpy())

    def encode(self, records: list, out_path) -> Path:
        """Encode every record and write one parquet, a chunk at a time."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_path = Path(out_path)
        self.load()
        writer = None
        for start in range(0, len(records), WRITE_CHUNK):
            chunk = records[start:start + WRITE_CHUNK]
            parts = [self._forward(chunk[i:i + self.batch_size])
                     for i in range(0, len(chunk), self.batch_size)]
            frame = latent_frame(chunk, np.concatenate([p[0] for p in parts]),
                                 np.concatenate([p[1] for p in parts]))
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
            writer.write_table(table)
            log(f"  encoded {start + len(chunk):,} / {len(records):,}")
        if writer:
            writer.close()
        return out_path
