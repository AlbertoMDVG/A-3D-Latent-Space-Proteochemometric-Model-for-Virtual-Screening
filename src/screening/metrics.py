#!/usr/bin/env python
"""
Virtual-screening metrics: enrichment and the Bayes enrichment factor.
efb()/efb_max() are the Bayes enrichment factor (arXiv:2403.10478)
"""
from __future__ import annotations

import numpy as np

__all__ = ["enrichment_factor", "ef_ceiling", "ef_realised",
           "efb", "efb_max", "efb_max_ci"]


def _expected_actives_in_top(y: np.ndarray, s: np.ndarray, n_top: int) -> float:
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    found, filled = 0.0, 0
    for _, idx in _tied_groups(s):
        if filled >= n_top:
            break
        g_size = len(idx)
        g_act = float(y[idx].sum())
        room = min(g_size, n_top - filled)
        found += g_act * room / g_size
        filled += room
    return found


def _tied_groups(s: np.ndarray):
    if len(s) == 0:
        return
    bounds = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(s)]])
    for a, b in zip(starts, ends):
        yield s[a], np.arange(a, b)


def enrichment_factor(y, s, frac: float = 0.01) -> float:
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n, n_act = len(y), float(y.sum())
    if n == 0 or n_act == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(frac * n)))
    return (_expected_actives_in_top(y, s, n_top) / n_top) / (n_act / n)


def ef_ceiling(y, frac: float = 0.01) -> float:
    y = np.asarray(y, dtype=float)
    n, n_act = len(y), float(y.sum())
    if n == 0 or n_act == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(frac * n)))
    return (min(n_top, n_act) / n_top) / (n_act / n)


def ef_realised(y, s, frac: float = 0.01) -> float:
    ef = enrichment_factor(y, s, frac)
    ceil = ef_ceiling(y, frac)
    if not np.isfinite(ef) or not np.isfinite(ceil) or ceil <= 1.0:
        return float("nan")
    return float((ef - 1.0) / (ceil - 1.0))



def _chi_cutoff(rand_sorted: np.ndarray, chi: float) -> float:
    n = len(rand_sorted)
    k = int(round((1.0 - chi) * n))
    return float(rand_sorted[min(max(k, 0), n) - 1])


def efb(act_scores, rand_scores, chi: float = 0.01, rand_sorted=None) -> float:
    a = np.asarray(act_scores, dtype=float)
    r = np.asarray(rand_scores, dtype=float) if rand_sorted is None else rand_sorted
    if len(a) == 0 or len(r) == 0:
        return float("nan")
    rs = r if rand_sorted is not None else np.sort(r)
    cut = _chi_cutoff(rs, chi)
    p_rand = float((rs >= cut).sum()) / len(rs)
    if p_rand == 0:
        return float("nan")
    p_act = float((a >= cut).sum()) / len(a)
    return 0.0 if p_act == 0 else p_act / p_rand


def efb_max(act_scores, rand_scores, rand_sorted=None) -> tuple[float, float]:
    a = np.asarray(act_scores, dtype=float)
    r = np.asarray(rand_scores, dtype=float) if rand_sorted is None else rand_sorted
    if len(a) == 0 or len(r) == 0:
        return float("nan"), float("nan")
    rs = r if rand_sorted is not None else np.sort(r)

    best, best_chi, seen = None, float("nan"), set()
    for cut in np.unique(a.astype(np.float16)):
        chi = float((rs >= cut).sum()) / len(rs)
        if chi == 0:
            chi = 1.0 / len(rs)        
        if chi in seen:
            continue
        seen.add(chi)
        v = efb(a, rs, chi, rand_sorted=rs)
        if np.isfinite(v) and (best is None or v > best):
            best, best_chi = v, chi
    return (float("nan"), float("nan")) if best is None else (float(best), best_chi)


def efb_max_ci(act_scores, rand_scores, reps: int = 1000, seed: int = 0,
               rand_sorted=None) -> tuple[float, float]:
    a = np.asarray(act_scores, dtype=float)
    rs = np.sort(np.asarray(rand_scores, dtype=float)) if rand_sorted is None else rand_sorted
    if len(a) == 0 or len(rs) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    v = np.array([efb_max(a[rng.integers(0, len(a), len(a))], rs, rand_sorted=rs)[0]
                  for _ in range(reps)], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
