"""Jitter, phase and CCG metrics."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy import stats

def _moving_mean(x: NDArray, w: int = 100) -> NDArray:
    return np.convolve(x, np.ones(w) / w, mode="valid")

def jitter_curves(dts: list[NDArray], window: int = 100):
    movings = [_moving_mean(np.abs(dt), window) for dt in dts]
    return _stack_mean_sem(movings)

def early_late_stats(dt: NDArray) -> tuple[float, float]:
    n = len(dt)
    return dt[: n // 5].mean(), dt[-n // 5 :].mean()

def phase_curves(dts: list[NDArray], T: float = 0.200, window: int = 100):
    curves = []
    for dt in dts:
        dphi = (2 * np.pi * dt / T + np.pi) % (2 * np.pi) - np.pi
        expi = np.exp(1j * dphi)
        cs = np.cumsum(expi)
        mov = (cs[window - 1 :] - np.concatenate(([0], cs[:-window]))) / window
        curves.append(np.abs(mov))
    return _stack_mean_sem(curves)

def cross_correlogram(tA: NDArray, tB: NDArray, bins: NDArray):
    # vectorised outer difference
    lags = np.subtract.outer(tB, tA).ravel()
    hist, _ = np.histogram(lags, bins=bins)
    bw = bins[1] - bins[0]
    return hist / (hist.sum() * bw)

def paired_t(early: NDArray, late: NDArray) -> tuple[float, float]:
    return stats.ttest_rel(early, late)

# ---------- internal ----------
def _stack_mean_sem(curves: list[NDArray]):
    L = min(map(len, curves))
    stack = np.vstack([c[:L] for c in curves])
    mean = stack.mean(0)
    sem = stack.std(0, ddof=1) / np.sqrt(stack.shape[0])
    return mean, sem
