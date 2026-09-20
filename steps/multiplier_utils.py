"""Shared diversification-multiplier math (FDM / IDM / PDM).

Every multiplier in the app has the form M = min(1 / sqrt(w' C w), cap). C here is NOT the raw
correlation matrix: negative off-diagonal correlations are floored at zero first, so a negatively
correlated pair can never inflate the multiplier beyond what uncorrelated inputs would give
(two equally weighted uncorrelated rules -> sqrt(2)). The raw matrix is left untouched so it can
still be displayed / audited; only the floored copy feeds the multiplier.

This module is deliberately free of Streamlit so it can be unit-tested on its own.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

VALIDITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class MultiplierResult:
    multiplier: float
    uncapped_multiplier: float
    wcw: float
    raw_corr: pd.DataFrame
    multiplier_corr: pd.DataFrame
    psd_projected: bool


def _as_frame(matrix) -> pd.DataFrame:
    return matrix.copy() if isinstance(matrix, pd.DataFrame) else pd.DataFrame(np.array(matrix, dtype=float))


def is_positive_semidefinite(matrix, tol: float = VALIDITY_TOLERANCE) -> bool:
    arr = np.asarray(matrix, dtype=float)
    if np.isnan(arr).any():
        return False
    sym = (arr + arr.T) / 2.0
    return bool(np.linalg.eigvalsh(sym).min() >= -tol)


def nearest_correlation_matrix(matrix, max_iter: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """Nearest matrix to `matrix` (Frobenius norm) that is symmetric positive semidefinite, has a
    unit diagonal AND non-negative off-diagonal entries. Dykstra's alternating projections onto
    those three convex sets, then an exact repair so the three properties hold to within
    VALIDITY_TOLERANCE (not merely in the limit)."""
    a = np.asarray(matrix, dtype=float)
    x = (a + a.T) / 2.0
    off_diag = ~np.eye(x.shape[0], dtype=bool)

    def project_psd(m):
        w, v = np.linalg.eigh((m + m.T) / 2.0)
        return (v * np.maximum(w, 0.0)) @ v.T

    def project_unit_diagonal(m):
        out = m.copy()
        np.fill_diagonal(out, 1.0)
        return out

    def project_non_negative(m):
        out = m.copy()
        out[off_diag] = np.maximum(m[off_diag], 0.0)
        return out

    projections = (project_psd, project_unit_diagonal, project_non_negative)
    increments = [np.zeros_like(x) for _ in projections]
    for _ in range(max_iter):
        previous = x
        for k, project in enumerate(projections):
            y = x + increments[k]
            x = project(y)
            increments[k] = y - x
        if np.linalg.norm(x - previous, "fro") <= tol * max(np.linalg.norm(x, "fro"), 1.0):
            break

    # Exact repair: make the two cheap constraints hold exactly, then remove any residual negative
    # eigenvalue by the smallest possible shrink towards the identity (which preserves both).
    x = (x + x.T) / 2.0
    x[off_diag] = np.maximum(x[off_diag], 0.0)
    np.fill_diagonal(x, 1.0)
    lowest = np.linalg.eigvalsh(x).min()
    if lowest < 0.0:
        t = -lowest / (1.0 - lowest) + 1e-12
        x = (1.0 - t) * x + t * np.eye(x.shape[0])
    return x


def validate_multiplier_correlation(matrix, tol: float = VALIDITY_TOLERANCE) -> None:
    """The matrix that feeds a multiplier must have H_ii = 1, H_ij >= 0 and H PSD. Undefined (NaN)
    correlations are skipped - they propagate to an undefined multiplier instead."""
    arr = np.asarray(matrix, dtype=float)
    if np.isnan(arr).any():
        return
    if not np.allclose(np.diag(arr), 1.0):
        raise ValueError("multiplier correlation matrix must have a unit diagonal")
    if arr.min() < -tol:
        raise ValueError(f"multiplier correlation matrix has a negative entry ({arr.min():.3g})")
    lowest = np.linalg.eigvalsh((arr + arr.T) / 2.0).min()
    if lowest < -tol:
        raise ValueError(f"multiplier correlation matrix is not positive semidefinite (min eigenvalue {lowest:.3g})")


def multiplier_correlation(raw_corr) -> tuple[pd.DataFrame, bool]:
    """Floored copy of raw_corr used only for the multiplier: negative entries -> 0.0, diagonal
    forced to 1.0, and (for more than two rules/instruments) projected to the nearest valid
    correlation matrix (still non-negative, unit diagonal) if flooring left it not positive
    semidefinite. NaNs are left as NaN. The result is always validated: unit diagonal, no negative
    entries, positive semidefinite.

    Returns (adjusted_matrix, was_projected). raw_corr itself is never modified."""
    raw = _as_frame(raw_corr)
    arr = raw.to_numpy(dtype=float, copy=True)
    arr = np.clip(arr, 0.0, None)
    np.fill_diagonal(arr, 1.0)

    projected = False
    if arr.shape[0] > 2 and not np.isnan(arr).any() and not is_positive_semidefinite(arr):
        arr = nearest_correlation_matrix(arr)
        projected = True

    validate_multiplier_correlation(arr)
    return pd.DataFrame(arr, index=raw.index, columns=raw.columns), projected


def diversification_multiplier(raw_corr, weights, cap: float) -> MultiplierResult:
    """M = min(1 / sqrt(w' C_adj w), cap), C_adj = multiplier_correlation(raw_corr).

    A non-positive w' C w gives a multiplier of 1.0; an undefined (NaN) one propagates as NaN so
    callers keep their existing handling of not-yet-defined correlations."""
    raw = _as_frame(raw_corr)
    adjusted, projected = multiplier_correlation(raw)

    w = np.asarray(weights, dtype=float).ravel()
    wcw = float(w.T @ adjusted.to_numpy() @ w)

    if np.isnan(wcw):
        uncapped = multiplier = float("nan")
    elif wcw > 0:
        uncapped = 1.0 / np.sqrt(wcw)
        multiplier = min(uncapped, cap)
    else:
        uncapped = multiplier = 1.0

    return MultiplierResult(
        multiplier=float(multiplier),
        uncapped_multiplier=float(uncapped),
        wcw=wcw,
        raw_corr=raw,
        multiplier_corr=adjusted,
        psd_projected=projected,
    )
