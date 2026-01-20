"""
Generic utilities for solving and simulating linear rational expectations (LRE) models.

Expected model input format (your matrices):
    X0 x_t = X1 x_{t-1} + X2 E_t[x_{t+1}] + X3 u_t

This module:
  1) Builds an augmented Sims/GENSYS representation in y_t = [x_t; f_t],
     where f_t := E_t[x_{t+1}] and the forecast-error identity is enforced.
  2) Runs eigenvalue diagnostics on the generalized pencil (g0, g1).
  3) Provides a clean wrapper to solve with a user-supplied gensys implementation.
  4) Provides simulation and IRF helpers.

Important:
  - You must provide a working gensys function from a tested port.
  - This module does NOT modify your model matrices. Keep model-specific fixes in your main file.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

try:
    from scipy import linalg as spla
except ImportError as e:
    raise ImportError("aux_DSGE.py requires scipy (scipy.linalg). Install scipy.") from e


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class PencilDiagnostics:
    eigvals: np.ndarray
    n_stable: int
    n_unit: int
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class LRESolution:
    # Augmented reduced form (recommended to simulate):
    # y_t = G1 y_{t-1} + C + impact u_t
    G1: np.ndarray
    C: np.ndarray
    impact: np.ndarray

    # Existence / uniqueness flags as returned by gensys (if available)
    eu_exist: Optional[bool]
    eu_unique: Optional[bool]

    # Diagnostics on the generalized pencil
    diagnostics: PencilDiagnostics

    # Dimensions
    nx: int
    nu: int


# -----------------------------
# Core: build augmented system
# -----------------------------

def validate_lre_mats(X0: np.ndarray, X1: np.ndarray, X2: np.ndarray, X3: np.ndarray) -> Tuple[int, int]:
    """
    Validate shapes and types of the LRE matrices.

    Returns
    -------
    nx, nu
    """
    X0 = np.asarray(X0, float)
    X1 = np.asarray(X1, float)
    X2 = np.asarray(X2, float)
    X3 = np.asarray(X3, float)

    if X0.ndim != 2 or X0.shape[0] != X0.shape[1]:
        raise ValueError("X0 must be square (nx x nx).")
    nx = X0.shape[0]

    for name, A in [("X1", X1), ("X2", X2)]:
        if A.shape != (nx, nx):
            raise ValueError(f"{name} must have shape {(nx, nx)}.")

    if X3.ndim != 2 or X3.shape[0] != nx:
        raise ValueError("X3 must have shape (nx, nu).")
    nu = X3.shape[1]
    if nu <= 0:
        raise ValueError("X3 must have at least one shock (nu >= 1).")

    return nx, nu


def build_augmented_system(
    X0: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    *,
    lead_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an augmented Sims/GENSYS system from the LRE matrices.

    Original:
        X0 x_t = X1 x_{t-1} + X2 E_t[x_{t+1}] + X3 u_t

    We augment ONLY the variables that actually appear in X2 (i.e., columns of X2
    with any nonzero). Let lead_idx be those indices and k = len(lead_idx).

    Define f_t := E_t[x_{t+1, lead_idx}] (k-dimensional) and y_t := [x_t; f_t].

    The augmented system has the Sims/GENSYS form:
        g0 y_t = g1 y_{t-1} + c + psi u_t + pi eta_t
    where eta_t are the k-dimensional one-step-ahead forecast errors for the lead variables.

    Returns
    -------
    g0, g1, c, psi, pi
    """
    nx, nu = validate_lre_mats(X0, X1, X2, X3)

    X0 = np.asarray(X0, float)
    X1 = np.asarray(X1, float)
    X2 = np.asarray(X2, float)
    X3 = np.asarray(X3, float)

    # Variables that appear with a lead (any nonzero column in X2)
    lead_idx = np.where(np.any(np.abs(X2) > lead_tol, axis=0))[0]
    k = int(lead_idx.size)
    if k == 0:
        raise ValueError("X2 has no nonzero columns: no forward-looking terms detected.")

    # Selector E picks x_{lead_idx,t} from x_t: shape (k, nx)
    E = np.zeros((k, nx))
    for j, i in enumerate(lead_idx):
        E[j, i] = 1.0

    # y_t = [x_t ; f_t], where f_t := E_t[x_{t+1, lead_idx}] is k-dimensional
    g0 = np.block([
        [X0,               -X2[:, lead_idx]],
        [E,                np.zeros((k, k))]
    ])

    g1 = np.block([
        [X1,               np.zeros((nx, k))],
        [np.zeros((k, nx)), np.eye(k)]
    ])

    c = np.zeros(nx + k)

    psi = np.vstack([
        X3,
        np.zeros((k, nu))
    ])

    # Forecast errors only for the k lead variables
    pi = np.vstack([
        np.zeros((nx, k)),
        np.eye(k)
    ])

    return g0, g1, c, psi, pi


# -----------------------------
# Diagnostics
# -----------------------------

def pencil_diagnostics(
    g0: np.ndarray,
    g1: np.ndarray,
    *,
    div: float = 1.0000001,
    unit_tol: float = 1e-7
) -> PencilDiagnostics:
    """
    Diagnostics for generalized eigenvalues of det(g0 - lambda g1)=0.

    Parameters
    ----------
    div : float
        Stability cutoff. gensys typically uses a div slightly above 1.
    unit_tol : float
        Threshold for "near-unit" eigenvalues.

    Returns
    -------
    PencilDiagnostics
    """
    g0 = np.asarray(g0, float)
    g1 = np.asarray(g1, float)

    warnings = []

    # Rank warning (not a proof of failure but a strong signal)
    if np.linalg.matrix_rank(g0) < g0.shape[0]:
        warnings.append(
            "g0 is rank-deficient. Often indicates missing equation, unused variable, "
            "or mis-specified shock buffer / timing."
        )

    # QZ with ordering (outside unit circle last)
    # 'ouc' = order unstable eigenvalues (|lambda|>1) to the bottom-right block
    S, T, alpha, beta, Q, Z = spla.ordqz(g0, g1, sort="ouc")

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = alpha / beta
    abs_lam = np.abs(lam)

    if np.any(~np.isfinite(lam)):
        warnings.append("Non-finite generalized eigenvalues (inf/nan). Likely singular pencil (g0,g1).")

    n_unit = int(np.sum(np.isfinite(abs_lam) & (np.abs(abs_lam - 1.0) <= unit_tol)))
    if n_unit > 0:
        warnings.append(f"{n_unit} eigenvalue(s) are very close to 1 (unit/near-unit root).")

    n_stable = int(np.sum(np.isfinite(abs_lam) & (abs_lam < div)))

    return PencilDiagnostics(eigvals=lam, n_stable=n_stable, n_unit=n_unit, warnings=tuple(warnings))


# -----------------------------
# Solve wrapper (requires gensys)
# -----------------------------

def solve_lre(
    X0: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    gensys: Callable,
    *,
    div: float = 1.0000001,
    unit_tol: float = 1e-7,
    require_unique: bool = True
) -> LRESolution:
    """
    Solve an LRE model using a user-supplied gensys implementation.

    Parameters
    ----------
    gensys : Callable
        A tested implementation of Sims' gensys.
        Expected signature (common):
            G1, C, impact, eu = gensys(g0, g1, c, psi, pi, div=div)
        If your gensys returns more outputs, that's OK (we take the first 4).

    Returns
    -------
    LRESolution
    """
    nx, nu = validate_lre_mats(X0, X1, X2, X3)

    g0, g1, c, psi, pi = build_augmented_system(X0, X1, X2, X3)
    diag = pencil_diagnostics(g0, g1, div=div, unit_tol=unit_tol)

    # Call gensys
    out = gensys(g0, g1, c, psi, pi, div=div)
    if not isinstance(out, (tuple, list)) or len(out) < 3:
        raise ValueError("gensys must return at least (G1, C, impact, ...)")

    G1 = np.asarray(out[0], float)
    C = np.asarray(out[1], float).reshape(-1)
    impact = np.asarray(out[2], float)

    # Try to parse eu if present
    eu_exist = eu_unique = None
    if len(out) >= 4:
        eu = np.asarray(out[3]).reshape(-1)
        if eu.size >= 2:
            eu_exist = bool(eu[0] > 0.5)
            eu_unique = bool(eu[1] > 0.5)

    # Add determinacy warnings
    warnings = list(diag.warnings)
    if eu_exist is False:
        warnings.append("GENSYS reports: no solution exists (existence/BK failure).")
    if eu_unique is False:
        warnings.append("GENSYS reports: solution not unique (indeterminacy).")
    if require_unique and (eu_unique is False):
        warnings.append("You set require_unique=True but model is indeterminate. Simulations depend on selection.")

    # Validate solved shapes
    if G1.shape != (2 * nx, 2 * nx):
        raise ValueError(f"G1 shape {G1.shape} unexpected; expected {(2*nx, 2*nx)}.")
    if impact.shape[0] != 2 * nx or impact.shape[1] != nu:
        raise ValueError(f"impact shape {impact.shape} unexpected; expected {(2*nx, nu)}.")
    if C.shape[0] != 2 * nx:
        raise ValueError(f"C length {C.shape[0]} unexpected; expected {2*nx}.")

    diag2 = PencilDiagnostics(
        eigvals=diag.eigvals,
        n_stable=diag.n_stable,
        n_unit=diag.n_unit,
        warnings=tuple(warnings),
    )

    return LRESolution(
        G1=G1,
        C=C,
        impact=impact,
        eu_exist=eu_exist,
        eu_unique=eu_unique,
        diagnostics=diag2,
        nx=nx,
        nu=nu
    )

# -----------------------------
# Small convenience helpers
# -----------------------------

def format_warnings(solution: LRESolution, max_eigs: int = 10) -> str:
    """
    Pretty-print warnings + key stats (useful in logs).
    """
    lines = []
    if solution.eu_exist is not None and solution.eu_unique is not None:
        lines.append(f"GENSYS existence={solution.eu_exist}, uniqueness={solution.eu_unique}")
    lines.append(f"stable eigs count={solution.diagnostics.n_stable}, near-unit eigs={solution.diagnostics.n_unit}")
    if solution.diagnostics.warnings:
        lines.append("Warnings:")
        for w in solution.diagnostics.warnings:
            lines.append(f" - {w}")

    # Show a few eigenvalues by magnitude (helpful when debugging)
    lam = solution.diagnostics.eigvals
    finite = lam[np.isfinite(lam)]
    if finite.size > 0:
        idx = np.argsort(np.abs(finite))[::-1]
        top = finite[idx[:max_eigs]]
        lines.append("Largest |eig| (top): " + ", ".join([f"{z:.4g}" for z in top]))

    return "\n".join(lines)
