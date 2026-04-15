"""
solve_news_model: direct QZ + Newton solver for the linear RE system

    X0 x_t = X1 x_{t-1} + X2 x_{t+1} + X3 u_t

Returns the stable reduced form x_t = F x_{t-1} + G u_t.
"""

import numpy as np
from scipy.linalg import ordqz


def solve_news_model(X0, X1, X2, X3,
                     tol_eig=1e-8,
                     newton_tol_res=1e-10,
                     newton_tol_step=1e-14,
                     newton_max_iter=30):
    """
    Solve the linear RE system

        X0 x_t = X1 x_{t-1} + X2 x_{t+1} + X3 u_t

    and return a stable reduced form

        x_t = F x_{t-1} + G u_t

    First, build an initial F from the QZ decomposition,
    then refine F with a Newton method that enforces

        X0 F = X1 + X2 F^2

    up to numerical precision.
    """

    n = X0.shape[0]
    k = X3.shape[1]

    # -----------------------------
    # 1. QZ step: initial F (your old code)
    # -----------------------------
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-X1,              -X0     ]
    ])

    B = np.block([
        [np.eye(n),        np.zeros((n, n))],
        [np.zeros((n, n)), -X2             ]
    ])

    S, T, alpha, beta, Q, Z = ordqz(
        A, B,
        sort='iuc',      # inside unit circle first
        output='complex'
    )

    eigvals = np.full_like(alpha, np.inf, dtype=np.complex128)
    mask = np.abs(beta) > tol_eig
    eigvals[mask] = alpha[mask] / beta[mask]

    stable_idx = np.where(np.abs(eigvals) < 1 - tol_eig)[0]
    n_stable = len(stable_idx)
    if n_stable < n:
        raise RuntimeError(
            f"Not enough stable eigenvalues to build solution: {n_stable} < {n}"
        )

    # Select n linearly independent stable eigenvectors
    chosen = []
    Z_top = Z[:n, :]
    for j in stable_idx:
        cols = chosen + [j]
        Z1_candidate = Z_top[:, cols]
        rank = np.linalg.matrix_rank(Z1_candidate, tol_eig)
        if rank == len(cols):
            chosen = cols
            if len(chosen) == n:
                break

    if len(chosen) < n:
        raise RuntimeError(
            "Could not find n linearly independent stable eigenvectors; "
            "model likely indeterminate or mis-specified."
        )

    Zs = Z[:, chosen]        # 2n x n
    Z1 = Zs[:n, :]           # n x n
    Z2 = Zs[n:, :]           # n x n

    # Initial policy matrix from QZ
    F = np.real(Z2 @ np.linalg.inv(Z1))

    # -----------------------------
    # 2. Newton refinement on F
    # -----------------------------
    def vec(M):
        return M.reshape(-1, order="F")

    def mat(v):
        return v.reshape((n, n), order="F")

    def R_of(Fmat):
        # residual of the Sylvester-type equation: X0 F = X1 + X2 F^2
        return X0 @ Fmat - X1 - X2 @ (Fmat @ Fmat)

    I_n = np.eye(n)

    Fcur = F.copy()
    Rcur = R_of(Fcur)
    res = np.max(np.abs(Rcur))
    # small diagnostic print (optional)
    print(f"Newton for F: start ||R||_inf = {res:.3e}")

    for it in range(1, newton_max_iter + 1):
        if res < newton_tol_res:
            break

        # Jacobian of R(F) w.r.t. vec(F): J = kron(I, X0) - kron(I, X2 F) - kron(F^T, X2)
        J = np.kron(I_n, X0) - np.kron(I_n, X2 @ Fcur) - np.kron(Fcur.T, X2)

        gv = vec(Rcur)
        try:
            dvec = np.linalg.solve(J, -gv)
        except Exception:
            dvec, *_ = np.linalg.lstsq(J, -gv, rcond=None)

        dF = mat(dvec)
        maxd = np.max(np.abs(dF))
        if maxd < newton_tol_step:
            print(f"Newton: step tiny at iter {it}, ||dF||_inf={maxd:.3e} — stopping")
            break

        # backtracking line search on step length
        step = 1.0
        best_F = Fcur
        best_R = Rcur
        best_res = res
        improved = False

        for _bt in range(11):
            Ftrial = Fcur + step * dF
            Rtrial = R_of(Ftrial)
            res_trial = np.max(np.abs(Rtrial))
            if (res_trial < res * (1 - 1e-12)) or (res_trial < res - 1e-12):
                best_F = Ftrial
                best_R = Rtrial
                best_res = res_trial
                improved = True
                break
            step *= 0.5

        if not improved:
            # accept smallest step even if not strictly improving — just in case
            Fcur = Ftrial
            Rcur = Rtrial
            res = res_trial
            print(f"Newton iter {it:2d}: no improvement, step={step:.3e}, ||R||_inf={res:.3e}")
        else:
            Fcur = best_F
            Rcur = best_R
            res = best_res
            print(f"Newton iter {it:2d}: ||R||_inf={res:.3e}, step={step:.3e}")

        if res < newton_tol_res:
            break

    F = Fcur
    print(f"Newton for F: final ||R||_inf = {res:.3e}")

    # -----------------------------
    # 3. Solve for G from (X0 - X2 F) G = X3
    # -----------------------------
    M = X0 - X2 @ F
    try:
        if np.linalg.cond(M) < 1e12:
            G = np.linalg.solve(M, X3)
        else:
            G = np.linalg.lstsq(M, X3, rcond=None)[0]
    except Exception:
        G = np.linalg.lstsq(M, X3, rcond=None)[0]

    return F, G, eigvals
