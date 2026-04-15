"""
Factor extraction utilities: EM-PCA and Bai-Ng information criteria.
"""

import numpy as np


def em_pca(Xs: np.ndarray, k: int, max_iter=300, tol=1e-6, seed=0, verbose=False):
    """
    EM-based PCA to impute missing values and extract factors.

    Parameters
    ----------
    Xs : (T, N) array
        Standardized data, may contain NaNs.
    k : int
        Number of factors (principal components) to extract.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on relative change of missing-value imputation.
    seed : int
        Unused (kept for API compatibility).
    verbose : bool
        Print convergence progress every 25 iterations.

    Returns
    -------
    F : (T, k) array
        Factor scores.
    Lambda : (N, k) array
        Factor loadings.
    Xhat : (T, N) array
        Low-rank reconstruction.
    Ximp : (T, N) array
        Imputed data matrix (observed values unchanged, missing filled by Xhat).
    """
    T, N = Xs.shape
    mask_obs = ~np.isnan(Xs)

    # init missing with column means
    col_means = np.nanmean(Xs, axis=0)
    Ximp = Xs.copy()
    Ximp[~mask_obs] = np.take(col_means, np.where(~mask_obs)[1])

    if k == 0:
        Xhat = np.zeros_like(Ximp)
        return np.zeros((T, 0)), np.zeros((N, 0)), Xhat, Ximp

    prev = Ximp.copy()
    for it in range(max_iter):
        U, s, Vt = np.linalg.svd(Ximp, full_matrices=False)

        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]

        Xhat = (U_k * s_k) @ Vt_k

        Xnew = Ximp.copy()
        Xnew[~mask_obs] = Xhat[~mask_obs]

        num = np.linalg.norm((Xnew - prev)[~mask_obs])
        den = np.linalg.norm(prev[~mask_obs]) + 1e-12
        rel = num / den

        Ximp = Xnew
        prev = Xnew

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(f"iter {it:3d} rel_change_missing={rel:.3e}")

        if rel < tol:
            break

    # factor scores: U*s, loadings: V
    F = U_k * s_k
    Lambda = Vt_k.T
    return F, Lambda, Xhat, Ximp


def bai_ng_ic(Vk: float, k: int, N: int, T: int):
    """
    Bai-Ng (2002) information criteria ICp1, ICp2, ICp3 for choosing
    the number of factors.

    Parameters
    ----------
    Vk : float
        Residual variance for k factors: mean((Ximp - Xhat)**2).
    k : int
        Number of factors.
    N : int
        Number of series.
    T : int
        Number of time periods.

    Returns
    -------
    ICp1, ICp2, ICp3 : float
    """
    C2 = min(N, T)
    Vk = max(Vk, 1e-12)

    pen1 = ((N + T) / (N * T)) * np.log((N * T) / (N + T))
    pen2 = ((N + T) / (N * T)) * np.log(C2)
    pen3 = (np.log(C2) / C2)

    ICp1 = np.log(Vk) + k * pen1
    ICp2 = np.log(Vk) + k * pen2
    ICp3 = np.log(Vk) + k * pen3
    return ICp1, ICp2, ICp3
