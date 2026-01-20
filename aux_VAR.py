'''
File with auxiliary functions
'''

import numpy as np

# === FUNCTION TO ESTIMATE VAR =====

# I dont care about confidence intervals, because I am going to bootstrap using this function. 

def var(Y, p, intercept=True):

    '''
    Takes

    Y: a numpy array where the columns are the variables and the rows the time periods, with the first row being the first time period
    p: scalar. Number of lags. Order of the VAR
    intercept: boolean. Whether the VAR fits a intercept or not.

    Returns

    U: a numpy array where the columns are the residual of each variable, and the rows are the same as for Y. The first rows are zero. Its just 0 padding, because the VAR cannot estimate residuals for the first p observations (no available data).
    B: a numpy array. Each column is for one of the outcome variables. In the same order as in Y. The rows are as follows: first n rows are the coefficients for the first lag of the n variables, in the order of Y. n+1 to 2n rows are the coefficients for the second lag of the n variables. And so on... And the abosolute first column is the intercept when intercept = True.
    '''

    # The columns are variables, and the rows are time periods
    T, n = Y.shape 

    # Build regressor matrix X
    X = []
    for t in range(p, T): # range is from p to T, because earliest usable observation is p, and last is T-1
        row = np.hstack([Y[t - lag - 1] for lag in range(p)])
        if intercept:
            row = np.hstack([1.0, row])
        X.append(row)
    X = np.asarray(X)          # (T-p, n*p)

    # Dependent variable
    Y_dep = Y[p:]               # (T-p, n)

    # OLS
    B, *_ = np.linalg.lstsq(X, Y_dep, rcond=None)

    # Residuals
    U_dep = Y_dep - X @ B       # (T-p, n)

    # Pad residuals to match original shape
    U = np.zeros_like(Y)
    U[p:] = U_dep

    return U, B

# === FUNCTION TO DO CHOLESKY DECOMPOSITION =====

def cholesky(U, p, ddof=None):
    """
    Cholesky identification: compute impact matrix P such that Sigma_u = P P',
    and recover structural shocks eps_t from u_t = P eps_t.

    Parameters
    ----------
    U : array (T, n)
        Reduced-form residuals, possibly padded with zeros in first p rows.
    p : int
        VAR lag order (used to drop the padded first p rows).
    ddof : int or None
        Degrees of freedom for covariance scaling.
        If None, uses (T-p). If you prefer unbiased, set ddof=1.

    Returns
    -------
    P : array (n, n)
        Lower-triangular impact matrix.
    Sigma_u : array (n, n)
        Covariance matrix of reduced-form residuals.
    EPS : array (T, n)
        Structural shocks, padded with zeros in first p rows.
    """

    U_eff = U[p:]  # drop padding
    T_eff, n = U_eff.shape
    if T_eff <= 0:
        raise ValueError("Not enough observations after dropping p lags.")

    denom = T_eff if ddof is None else (T_eff - ddof)
    if denom <= 0:
        raise ValueError("Invalid ddof: denominator non-positive.")

    # Reduced-form covariance
    Sigma_u = (U_eff.T @ U_eff) / denom

    # Impact matrix
    P = np.linalg.cholesky(Sigma_u)

    # Ensure positive impact on own variable
    for j in range(n):
        if P[j, j] < 0:
            P[:, j] *= -1

    # Structural shocks: eps_t = P^{-1} u_t
    eps_eff = np.linalg.solve(P, U_eff.T).T  # (T-p, n)

    EPS = np.zeros_like(U)
    EPS[p:] = eps_eff

    return P, Sigma_u, EPS

# === FUNCTION TO GET IRFs ====

def IRFs(B,n,p,P,H, intercept=True):
    '''
    Docstring for IRFs

    Inputs
    
    :param B: Coefficient matrix obtained from the function var in this file
    :param n: Number of endogenous variables in the VAR
    :param p: Order of the VAR
    :param P: Identified Impact matrix
    :param H: Maximum IRF horizon
    :param intercept: Boolean. Whether you are passing a B with intercept or not.

    Returns 

    IRF: a tensor. 
    - The first dimension is each of the horizons. From 0 (contemporaneous) until H.
    - The second dimension is each of the variables. In the order passed to VAR.
    - The third dimension is each of the shocks. A structural shock to a given variable. 
    '''

    A = []
    for j in range(p):
        offset = 1 if intercept else 0
        block = B[offset + j*n : offset + (j+1)*n, :]
        A.append(block.T)

    n = P.shape[0]

    IRF = np.zeros((H + 1, n, n))
    IRF[0] = P

    for h in range(1, H + 1):
        acc = np.zeros((n, n))
        for j in range(1, p + 1):
            if h - j >= 0:
                acc += A[j - 1] @ IRF[h - j]
        IRF[h] = acc

    return IRF

# === FUNCTION TO GET A GIVEN FEVD ====

def fevd_number(IRF, var_idx, shock_idx, horizon, eps=1e-15):
    """
    Return FEVD share for one variable/shock at a given horizon.

    Parameters
    ----------
    IRF : array (H+1, n, n)
        IRF[h, i, j] = response of variable i at horizon h to shock j.
    var_idx : int
        Index of the variable (row in IRF[h]).
    shock_idx : int
        Index of the shock (column in IRF[h]).
    horizon : int
        FEVD horizon h (uses contributions from 0..h).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    share : float
        FEVD(var_idx, shock_idx, horizon)
    """

    if IRF.ndim != 3:
        raise ValueError("IRF must be a 3D array of shape (H+1, n, n).")

    Hp1, n1, n2 = IRF.shape
    if n1 != n2:
        raise ValueError("IRF must have shape (H+1, n, n).")

    if not (0 <= horizon < Hp1):
        raise ValueError(f"horizon must be between 0 and {Hp1-1}.")
    if not (0 <= var_idx < n1):
        raise ValueError(f"var_idx must be between 0 and {n1-1}.")
    if not (0 <= shock_idx < n1):
        raise ValueError(f"shock_idx must be between 0 and {n1-1}.")

    num = np.sum(IRF[:horizon+1, var_idx, shock_idx] ** 2)
    den = np.sum(IRF[:horizon+1, var_idx, :] ** 2)

    return float(num / (den + eps))

# === FUNCTION TO GET A GIVEN SPECTRAL VD ====

def spectral_vd_in_band(IRF, var_idx, shock_idx,
                        w_low=None, w_high=None,
                        max_freq=np.pi,
                        ngrid=4096,
                        eps=1e-15):
    """
    Spectral variance decomposition share for one variable/shock over a frequency band.

    Parameters
    ----------
    IRF : array (H+1, n, n)
        IRF[h, i, j] = response of variable i at horizon h to shock j (h=0..H).
        Shocks assumed orthonormal (as with Cholesky structural shocks).
    var_idx : int
        Index of variable of interest (i).
    shock_idx : int
        Index of shock (j).
    w_low : float or None
        Lower frequency bound. If None, uses minimum possible (0.0).
    w_high : float or None
        Upper frequency bound. If None, uses max_freq.
    max_freq : float
        Maximum frequency considered (default pi for standard discrete-time analysis).
    ngrid : int
        Number of grid points used for numerical integration.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    share : float
        Share of variance of var_idx in the band [w_low, w_high] attributable to shock_idx.
    """
    if IRF.ndim != 3:
        raise ValueError("IRF must be a 3D array of shape (H+1, n, n).")

    Hp1, n1, n2 = IRF.shape
    if n1 != n2:
        raise ValueError("IRF must have shape (H+1, n, n).")
    if not (0 <= var_idx < n1):
        raise ValueError(f"var_idx must be between 0 and {n1-1}.")
    if not (0 <= shock_idx < n1):
        raise ValueError(f"shock_idx must be between 0 and {n1-1}.")

    if w_low is None:
        w_low = 0.0
    if w_high is None:
        w_high = float(max_freq)

    if not (0.0 <= w_low < w_high <= max_freq):
        raise ValueError("Require 0 <= w_low < w_high <= max_freq.")

    # Frequency grid on [0, max_freq]
    w = np.linspace(0.0, float(max_freq), int(ngrid))
    dw = w[1] - w[0]

    mask = (w >= w_low) & (w <= w_high)
    w_band = w[mask]
    if w_band.size < 2:
        raise ValueError("Frequency band too narrow for the chosen ngrid.")

    # MA coefficients for chosen variable across all shocks: (K+1, n)
    psi = IRF[:, var_idx, :]          # (H+1, n)
    K = Hp1 - 1

    # Frequency response for all shocks at once: Psi_all (n, nb)
    k = np.arange(K + 1)[:, None]                         # (K+1, 1)
    E = np.exp(-1j * k * w_band[None, :])                 # (K+1, nb)
    Psi_all = (psi.T @ E)                                 # (n, nb)

    contrib = np.abs(Psi_all[shock_idx, :])**2            # (nb,)
    total   = np.sum(np.abs(Psi_all)**2, axis=0)          # (nb,)

    num = np.sum(contrib) * dw
    den = np.sum(total) * dw

    return float(num / (den + eps))
