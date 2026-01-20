'''
Section 4.3
'''

# === PACKAGES AND IMPORTS =======

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aux_VAR import var
data = np.load("dsge_solution.npz")
F = data["F"]
G = data["G"]

# =========================
# Table 7 replication (simulation + VAR residual projection)
# ========================

# Choose, for each VAR specification S1-S10, which indices of x_t go into the VAR
specs_var_indices = {
    "S1":  [2,1],  # e.g. [idx_TFP, idx_Inv]
    "S2":  [2,8],
    "S3":  [2,1,13,3],
    "S4":  [2,15,1,13,3],
    "S5":  [2,15,1,13,5],
    "S6":  [2,15,8,13,3],
    "S7":  [2,15,8,13,5],
    "S8":  [2,15,8,13,3,5],
    "S9":  [2,15,1,8,13,3],
    "S10": [2,15,1,8,13,3,5],
}

# Shock ordering in u_t (columns of G) 
shock_names = [
    "Mon. pol.",
    "Gov't exp.",
    "Wage markup",
    "Temp. tech.",
    "News",
    "Price markup",
    "Inv. spec."
]

# Map each Table-7 shock name to the correct index in u_t
shock_index_map = {
    "News":         4, 
    "Temp. tech.":  3,
    "Price markup": 5,
    "Wage markup":  2,
    "Gov't exp.":   1,
    "Inv. spec.":   6,
    "Mon. pol.":    0,
}

# Simulation / VAR settings you requested
T = 10**4
K = 10
burn = 1000
seed = 100
intercept = False
Sigma_u = np.eye(7)  # orthogonal structural shocks

# ------------------------------------------------------------
# CORE FUNCTIONS
# ------------------------------------------------------------

def simulate_state_space(F, G, T, burn=2000, seed=0, Sigma_u=None, x0=None):
    """
    Simulate x_t = F x_{t-1} + G u_t

    Returns
    -------
    x : (T, n_state)
    u : (T, n_shock)
    """
    F = np.asarray(F)
    G = np.asarray(G)
    n_state = F.shape[0]
    n_shock = G.shape[1]

    rng = np.random.default_rng(seed)

    if Sigma_u is None:
        Sigma_u = np.eye(n_shock)
    Sigma_u = np.asarray(Sigma_u)

    u_full = rng.multivariate_normal(
        mean=np.zeros(n_shock), cov=Sigma_u, size=T + burn
    )

    x_full = np.zeros((T + burn, n_state), dtype=float)
    if x0 is not None:
        x_full[0] = np.asarray(x0)

    for t in range(1, T + burn):
        x_full[t] = F @ x_full[t - 1] + G @ u_full[t]

    return x_full[burn:], u_full[burn:]


def deficiency_table7_for_one_spec(X, U_true, K, intercept=False):
    """
    Given VAR data X (T,n) and true shocks U_true (T,q),
    compute delta_i(K)=1-R^2 from projecting each shock on VAR(K) reduced-form residuals.

    Returns
    -------
    delta : (q,) array
    """
    # Estimate reduced-form VAR(K) on X
    U_hat, B = var(X, p=K, intercept=intercept)

    # Align (drop padded rows)
    U_eff = U_hat[K:]         # (T-K, n)
    Utrue_eff = U_true[K:]    # (T-K, q)

    # Multi-response OLS: Utrue_eff = U_eff @ Beta + err
    # Beta: (n, q)
    Beta, *_ = np.linalg.lstsq(U_eff, Utrue_eff, rcond=None)
    Uproj = U_eff @ Beta

    # Columnwise R^2 via variance ratio (means ~0 in simulation)
    var_true = np.var(Utrue_eff, axis=0)
    var_proj = np.var(Uproj, axis=0)
    R2 = np.divide(var_proj, var_true, out=np.zeros_like(var_proj), where=var_true > 0)

    delta = 1.0 - R2
    return delta


def build_table7_dataframe(F, G, specs_var_indices, shock_names, shock_index_map,
                           T=10_000, K=1000, burn=2000, seed=0, intercept=False, Sigma_u=None,
                           round_decimals=3):
    """
    Returns a DataFrame with:
      index: S1..S10
      columns: Table-7 shock columns (News, Temp. tech., ..., Mon. pol.)
      values: delta_i(K), rounded like the paper table
    """
    # validate inputs
    missing_specs = [k for k, v in specs_var_indices.items() if v is None]
    if missing_specs:
        raise ValueError(f"Fill specs_var_indices for: {missing_specs}")

    missing_shocks = [k for k, v in shock_index_map.items() if v is None]
    if missing_shocks:
        raise ValueError(f"Fill shock_index_map for: {missing_shocks}")

    # simulate once (common underlying x,u), then slice X per spec
    x, u = simulate_state_space(F, G, T=T, burn=burn, seed=seed, Sigma_u=Sigma_u)

    # reorder/select shocks into Table-7 ordering
    shock_indices = [shock_index_map[name] for name in shock_names]
    u_table7 = u[:, shock_indices]  # (T, 7)

    # compute deltas spec-by-spec
    out = {}
    for sname in ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"]:
        idx = specs_var_indices[sname]
        X = x[:, idx]
        delta_vec = deficiency_table7_for_one_spec(X, u_table7, K=K, intercept=intercept)
        out[sname] = delta_vec
        print(f"{sname} finished") # printing progress

    df = pd.DataFrame.from_dict(out, orient="index", columns=shock_names)
    df.index.name = "Specification"
    df = df.round(round_decimals)
    return df

# ------------------------------------------------------------
# RUN + SAVE
# ------------------------------------------------------------

table7_df = build_table7_dataframe(
    F=F, G=G,
    specs_var_indices=specs_var_indices,
    shock_names=shock_names,
    shock_index_map=shock_index_map,
    T=T, K=K, burn=burn, seed=seed,
    intercept=intercept, Sigma_u=Sigma_u,
    round_decimals=3
)
print(table7_df)

table7_df.to_csv("table7_replication.csv")
print("Saved: table7_replication.csv")



# =========================
# Figure 7 replication (simulation + VAR residual projection)
# ========================

# ==========================================================
# Config
# ==========================================================
DATA_PATH = "fgs-data.txt"

P_LAGS = 4
H_IRF = 30
H_RESTR = 20
N_DRAWS = 500
SEED = 100

NEWS_SHOCK_IDX_DSGE = 4
NEWS_SHOCK_SIZE = 1.0

NEWS_SHOCK_IDX_EMP = 1

USE_INTERCEPT = True

# ==========================================================
# Data: load + transform to stationary series for the VAR
# ==========================================================
def load_and_transform_fgs(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=0,
        index_col=0,
        parse_dates=True,
    ).apply(pd.to_numeric, errors="coerce").sort_index()

    df = df_raw.copy()

    df["dlog_TFP_100"] = 100.0 * np.log(df["TFP"]).diff()
    df["dlog_Y_100"]   = 100.0 * np.log(df["Y"]).diff()
    df["dlog_C_100"]   = 100.0 * np.log(df["C"]).diff()
    df["dlog_I_100"]   = 100.0 * np.log(df["I"]).diff()

    df["H_100"] = df["H"]

    df["pi_100"]      = df["pi"]
    df["ffr"]         = df["FFR"]
    df["dlog_W_100"]  = df["W"]

    out = df[[
        "dlog_TFP_100",
        "dlog_Y_100",
        "dlog_C_100",
        "dlog_I_100",
        "H_100",
        "pi_100",
        "ffr",
        "dlog_W_100",
    ]].dropna()

    return out

# ==========================================================
# VAR/BVAR helpers (diffuse prior)
# ==========================================================
def _lagged_regressors(X: np.ndarray, p: int, intercept: bool):
    T, n = X.shape
    Y = X[p:, :]
    Zlags = np.hstack([X[p-j:T-j, :] for j in range(1, p + 1)])
    if intercept:
        Z = np.hstack([np.ones((T - p, 1)), Zlags])
    else:
        Z = Zlags
    return Y, Z


def _ols_var(X: np.ndarray, p: int, intercept: bool):
    Y, Z = _lagged_regressors(X, p, intercept=intercept)
    ZZ = Z.T @ Z
    ZZ_inv = np.linalg.inv(ZZ)
    B_ols = ZZ_inv @ (Z.T @ Y)
    U_eff = Y - Z @ B_ols
    SSE = U_eff.T @ U_eff
    return B_ols, U_eff, ZZ_inv, SSE


def _wishart_rvs(df: int, scale: np.ndarray):
    m = scale.shape[0]
    chol_scale = np.linalg.cholesky(scale)

    A = np.zeros((m, m))
    for i in range(m):
        A[i, i] = np.sqrt(np.random.chisquare(df - i))
        for j in range(i):
            A[i, j] = np.random.normal()

    LA = chol_scale @ A
    return LA @ LA.T


def _invwishart_rvs(df: int, scale: np.ndarray):
    scale_inv = np.linalg.inv(scale)
    W = _wishart_rvs(df, scale_inv)
    return np.linalg.inv(W)


def _var_A_mats(B: np.ndarray, n: int, p: int, intercept: bool):
    if intercept:
        B = B[1:, :]
    return [B[j * n:(j + 1) * n, :].T for j in range(p)]


def _ma_mats_from_var(B: np.ndarray, n: int, p: int, H: int, intercept: bool):
    A = _var_A_mats(B, n, p, intercept=intercept)
    Psi = [np.eye(n)]
    for h in range(1, H + 1):
        Ph = np.zeros((n, n))
        for j in range(1, min(p, h) + 1):
            Ph += A[j - 1] @ Psi[h - j]
        Psi.append(Ph)
    return Psi


def _identify_news_surprise_rotation(B: np.ndarray, Sigma_u: np.ndarray, n: int, p: int, H_restr: int = 20, intercept: bool = False):
    C = np.linalg.cholesky(Sigma_u)

    Psi = _ma_mats_from_var(B, n, p, H_restr, intercept=intercept)
    S_H = np.zeros((n, n))
    for h in range(H_restr + 1):
        S_H += Psi[h]

    c0 = C[0, :].copy()
    nc0 = np.linalg.norm(c0)
    if nc0 < 1e-14:
        raise ValueError("Degenerate impact row in Cholesky; cannot impose impact restriction.")
    q1 = c0 / nc0

    g = (S_H @ C)[0, :].copy()
    g_perp = g - (q1 @ g) * q1
    ng = np.linalg.norm(g_perp)
    if ng < 1e-14:
        e = np.zeros(n)
        e[1] = 1.0
        g_perp = e - (q1 @ e) * q1
        ng = np.linalg.norm(g_perp)
        if ng < 1e-14:
            raise ValueError("Cannot construct q2 orthogonal to q1 (degenerate case).")
    q2 = g_perp / ng

    Q = np.zeros((n, n))
    Q[:, 0] = q1
    Q[:, 1] = q2

    k = 2
    tries = 0
    while k < n:
        tries += 1
        if tries > 10_000:
            raise RuntimeError("Failed to complete an orthonormal basis for Q.")
        v = np.random.normal(size=n)
        for j in range(k):
            v -= (Q[:, j] @ v) * Q[:, j]
        nv = np.linalg.norm(v)
        if nv < 1e-10:
            continue
        Q[:, k] = v / nv
        k += 1

    P = C @ Q

    tfp_level_20 = (S_H @ P)[0, 1]
    if tfp_level_20 < 0:
        P[:, 1] *= -1.0

    return P


def IRFs(B: np.ndarray, n: int, p: int, P: np.ndarray, H: int, intercept: bool = False):
    Psi = _ma_mats_from_var(B, n, p, H, intercept=intercept)
    out = np.zeros((H + 1, n, n))
    for h in range(H + 1):
        out[h, :, :] = Psi[h] @ P
    return out


def bvar_s10_irfs(X: np.ndarray, p: int = 4, H: int = 30, H_restr: int = 20, ndraws: int = 500, seed: int = 123, intercept: bool = False):
    np.random.seed(seed)

    T, n = X.shape
    B_ols, U_eff, ZZ_inv, SSE = _ols_var(X, p, intercept=intercept)
    T_eff = U_eff.shape[0]
    k = n * p + (1 if intercept else 0)

    df_post = T_eff - k
    if df_post <= n + 1:
        raise ValueError(f"Posterior df too small ({df_post}).")

    irf_draws = np.zeros((ndraws, H + 1, n, n))

    L_V = np.linalg.cholesky(ZZ_inv)

    for s in range(ndraws):
        Sigma_u = _invwishart_rvs(df=df_post, scale=SSE)
        L_S = np.linalg.cholesky(Sigma_u)

        Z = np.random.normal(size=(k, n))
        B = B_ols + (L_V @ Z @ L_S.T)

        Pmat = _identify_news_surprise_rotation(B, Sigma_u, n=n, p=p, H_restr=H_restr, intercept=intercept)
        irf_draws[s] = IRFs(B=B, n=n, p=p, P=Pmat, H=H, intercept=intercept)

    return irf_draws

# ==========================================================
# Empirical Figure-7 objects (mean + 68/90 bands) for NEWS shock
# ==========================================================
def _summarize_draws(draws_2d: np.ndarray):
    mean = draws_2d.mean(axis=0)
    lo68, hi68 = np.quantile(draws_2d, [0.16, 0.84], axis=0)
    lo90, hi90 = np.quantile(draws_2d, [0.05, 0.95], axis=0)
    return mean, (lo68, hi68), (lo90, hi90)


def empirical_figure7_objects(irf_draws: np.ndarray, news_shock_idx: int = 1):
    emp = irf_draws[:, :, :, news_shock_idx]
    H = emp.shape[1] - 1

    names = ["TFP", "GDP", "Consumption", "Investment", "Hours", "Interest rate", "Inflation"]
    CUM = {0, 1, 2, 3}

    stats = {}
    for vidx, nm in enumerate(names):
        draws = emp[:, :, vidx]
        draws_plot = np.cumsum(draws, axis=1) if vidx in CUM else draws
        mean, band68, band90 = _summarize_draws(draws_plot)
        stats[nm] = {"mean": mean, "band68": band68, "band90": band90}

    return stats, np.arange(H + 1)

# ==========================================================
# DSGE theoretical IRFs
# ==========================================================
def dsge_figure7_objects(F: np.ndarray, G: np.ndarray, shock_idx: int, H: int = 30, shock_size: float = 1.0):
    n_state = F.shape[0]
    k_shock = G.shape[1]

    x_path = np.zeros((H + 1, n_state))
    x = np.zeros(n_state)

    u0 = np.zeros(k_shock)
    u0[shock_idx] = shock_size

    x = F @ x + G @ u0
    x_path[0, :] = x

    for t in range(1, H + 1):
        x = F @ x
        x_path[t, :] = x

    da   = x_path[:, 2]
    chat = x_path[:, 1]
    r    = x_path[:, 3]
    pi   = x_path[:, 5]
    ihat = x_path[:, 8]
    nhrs = x_path[:, 13]
    yhat = x_path[:, 15]

    a = np.cumsum(da)
    logC = chat + a
    logI = ihat + a
    logY = yhat + a
    logH = nhrs

    th = {
        "TFP": 1.0 * a,
        "GDP": 1.0 * logY,
        "Consumption": 1.0 * logC,
        "Investment": 1.0 * logI,
        "Hours": 1.0 * logH,
        "Interest rate": 1.0 * r,
        "Inflation": 1.0 * pi,
    }
    return th

# ==========================================================
# Plot Figure 7
# ==========================================================
def plot_figure7(emp_stats: dict, th: dict, h: np.ndarray, H: int = 30):
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 7.0), sharex=True)
    axes = axes.ravel()

    panels = [
        ("TFP", 0),
        ("GDP", 1),
        ("Consumption", 2),
        ("Investment", 3),
        ("Hours", 4),
        ("Inflation", 5),
        ("Interest rate", 6),
    ]

    for title, ax_i in panels:
        ax = axes[ax_i]
        m = emp_stats[title]["mean"]
        lo68, hi68 = emp_stats[title]["band68"]
        lo90, hi90 = emp_stats[title]["band90"]

        ax.fill_between(h, lo90, hi90, color="lightgray", alpha=1.0, linewidth=0)
        ax.fill_between(h, lo68, hi68, color="dimgray", alpha=1.0, linewidth=0)

        ax.plot(h, m, color='black', linewidth=2.5)
        ax.plot(h, th[title], linestyle="--", linewidth=2.0, color='blue')

        ax.axhline(0.0, linewidth=1.0, color="red")
        ax.set_title(title)
        ax.set_xlim(0, H)
        ax.set_xticks([5, 10, 15, 20, 25, 30])

    axes[7].axis("off")
    axes[6].set_xlabel("quarters")

    plt.tight_layout()
    plt.savefig("figure7_replication.png", dpi=300, bbox_inches="tight")
    plt.savefig("figure7_replication.pdf", bbox_inches="tight")
    plt.show()

# ==========================================================
# Run
# ==========================================================
bvar_df = load_and_transform_fgs(DATA_PATH)

X_s10 = bvar_df[[
    "dlog_TFP_100",
    "dlog_Y_100",
    "dlog_C_100",
    "dlog_I_100",
    "H_100",
    "ffr",
    "pi_100",
]].to_numpy(dtype=float)

irf_draws = bvar_s10_irfs(
    X_s10,
    p=P_LAGS,
    H=H_IRF,
    H_restr=H_RESTR,
    ndraws=N_DRAWS,
    seed=SEED,
    intercept=USE_INTERCEPT,
)

emp_stats, h = empirical_figure7_objects(irf_draws, news_shock_idx=NEWS_SHOCK_IDX_EMP)

th = dsge_figure7_objects(
    F=F,
    G=G,
    shock_idx=NEWS_SHOCK_IDX_DSGE,
    H=H_IRF,
    shock_size=NEWS_SHOCK_SIZE,
)

plot_figure7(emp_stats, th, h, H=H_IRF)

