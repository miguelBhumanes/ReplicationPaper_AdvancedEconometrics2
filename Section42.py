'''
Section 4.2
'''

# === PACKAGES =======

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from aux_VAR import var, IRFs

# ==== PARAMETERS ====

n_obs = 200
n_sim = 1000

alpha = 0.5
beta = 0.99
gamma = 20
theta = 0.5

b = beta / (1-beta)

sigma = 1
p = 4
H = 5

intercept = False

# === DEFINITIONS =====

def deltastara(alpha, eps, Leps, theta, e, Le):
    return alpha * eps + Leps + theta * e - theta * Le

def deltap(b, alpha, gamma, eps, d, Ld):
    return b * (1+alpha) * eps + gamma * d - gamma * Ld

# ==== STORAGE ARRAYS ====
n = 2
IRF_store     = np.zeros((n_sim, H+1, n, n))  # (sim, h, var, shock)
rho_eps_store = np.zeros(n_sim)
rho_d_store   = np.zeros(n_sim)

# ==== MONTE CARLO LOOP ====
np.random.seed(100)

for s in range(n_sim):

    # --- Simulate DGP (3 shocks) ---
    shocks = np.random.normal(loc=0.0, scale=sigma, size=(n_obs, 3))
    eps = shocks[:, 0]
    d   = shocks[:, 1]
    e   = shocks[:, 2]

    Leps = 0.0
    Ld   = 0.0
    Le   = 0.0

    deltastara_vec = np.zeros(n_obs)
    deltap_vec     = np.zeros(n_obs)

    for t in range(n_obs):
        deltastara_vec[t] = deltastara(alpha, eps[t], Leps, theta, e[t], Le)
        deltap_vec[t]     = deltap(b, alpha, gamma, eps[t], d[t], Ld)
        Leps = eps[t]
        Ld   = d[t]
        Le   = e[t]

    X = np.column_stack((deltastara_vec, deltap_vec))

    # --- Reduced-form VAR ---
    U, B = var(X, p=p, intercept=intercept)
    U_eff = U[p:]  # drop padded zeros for covariances/correlations

    Sigma_u = (U_eff.T @ U_eff) / U_eff.shape[0]
    C = np.linalg.cholesky(Sigma_u)

    # --- Long-run restriction identification (Example 2) ---
    offset = 1 if intercept else 0  # consistent with aux.IRFs()

    A_sum = np.zeros((n, n))
    for j in range(p):
        A_sum += B[offset + j*n : offset + (j+1)*n, :].T   # A_{j+1}

    LR = np.linalg.inv(np.eye(n) - A_sum)

    M  = LR @ C
    m1 = M[0, :]

    # guard against degenerate case
    if np.linalg.norm(m1) < 1e-12:
        Q = np.eye(n)
    else:
        q2 = np.array([-m1[1], m1[0]])
        q2 = q2 / np.linalg.norm(q2)
        q1 = np.array([ q2[1], -q2[0] ])
        Q  = np.column_stack((q1, q2))

    P = C @ Q

    # --- Structural shocks (effective sample only) ---
    Structural_Shocks_eff = U_eff @ np.linalg.inv(P).T   # (T-p, 2)

    # --- IRFs (structural) ---
    IRF = IRFs(B=B, n=n, p=p, P=P, H=H, intercept=intercept)

    # ==========================================================
    # Align shocks across simulations (permute + sign) using truth
    # ==========================================================
    S = Structural_Shocks_eff  # shorthand

    # correlations between true shocks and identified shocks
    c_eps = np.array([np.corrcoef(eps[p:], S[:, 0])[0, 1],
                      np.corrcoef(eps[p:], S[:, 1])[0, 1]])
    c_d   = np.array([np.corrcoef(d[p:],   S[:, 0])[0, 1],
                      np.corrcoef(d[p:],   S[:, 1])[0, 1]])

    # choose permutation that best matches (eps, d)
    score_id   = abs(c_eps[0]) + abs(c_d[1])   # (0->eps, 1->d)
    score_swap = abs(c_eps[1]) + abs(c_d[0])   # (1->eps, 0->d)

    if score_swap > score_id:
        perm = [1, 0]
    else:
        perm = [0, 1]

    # apply permutation (shock dimension = last axis / columns)
    S   = S[:, perm]
    P   = P[:, perm]
    IRF = IRF[:, :, perm]

    # set signs so corr(true, identified) is positive on the diagonal
    s0 = 1.0 if np.corrcoef(eps[p:], S[:, 0])[0, 1] >= 0 else -1.0
    s1 = 1.0 if np.corrcoef(d[p:],   S[:, 1])[0, 1] >= 0 else -1.0

    S[:, 0] *= s0
    S[:, 1] *= s1
    P[:, 0] *= s0
    P[:, 1] *= s1
    IRF[:, :, 0] *= s0
    IRF[:, :, 1] *= s1

    # --- Store aligned IRFs and correlations ---
    IRF_store[s] = IRF
    rho_eps_store[s] = np.corrcoef(eps[p:], S[:, 0])[0, 1]
    rho_d_store[s]   = np.corrcoef(d[p:],   S[:, 1])[0, 1]

# === TRUE IRFS (Example 2) =====
# True IRFs by propagating a MIT structural shock through the DGP in differences:
#   Δa*_t = α ε_t + ε_{t-1} + θ e_t − θ e_{t-1}
#   Δp_t  = b(1+α) ε_t + γ d_t − γ d_{t-1}

def true_IRF_example2(alpha, beta, gamma, theta, H):
    """
    True IRFs for Example 2 DGP in differences:
        Δa*_t = α ε_t + ε_{t-1} + θ e_t − θ e_{t-1}
        Δp_t  = b(1+α) ε_t + γ d_t − γ d_{t-1}
    with b = beta/(1-beta).

    Returns
    -------
    IRF_true : array (H+1, 2, 2)
        Variables: [Δa*, Δp]
        Shocks:    [ε (tech), d (demand)]
    Notes
    -----
    The measurement error shock e_t is present in the DGP, but the VAR is 2×2,
    so we report true IRFs only for the two economically relevant shocks (ε, d),
    consistent with the paper's focus.
    """
    b = beta / (1 - beta)
    IRF_true = np.zeros((H + 1, 2, 2))

    # Loop over shocks: 0=ε (tech), 1=d (demand)
    for shock_idx in [0, 1]:
        eps_imp = np.zeros(H + 1)
        d_imp   = np.zeros(H + 1)
        e_imp   = np.zeros(H + 1)  # set to zero always (measurement error not targeted)

        if shock_idx == 0:
            eps_imp[0] = 1.0
        else:
            d_imp[0] = 1.0

        # initial lags
        Leps = 0.0
        Ld   = 0.0
        Le   = 0.0

        da_path = np.zeros(H + 1)
        dp_path = np.zeros(H + 1)

        for t in range(H + 1):
            da_path[t] = alpha * eps_imp[t] + Leps + theta * e_imp[t] - theta * Le
            dp_path[t] = b * (1 + alpha) * eps_imp[t] + gamma * d_imp[t] - gamma * Ld

            # update lags
            Leps = eps_imp[t]
            Ld   = d_imp[t]
            Le   = e_imp[t]

        IRF_true[:, 0, shock_idx] = da_path
        IRF_true[:, 1, shock_idx] = dp_path

    return IRF_true

# build true IRFs (ε and d)
IRF_true = true_IRF_example2(alpha=alpha, beta=beta, gamma=gamma, theta=theta, H=H)

# === COMPUTING INFORMATION SUFFICIENCY / DEFICIENCY METRIC (Example 2) =====

# delta_i(K) = 1 - R^2 from projecting true shock i on VAR(K) reduced-form residuals

np.random.seed(100)

# Simulating a long sample from Example 2 DGP (3 shocks, VAR uses 2 variables: Δa*, Δp)
n_obs_ism = 20**4   # 160,000 (same style as your block; change if you want smaller)

shocks = np.random.normal(loc=0.0, scale=sigma, size=(n_obs_ism, 3))
eps_true = shocks[:, 0]   # technology shock ε
d_true   = shocks[:, 1]   # demand shock d
e_true   = shocks[:, 2]   # measurement error shock e (nuisance)

Leps = 0.0
Ld   = 0.0
Le   = 0.0

da_vec_ism = np.zeros(n_obs_ism)   # Δa*
dp_vec_ism = np.zeros(n_obs_ism)   # Δp

for t in range(n_obs_ism):
    da_vec_ism[t] = deltastara(alpha, eps_true[t], Leps, theta, e_true[t], Le)
    dp_vec_ism[t] = deltap(b, alpha, gamma, eps_true[t], d_true[t], Ld)
    Leps = eps_true[t]
    Ld   = d_true[t]
    Le   = e_true[t]

X_ism = np.column_stack((da_vec_ism, dp_vec_ism))

# storage: rows = shocks [eps, d], cols = K in [1,4,1000]
Ks = [1, 4, 1000]
ism_store = np.zeros((3, len(Ks)))  # delta (deficiency); sufficiency = 1 - delta

for k_idx, K in enumerate(Ks):

    U, B = var(X_ism, p=K, intercept=intercept)

    # drop padded rows for projection
    U_eff   = U[K:]             # (T-K, 2)
    eps_eff = eps_true[K:]      # (T-K,)
    d_eff   = d_true[K:]        # (T-K,)
    e_eff   = e_true[K:]

    # project ε on reduced-form residuals U_eff
    beta_eps, *_ = np.linalg.lstsq(U_eff, eps_eff, rcond=None)
    eps_hat = U_eff @ beta_eps
    R2_eps = np.var(eps_hat) / np.var(eps_eff)

    # project d on reduced-form residuals U_eff
    beta_d, *_ = np.linalg.lstsq(U_eff, d_eff, rcond=None)
    d_hat = U_eff @ beta_d
    R2_d = np.var(d_hat) / np.var(d_eff)

    # project e as well
    beta_e, *_ = np.linalg.lstsq(U_eff, e_eff, rcond=None)
    e_hat = U_eff @ beta_e
    R2_e = np.var(e_hat) / np.var(e_eff)

    # informational deficiency delta = 1 - R^2
    ism_store[0, k_idx] = 1.0 - R2_eps   # for shock ε (tech)
    ism_store[1, k_idx] = 1.0 - R2_d     # for shock d (demand)
    ism_store[2, k_idx] = 1.0 - R2_e   # for the measurement error

# === REPORTED OBJECTS ===

rho_eps_all = rho_eps_store.copy()
rho_d_all   = rho_d_store.copy()

# 1) Cumulate within each simulation to get LEVEL IRFs
IRF_store_lvl = IRF_store.cumsum(axis=1)   # axis=1 is horizon in IRF_store

# 2) Take quantiles/median across simulations on LEVEL IRFs
IRF_median_lvl = np.median(IRF_store_lvl, axis=0)          # (H+1, n, n)
IRF_p05_lvl    = np.quantile(IRF_store_lvl, 0.05, axis=0)
IRF_p95_lvl    = np.quantile(IRF_store_lvl, 0.95, axis=0)

# True level IRFs
IRF_true_lvl = IRF_true.cumsum(axis=0)

# === REPORTED FIGURE 2 + TABLE 4 (Example 2) =====

outdir = Path(".")
outdir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# TABLE 4: informational deficiency δ(K)  -> CSV
# -----------------------------
table4 = pd.DataFrame(
    ism_store,
    index=[r"Shock $\varepsilon_t$", r"Shock $d_t$", r"Shock $e_t$"],
    columns=[f"δ({K})" for K in Ks],
)
table4.index.name = "Shocks of interest"
table4.to_csv(outdir / "example2_table4_deficiency.csv")

# -----------------------------
# FIGURE 2: IRFs (LEVELS = cumulative sums of Δ-IRFs) + correlation histograms
# -----------------------------
shock_names = ["Technology shock", "Non-technology shock"]
var_names   = ["Technology", "Stock prices"]   # matches Figure 2 labels

x = np.arange(H + 1)

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 1.0, 0.8], hspace=0.35, wspace=0.25)

axs = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
                [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]])

# Column titles
axs[0, 0].set_title(shock_names[0], fontsize=12)
axs[0, 1].set_title(shock_names[1], fontsize=12)

# IRF panels: rows 0-1, cols 0-1
for col, shock_idx_plot in enumerate([0, 1]):
    for row, var_idx_plot in enumerate([0, 1]):
        ax = axs[row, col]

        # fixed y-limits by variable
        if var_idx_plot == 0:      # Technology
            ax.set_ylim(-1, 2)
        elif var_idx_plot == 1:    # Stock prices
            ax.set_ylim(-50, 200)


        true_line = IRF_true_lvl[:, var_idx_plot, shock_idx_plot]
        med_line  = IRF_median_lvl[:, var_idx_plot, shock_idx_plot]
        lo_line   = IRF_p05_lvl[:, var_idx_plot, shock_idx_plot]
        hi_line   = IRF_p95_lvl[:, var_idx_plot, shock_idx_plot]

        ax.fill_between(x, lo_line, hi_line, alpha=0.25, linewidth=0)
        ax.plot(x, med_line, linestyle="--", linewidth=2.0)
        ax.plot(x, true_line, color="red", linestyle="-", linewidth=2.0)
        ax.axhline(0.0, linewidth=0.8)

        if col == 0:
            ax.set_ylabel(var_names[var_idx_plot], fontsize=11)

        ax.set_xlabel("Horizon", fontsize=10)
        ax.tick_params(labelsize=9)

# Legend (once)
legend_elems = [
    Patch(alpha=0.25, label="90% confidence interval"),
    Line2D([0], [0], linestyle="--", color="black", linewidth=2.0, label="Median estimated IRF"),
    Line2D([0], [0], linestyle="-",  color="red",   linewidth=2.0, label="True IRF"),
]
axs[0, 1].legend(handles=legend_elems, loc="upper right", fontsize=9, frameon=False)

# Histogram panels (row 2): correlations
hist_bins = 30

axs[2, 0].hist(rho_eps_all[np.isfinite(rho_eps_all)], bins=hist_bins, edgecolor="white")
axs[2, 0].set_xlabel("Technology shock", fontsize=10)
axs[2, 0].set_ylabel("Count", fontsize=10)
axs[2, 0].tick_params(labelsize=9)

axs[2, 1].hist(rho_d_all[np.isfinite(rho_d_all)], bins=hist_bins, edgecolor="white")
axs[2, 1].set_xlabel("Non-technology shock", fontsize=10)
axs[2, 1].set_ylabel("Count", fontsize=10)
axs[2, 1].tick_params(labelsize=9)

fig.suptitle("Example 2 — Impulse Responses and Shock Recovery", fontsize=14, y=0.98)

fig.savefig(outdir / "example2_figure2.png", dpi=300, bbox_inches="tight")
fig.savefig(outdir / "example2_figure2.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(" - example2_table4_deficiency.csv")
print(" - example2_figure2.png")
print(" - example2_figure2.pdf")
