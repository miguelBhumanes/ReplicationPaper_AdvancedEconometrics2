'''
Section 4.2
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from replication.var import var, IRFs, fevd_number, spectral_vd_in_band
from replication.io.paths import TABLES_DIR, FIGURES_DIR

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

    U, B = var(X, p=p, intercept=intercept)
    U_eff = U[p:]

    Sigma_u = (U_eff.T @ U_eff) / U_eff.shape[0]
    C = np.linalg.cholesky(Sigma_u)

    offset = 1 if intercept else 0

    A_sum = np.zeros((n, n))
    for j in range(p):
        A_sum += B[offset + j*n : offset + (j+1)*n, :].T

    LR = np.linalg.inv(np.eye(n) - A_sum)

    M  = LR @ C
    m1 = M[0, :]

    if np.linalg.norm(m1) < 1e-12:
        Q = np.eye(n)
    else:
        q2 = np.array([-m1[1], m1[0]])
        q2 = q2 / np.linalg.norm(q2)
        q1 = np.array([ q2[1], -q2[0] ])
        Q  = np.column_stack((q1, q2))

    P = C @ Q

    Structural_Shocks_eff = U_eff @ np.linalg.inv(P).T

    IRF = IRFs(B=B, n=n, p=p, P=P, H=H, intercept=intercept)

    S = Structural_Shocks_eff

    c_eps = np.array([np.corrcoef(eps[p:], S[:, 0])[0, 1],
                      np.corrcoef(eps[p:], S[:, 1])[0, 1]])
    c_d   = np.array([np.corrcoef(d[p:],   S[:, 0])[0, 1],
                      np.corrcoef(d[p:],   S[:, 1])[0, 1]])

    score_id   = abs(c_eps[0]) + abs(c_d[1])
    score_swap = abs(c_eps[1]) + abs(c_d[0])

    if score_swap > score_id:
        perm = [1, 0]
    else:
        perm = [0, 1]

    S   = S[:, perm]
    P   = P[:, perm]
    IRF = IRF[:, :, perm]

    s0 = 1.0 if np.corrcoef(eps[p:], S[:, 0])[0, 1] >= 0 else -1.0
    s1 = 1.0 if np.corrcoef(d[p:],   S[:, 1])[0, 1] >= 0 else -1.0

    S[:, 0] *= s0
    S[:, 1] *= s1
    P[:, 0] *= s0
    P[:, 1] *= s1
    IRF[:, :, 0] *= s0
    IRF[:, :, 1] *= s1

    IRF_store[s] = IRF
    rho_eps_store[s] = np.corrcoef(eps[p:], S[:, 0])[0, 1]
    rho_d_store[s]   = np.corrcoef(d[p:],   S[:, 1])[0, 1]

# === TRUE IRFS (Example 2) =====

def true_IRF_example2(alpha, beta, gamma, theta, H):
    """
    True IRFs for Example 2 DGP in differences:
        Δa*_t = α ε_t + ε_{t-1} + θ e_t − θ e_{t-1}
        Δp_t  = b(1+α) ε_t + γ d_t − γ d_{t-1}
    with b = beta/(1-beta).
    """
    b = beta / (1 - beta)
    IRF_true = np.zeros((H + 1, 2, 2))

    for shock_idx in [0, 1]:
        eps_imp = np.zeros(H + 1)
        d_imp   = np.zeros(H + 1)
        e_imp   = np.zeros(H + 1)

        if shock_idx == 0:
            eps_imp[0] = 1.0
        else:
            d_imp[0] = 1.0

        Leps = 0.0
        Ld   = 0.0
        Le   = 0.0

        da_path = np.zeros(H + 1)
        dp_path = np.zeros(H + 1)

        for t in range(H + 1):
            da_path[t] = alpha * eps_imp[t] + Leps + theta * e_imp[t] - theta * Le
            dp_path[t] = b * (1 + alpha) * eps_imp[t] + gamma * d_imp[t] - gamma * Ld

            Leps = eps_imp[t]
            Ld   = d_imp[t]
            Le   = e_imp[t]

        IRF_true[:, 0, shock_idx] = da_path
        IRF_true[:, 1, shock_idx] = dp_path

    return IRF_true

IRF_true = true_IRF_example2(alpha=alpha, beta=beta, gamma=gamma, theta=theta, H=H)

# === COMPUTING INFORMATION SUFFICIENCY / DEFICIENCY METRIC (Example 2) =====

np.random.seed(100)

n_obs_ism = 20**4

shocks = np.random.normal(loc=0.0, scale=sigma, size=(n_obs_ism, 3))
eps_true = shocks[:, 0]
d_true   = shocks[:, 1]
e_true   = shocks[:, 2]

Leps = 0.0
Ld   = 0.0
Le   = 0.0

da_vec_ism = np.zeros(n_obs_ism)
dp_vec_ism = np.zeros(n_obs_ism)

for t in range(n_obs_ism):
    da_vec_ism[t] = deltastara(alpha, eps_true[t], Leps, theta, e_true[t], Le)
    dp_vec_ism[t] = deltap(b, alpha, gamma, eps_true[t], d_true[t], Ld)
    Leps = eps_true[t]
    Ld   = d_true[t]
    Le   = e_true[t]

X_ism = np.column_stack((da_vec_ism, dp_vec_ism))

Ks = [1, 4, 1000]
ism_store = np.zeros((3, len(Ks)))

for k_idx, K in enumerate(Ks):

    U, B = var(X_ism, p=K, intercept=intercept)

    U_eff   = U[K:]
    eps_eff = eps_true[K:]
    d_eff   = d_true[K:]
    e_eff   = e_true[K:]

    beta_eps, *_ = np.linalg.lstsq(U_eff, eps_eff, rcond=None)
    eps_hat = U_eff @ beta_eps
    R2_eps = np.var(eps_hat) / np.var(eps_eff)

    beta_d, *_ = np.linalg.lstsq(U_eff, d_eff, rcond=None)
    d_hat = U_eff @ beta_d
    R2_d = np.var(d_hat) / np.var(d_eff)

    beta_e, *_ = np.linalg.lstsq(U_eff, e_eff, rcond=None)
    e_hat = U_eff @ beta_e
    R2_e = np.var(e_hat) / np.var(e_eff)

    ism_store[0, k_idx] = 1.0 - R2_eps
    ism_store[1, k_idx] = 1.0 - R2_d
    ism_store[2, k_idx] = 1.0 - R2_e

# === REPORTED OBJECTS ===

rho_eps_all = rho_eps_store.copy()
rho_d_all   = rho_d_store.copy()

IRF_store_lvl = IRF_store.cumsum(axis=1)

IRF_median_lvl = np.median(IRF_store_lvl, axis=0)
IRF_p05_lvl    = np.quantile(IRF_store_lvl, 0.05, axis=0)
IRF_p95_lvl    = np.quantile(IRF_store_lvl, 0.95, axis=0)

IRF_true_lvl = IRF_true.cumsum(axis=0)

# TABLE 4: informational deficiency δ(K)
table4 = pd.DataFrame(
    ism_store,
    index=[r"Shock $\varepsilon_t$", r"Shock $d_t$", r"Shock $e_t$"],
    columns=[f"δ({K})" for K in Ks],
)
table4.index.name = "Shocks of interest"
table4.to_csv(TABLES_DIR / "example2_table4_deficiency.csv")

# FIGURE 2: IRFs (LEVELS = cumulative sums of Δ-IRFs) + correlation histograms
shock_names = ["Technology shock", "Non-technology shock"]
var_names   = ["Technology", "Stock prices"]

x = np.arange(H + 1)

fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 1.0, 0.8], hspace=0.35, wspace=0.25)

axs = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
                [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]])

axs[0, 0].set_title(shock_names[0], fontsize=12)
axs[0, 1].set_title(shock_names[1], fontsize=12)

for col, shock_idx_plot in enumerate([0, 1]):
    for row, var_idx_plot in enumerate([0, 1]):
        ax = axs[row, col]

        if var_idx_plot == 0:
            ax.set_ylim(-1, 2)
        elif var_idx_plot == 1:
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

legend_elems = [
    Patch(alpha=0.25, label="90% confidence interval"),
    Line2D([0], [0], linestyle="--", color="black", linewidth=2.0, label="Median estimated IRF"),
    Line2D([0], [0], linestyle="-",  color="red",   linewidth=2.0, label="True IRF"),
]
axs[0, 1].legend(handles=legend_elems, loc="upper right", fontsize=9, frameon=False)

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

fig.savefig(FIGURES_DIR / "example2_figure2.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "example2_figure2.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(f"  {TABLES_DIR / 'example2_table4_deficiency.csv'}")
print(f"  {FIGURES_DIR / 'example2_figure2.png'}")
print(f"  {FIGURES_DIR / 'example2_figure2.pdf'}")
