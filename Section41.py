'''
Section 4.1
'''

# === PACKAGES =======

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from aux_VAR import var, cholesky, IRFs, fevd_number, spectral_vd_in_band

# ==== PARAMETERS ====
n_sim = 1000
n_obs = 200

alpha = 3
gamma = 0.4
beta  = 1
sigma = 1

p = 4
H = 16

horizons = [0, 1, 4, 16]
shock_idx = 1          # second shock (v)
var_labels = ["y", "r"]

w_2y = np.pi/4         # 8 periods
w_8y = np.pi/16        # 32 periods

# === DEFINITIONS =====

def y(alpha,d,Ld,beta,Lr):
    return d + alpha * Ld - beta * Lr

def r(gamma,y,v):
    return gamma * y + v

# ==== STORAGE ARRAYS ====
n = 2
nh = len(horizons)
bands = ["<2y", "2-8y", ">8y"]  # high, medium, low

IRF_store   = np.zeros((n_sim, H+1, n, n))        # (sim, h, var, shock)
rho_d_store = np.zeros(n_sim)
rho_v_store = np.zeros(n_sim)

FEVD_store  = np.zeros((n_sim, n, nh))            # (sim, var, horizon_index) for shock_idx fixed

SPVD_store  = np.zeros((n_sim, n, 3))             # (sim, var, band_index) for shock v fixed

# ==== MONTE CARLO LOOP ====
np.random.seed(100)

for s in range(n_sim):

    shocks = np.random.normal(loc=0.0, scale=sigma, size=(n_obs, 2))
    d = shocks[:, 0]
    v = shocks[:, 1]

    Ld = 0.0
    Lr = 0.0

    y_vec = np.zeros(n_obs)
    r_vec = np.zeros(n_obs)

    for t in range(n_obs):
        y_vec[t] = y(alpha, d[t], Ld, beta, Lr)
        r_vec[t] = r(gamma, y_vec[t], v[t])
        Ld = d[t]
        Lr = r_vec[t]

    X = np.column_stack((y_vec, r_vec))

    # reduced-form VAR
    U, B = var(X, p=p, intercept=False)

    # Cholesky identification
    P, Sigma_u, Eps = cholesky(U, p=p, ddof=None)

    # IRFs
    IRF = IRFs(B=B, n=n, p=p, P=P, H=H, intercept=False)
    IRF_store[s] = IRF

    # correlations (drop padded rows)
    rho_d_store[s] = np.corrcoef(d[p:], Eps[p:, 0])[0, 1]
    rho_v_store[s] = np.corrcoef(v[p:], Eps[p:, 1])[0, 1]

    # FEVD for shock_idx across requested horizons
    for i in range(n):
        for hi, h in enumerate(horizons):
            FEVD_store[s, i, hi] = fevd_number(IRF, var_idx=i, shock_idx=shock_idx, horizon=h)

    # Spectral VD for shock v across bands
    # <2y = [pi/4, pi], 2-8y = [pi/16, pi/4], >8y = [0, pi/16]
    for i in range(n):
        SPVD_store[s, i, 0] = spectral_vd_in_band(IRF, var_idx=i, shock_idx=shock_idx, w_low=w_2y, w_high=np.pi)
        SPVD_store[s, i, 1] = spectral_vd_in_band(IRF, var_idx=i, shock_idx=shock_idx, w_low=w_8y, w_high=w_2y)
        SPVD_store[s, i, 2] = spectral_vd_in_band(IRF, var_idx=i, shock_idx=shock_idx, w_low=0.0,  w_high=w_8y)

# === REPORTED OBJECTS =====

# (1) Median FEVD table across simulations (rows = variables, cols = horizons)
fevd_median = np.median(FEVD_store, axis=0)  # (n, nh)
fevd_median_df = pd.DataFrame(
    fevd_median,
    index=var_labels,
    columns=[f"h={h}" for h in horizons],
)
fevd_median_df.index.name = "Impact variable"

# (2) Median spectral VD for each band and variable
spvd_median = np.median(SPVD_store, axis=0)  # (n, 3)
spvd_median_df = pd.DataFrame(
    spvd_median,
    index=var_labels,
    columns=bands,
)
spvd_median_df.index.name = "Impact variable"

# (3) Arrays with all correlations
rho_d_all = rho_d_store.copy()
rho_v_all = rho_v_store.copy()

# (4) IRFs median and 90% confidence intervals (5th and 95th percentiles)
IRF_median = np.median(IRF_store, axis=0)                 # (H+1, n, n)
IRF_p05    = np.quantile(IRF_store, 0.05, axis=0)         # (H+1, n, n)
IRF_p95    = np.quantile(IRF_store, 0.95, axis=0)         # (H+1, n, n)

# Reported outputs: fevd_median_df, spvd_median_df, rho_d_all, rho_v_all, IRF_median, IRF_p05, IRF_p95

# === TRUE FEVD AND SPECTRAL VD =====

# Get the true IRFs by propagating a MIT shock
def true_IRF_example1(alpha, gamma, beta, H):
    """
    True IRFs for Example 1 DGP:
        y_t = d_t + alpha d_{t-1} - beta r_{t-1}
        r_t = gamma y_t + v_t

    Returns
    -------
    IRF_true : array (H+1, 2, 2)
        Variables: [y, r]
        Shocks:    [d, v]
    """
    IRF_true = np.zeros((H + 1, 2, 2))

    # Loop over shocks: 0=d, 1=v
    for shock_idx in [0, 1]:
        # impulse shocks at t=0 only
        d_imp = np.zeros(H + 1)
        v_imp = np.zeros(H + 1)
        if shock_idx == 0:
            d_imp[0] = 1.0
        else:
            v_imp[0] = 1.0

        # initial lags
        Ld = 0.0
        Lr = 0.0

        y_path = np.zeros(H + 1)
        r_path = np.zeros(H + 1)

        for t in range(H + 1):
            y_path[t] = d_imp[t] + alpha * Ld - beta * Lr
            r_path[t] = gamma * y_path[t] + v_imp[t]

            # update lags
            Ld = d_imp[t]
            Lr = r_path[t]

        IRF_true[:, 0, shock_idx] = y_path
        IRF_true[:, 1, shock_idx] = r_path

    return IRF_true

# build true IRFs
IRF_true = true_IRF_example1(alpha=alpha, gamma=gamma, beta=beta, H=H)

# horizons like Table 1
horizons = [0, 1, 4, 16]
shock_v = 1  # [d=0, v=1]

# true FEVD for y (var 0) and r (var 1)
true_fevd_y = [fevd_number(IRF_true, var_idx=0, shock_idx=shock_v, horizon=h) for h in horizons]
true_fevd_r = [fevd_number(IRF_true, var_idx=1, shock_idx=shock_v, horizon=h) for h in horizons]

# true spectral variance decomposition

# < 2 years
true_spvd_y_high = spectral_vd_in_band(IRF_true, var_idx=0, shock_idx=shock_v, w_low=w_2y, w_high=np.pi)
true_spvd_r_high = spectral_vd_in_band(IRF_true, var_idx=1, shock_idx=shock_v, w_low=w_2y, w_high=np.pi)

# 2–8 years
true_spvd_y_medium = spectral_vd_in_band(IRF_true, var_idx=0, shock_idx=shock_v, w_low=w_8y, w_high=w_2y)
true_spvd_r_medium = spectral_vd_in_band(IRF_true, var_idx=1, shock_idx=shock_v, w_low=w_8y, w_high=w_2y)

# > 8 years
true_spvd_y_low = spectral_vd_in_band(IRF_true, var_idx=0, shock_idx=shock_v, w_low=0.0, w_high=w_8y)
true_spvd_r_low = spectral_vd_in_band(IRF_true, var_idx=1, shock_idx=shock_v, w_low=0.0, w_high=w_8y)

# === COMPUTING INFORMATION SUFFICIENCY / DEFICIENCY METRIC =====

# delta_i(K) = 1 - R^2 from projecting true shock i on VAR(K) reduced-form residuals

# Simulating the data
np.random.seed(100)

n_obs_ism = 20**4
shocks = np.random.normal(loc=0.0, scale=sigma, size=(n_obs_ism, 2))
d = shocks[:, 0]
v = shocks[:, 1]

Ld = 0.0
Lr = 0.0

y_vec_ism = np.zeros(n_obs_ism)
r_vec_ism = np.zeros(n_obs_ism)

for t in range(n_obs_ism):
    y_vec_ism[t] = y(alpha, d[t], Ld, beta, Lr)
    r_vec_ism[t] = r(gamma, y_vec_ism[t], v[t])
    Ld = d[t]
    Lr = r_vec_ism[t]

X_ism = np.column_stack((y_vec_ism, r_vec_ism))

# storage: rows = shocks [d, v], cols = K in [1,4,1000]
Ks = [1, 4, 1000]
ism_store = np.zeros((2, len(Ks)))  # will store delta (deficiency); sufficiency = 1-delta

for k_idx, K in enumerate(Ks):

    U, B = var(X_ism, p=K, intercept=False)

    # drop padded rows for projection
    U_eff = U[K:]              # (T-K, n)
    d_eff = d[K:]              # (T-K,)
    v_eff = v[K:]              # (T-K,)

    # project each true shock on reduced-form residuals
    # (no intercept needed: residuals are mean ~ 0; but harmless if you add one)
    beta_d, *_ = np.linalg.lstsq(U_eff, d_eff, rcond=None)
    d_hat = U_eff @ beta_d
    R2_d = np.var(d_hat) / np.var(d_eff)

    beta_v, *_ = np.linalg.lstsq(U_eff, v_eff, rcond=None)
    v_hat = U_eff @ beta_v
    R2_v = np.var(v_hat) / np.var(v_eff)

    # informational deficiency delta = 1 - R^2
    ism_store[0, k_idx] = 1.0 - R2_d   # for shock d
    ism_store[1, k_idx] = 1.0 - R2_v   # for shock v

# === REPORTED FIGURES AND METRICS =====

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

outdir = Path(".")
outdir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# TABLE 1 (FEVD, horizons 0/1/4/16)  -> CSV
# -----------------------------
h_cols = [str(h) for h in horizons]

table1 = pd.DataFrame(
    [
        ["y", "median_estimate", *fevd_median_df.loc["y", [f"h={h}" for h in horizons]].to_list()],
        ["y", "true",            *true_fevd_y],
        ["r", "median_estimate", *fevd_median_df.loc["r", [f"h={h}" for h in horizons]].to_list()],
        ["r", "true",            *true_fevd_r],
    ],
    columns=["variable", "stat"] + h_cols,
)
table1.to_csv(outdir / "example1_table1_fevd.csv", index=False)

# -----------------------------
# TABLE 2 (Spectral VD in 3 bands) -> CSV
# -----------------------------
table2 = pd.DataFrame(
    [
        ["y", "median_estimate", *spvd_median_df.loc["y", bands].to_list()],
        ["y", "true",            true_spvd_y_high, true_spvd_y_medium, true_spvd_y_low],
        ["r", "median_estimate", *spvd_median_df.loc["r", bands].to_list()],
        ["r", "true",            true_spvd_r_high, true_spvd_r_medium, true_spvd_r_low],
    ],
    columns=["variable", "stat"] + bands,
)
table2.to_csv(outdir / "example1_table2_spectral_vd.csv", index=False)

# -----------------------------
# TABLE 3 (Deficiency delta(K)) -> CSV
# -----------------------------
table3 = pd.DataFrame(
    ism_store,
    index=["d", "v"],
    columns=[str(K) for K in Ks],
)
table3.index.name = "shock"
table3.to_csv(outdir / "example1_table3_deficiency.csv")

# -----------------------------
# FIGURE 1 (IRFs + 90% bands + correlation distributions) -> PNG + PDF
# Layout: 3 rows x 2 cols
#   Col 1 = demand shock (d), Col 2 = monetary policy shock (v)
#   Row 1 = output (y), Row 2 = interest rate (r), Row 3 = histograms
# -----------------------------
shock_names = ["Demand shock (d)", "Monetary policy shock (v)"]
var_names   = ["Output (y)", "Interest rate (r)"]

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

        true_line = IRF_true[:, var_idx_plot, shock_idx_plot]
        med_line  = IRF_median[:, var_idx_plot, shock_idx_plot]
        lo_line   = IRF_p05[:, var_idx_plot, shock_idx_plot]
        hi_line   = IRF_p95[:, var_idx_plot, shock_idx_plot]

        ax.fill_between(x, lo_line, hi_line, alpha=0.25, linewidth=0)          # gray band
        ax.plot(x, med_line, linestyle="--", linewidth=2.0)                    # black dashed (default color)
        ax.plot(x, true_line, color="red", linestyle="-", linewidth=2.0)       # red solid
        ax.axhline(0.0, linewidth=0.8)

        if col == 0:
            ax.set_ylabel(var_names[var_idx_plot], fontsize=11)

        ax.set_xlabel("Horizon", fontsize=10)
        ax.tick_params(labelsize=9)

# Legend (only once)
axs[0, 1].legend(["90% Confidence Interval", "Median IRF","True IRF"],loc="upper right",fontsize=9,frameon=False)

# Histogram panels (row 2)
# Left: corr(d, identified d); Right: corr(v, identified v)
hist_bins = 30

axs[2, 0].hist(rho_d_all[np.isfinite(rho_d_all)], bins=hist_bins, edgecolor="white")
axs[2, 0].set_title("Corr(estimated shock, true shock): d", fontsize=11)
axs[2, 0].set_xlabel("Correlation", fontsize=10)
axs[2, 0].set_ylabel("Count", fontsize=10)
axs[2, 0].tick_params(labelsize=9)

axs[2, 1].hist(rho_v_all[np.isfinite(rho_v_all)], bins=hist_bins, edgecolor="white")
axs[2, 1].set_title("Corr(estimated shock, true shock): v", fontsize=11)
axs[2, 1].set_xlabel("Correlation", fontsize=10)
axs[2, 1].set_ylabel("Count", fontsize=10)
axs[2, 1].tick_params(labelsize=9)

fig.suptitle("Example 1 — Impulse Responses and Shock Recovery", fontsize=14, y=0.98)

# Save
fig.savefig(outdir / "example1_figure1.png", dpi=300, bbox_inches="tight")
fig.savefig(outdir / "example1_figure1.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(" - example1_table1_fevd.csv")
print(" - example1_table2_spectral_vd.csv")
print(" - example1_table3_deficiency.csv")
print(" - example1_correlations.csv")
print(" - example1_figure1.png")
print(" - example1_figure1.pdf")