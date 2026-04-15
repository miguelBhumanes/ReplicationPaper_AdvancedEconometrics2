'''
Sections 4.3 and 5
'''

import numpy as np
import pandas as pd

from replication.bvar import (
    build_table7_dataframe,
    bvar_s10_irfs,
    empirical_figure7_objects,
    dsge_figure7_objects,
    plot_figure7,
)
from replication.io.fgs import load_and_transform_fgs
from replication.io.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR

# =========================
# Table 7 replication
# =========================

data = np.load(PROCESSED_DATA_DIR / "dsge_solution.npz")
F = data["F"]
G = data["G"]

specs_var_indices = {
    "S1":  [2,1],
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

shock_names = [
    "Mon. pol.",
    "Gov't exp.",
    "Wage markup",
    "Temp. tech.",
    "News",
    "Price markup",
    "Inv. spec."
]

shock_index_map = {
    "News":         4,
    "Temp. tech.":  3,
    "Price markup": 5,
    "Wage markup":  2,
    "Gov't exp.":   1,
    "Inv. spec.":   6,
    "Mon. pol.":    0,
}

T = 10**4
K = 10
burn = 1000
seed = 100
intercept = False
Sigma_u = np.eye(7)

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

table7_df.to_csv(TABLES_DIR / "table7_replication.csv")
print(f"Saved: {TABLES_DIR / 'table7_replication.csv'}")

# =========================
# Figure 7 replication
# =========================

P_LAGS = 4
H_IRF = 30
H_RESTR = 20
N_DRAWS = 500
SEED = 100

NEWS_SHOCK_IDX_DSGE = 4
NEWS_SHOCK_SIZE = 1.0

NEWS_SHOCK_IDX_EMP = 1

USE_INTERCEPT = True

bvar_df = load_and_transform_fgs(RAW_DATA_DIR / "fgs-data.txt")

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

plot_figure7(emp_stats, th, h, H=H_IRF, save_path=FIGURES_DIR / "figure7_replication.png")

print(f"Saved: {FIGURES_DIR / 'figure7_replication.png'}")
print(f"Saved: {FIGURES_DIR / 'figure7_replication.pdf'}")
