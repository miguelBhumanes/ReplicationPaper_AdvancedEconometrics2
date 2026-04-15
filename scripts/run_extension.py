'''
Extension to the replication paper.

Produces:
  results/tables/fred_qd_factors.csv
  results/tables/var_residuals_factors.csv
  results/tables/macro_with_spf_expectations.csv
  results/tables/wedges_consolidated.csv
  results/tables/extension_table.csv
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from replication.io.fredqd import load_fred_qd, DATE_COL
from replication.io.spf import load_spf_inflation, load_spf_rgdp_growth
from replication.io.fred import fred_download
from replication.io.paths import RAW_DATA_DIR, TABLES_DIR
from replication.factors import em_pca, bai_ng_ic
from replication.wedges import compute_wedges
from replication.var import var

# =========================
# 1) Load and transform FRED-QD
# =========================

FRED_QD_PATH = RAW_DATA_DIR / "2025-12-QD.csv"

df_tr, use_in_factors, factor_cols = load_fred_qd(FRED_QD_PATH)

print("Number of series:", len(use_in_factors))
print("Series flagged for factors:", int(use_in_factors.sum()))

# =========================
# 2) Build the T x N matrix for factor extraction
# =========================

Z = df_tr[[DATE_COL] + factor_cols].copy()
Z2 = Z.iloc[2:].reset_index(drop=True)  # drop first 2 rows (NaNs from differencing)

dates = Z2[DATE_COL].copy()
X = Z2[factor_cols].to_numpy(dtype=float)

mu = np.nanmean(X, axis=0)
sd = np.nanstd(X, axis=0, ddof=0)
sd = np.where(sd == 0, 1.0, sd)
Xs = (X - mu) / sd

T, N = Xs.shape

# =========================
# 3) Bai-Ng IC to pick k
# =========================

kmax = 20
rows = []

for k in range(0, kmax + 1):
    Fk, Lk, Xhatk, Ximpk = em_pca(Xs, k=k, max_iter=300, tol=1e-6, verbose=False)
    Vk = np.mean((Ximpk - Xhatk) ** 2)
    ICp1, ICp2, ICp3 = bai_ng_ic(Vk, k, N=N, T=T)
    rows.append({"k": k, "Vk": Vk, "ICp1": ICp1, "ICp2": ICp2, "ICp3": ICp3})

ic_table = pd.DataFrame(rows)

crit = "ICp2"
best_k = int(ic_table.loc[ic_table[crit].idxmin(), "k"])
print(f"Selected k={best_k} factors by {crit}")

# =========================
# 4) Final EM-PCA with best_k
# =========================

F, Lam, Xhat, Ximp = em_pca(Xs, k=best_k, max_iter=500, tol=1e-6, verbose=True)

print("Final factors shape:", F.shape)
print("Final loadings shape:", Lam.shape)
print("Residual variance:", np.mean((Ximp - Xhat) ** 2))

factors_df = pd.DataFrame(F, columns=[f"F{i+1}" for i in range(F.shape[1])])
factors_df.insert(0, DATE_COL, dates.values)

factors_df.to_csv(TABLES_DIR / "fred_qd_factors.csv", index=False)
print(f"Saved factors to {TABLES_DIR / 'fred_qd_factors.csv'}")

# =========================
# 5) VAR on factors: choose p by AIC/BIC, save residuals
# =========================

factor_cols_var = [c for c in factors_df.columns if c.startswith("F")]
Y_var = factors_df[factor_cols_var].to_numpy(dtype=float)

T_var, n_var = Y_var.shape
print("VAR Y shape:", Y_var.shape)

def var_ic_using_aux_var(Y, p, intercept=True):
    U, B = var(Y, p, intercept=intercept)
    U_eff = U[p:, :]
    T_eff = U_eff.shape[0]

    Sigma = (U_eff.T @ U_eff) / T_eff
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        logdet = np.inf

    const = 1 if intercept else 0
    k_params = n_var * (n_var * p + const)

    aic = logdet + (2 * k_params) / T_eff
    bic = logdet + (np.log(T_eff) * k_params) / T_eff
    return aic, bic

pmax = 12
rows_p = []
for p in range(1, pmax + 1):
    aic_p, bic_p = var_ic_using_aux_var(Y_var, p, intercept=True)
    rows_p.append({"p": p, "AIC": aic_p, "BIC": bic_p})

ic_p_table = pd.DataFrame(rows_p).sort_values("p").reset_index(drop=True)
print(ic_p_table)

best_p_aic = int(ic_p_table.loc[ic_p_table["AIC"].idxmin(), "p"])
p_chosen = best_p_aic

U, B = var(Y_var, p_chosen, intercept=True)

U_out = U.astype(float).copy()
U_out[:p_chosen, :] = np.nan

resid_df = pd.DataFrame(U_out, columns=[f"u_{c}" for c in factor_cols_var])
resid_df.insert(0, DATE_COL, factors_df[DATE_COL].values)
resid_df = resid_df.dropna()

resid_df.to_csv(TABLES_DIR / "var_residuals_factors.csv", index=False)
print(f"Saved residuals to {TABLES_DIR / 'var_residuals_factors.csv'}")

# =========================
# 6) Download macro series from FRED and build wedges
# =========================

START = "1960-01-01"
END   = "2025-12-31"

rgdp     = fred_download("GDPC1",   start=START, end=END)
rpc      = fred_download("PCECC96", start=START, end=END)
hoanbs_m = fred_download("HOANBS",  start=START, end=END)
wasc_m   = fred_download("WASCUR",  start=START, end=END)
cpi_m    = fred_download("CPIAUCSL",start=START, end=END)
emp_m    = fred_download("CE16OV",  start=START, end=END)
gs1_m    = fred_download("GS1",     start=START, end=END)

hoanbs = hoanbs_m.resample("QE").mean()
wasc   = wasc_m.resample("QE").mean()
cpi    = cpi_m.resample("QE").mean()
emp    = emp_m.resample("QE").mean()
gs1    = gs1_m.resample("QE").mean()

def force_qend(s):
    s = s.copy()
    s.index = pd.PeriodIndex(pd.DatetimeIndex(s.index), freq="Q").to_timestamp("Q")
    return s

rgdp   = force_qend(rgdp)
rpc    = force_qend(rpc)
hoanbs = force_qend(hoanbs)
wasc   = force_qend(wasc)
cpi    = force_qend(cpi)
emp    = force_qend(emp)
gs1    = force_qend(gs1)

common_idx = (rgdp.index
              .intersection(rpc.index)
              .intersection(hoanbs.index)
              .intersection(wasc.index)
              .intersection(cpi.index)
              .intersection(emp.index)
              .intersection(gs1.index))

rgdp   = rgdp.loc[common_idx]
rpc    = rpc.loc[common_idx]
hoanbs = hoanbs.loc[common_idx]
wasc   = wasc.loc[common_idx]
cpi    = cpi.loc[common_idx]
emp    = emp.loc[common_idx]
gs1    = gs1.loc[common_idx]

rgdp_pw       = rgdp / emp
rpc_pw        = rpc / emp
hours_pw      = hoanbs / emp
real_wages_pw = (wasc / cpi) / emp

# =========================
# 7) Load SPF expectations and merge
# =========================

SPF_INFL_FILE  = RAW_DATA_DIR / "Inflation.xlsx"
SPF_LEVEL_FILE = RAW_DATA_DIR / "Median_RGDP_Growth.xlsx"

exp_infl_1y    = load_spf_inflation(SPF_INFL_FILE)
exp_gdp_grow_1y = load_spf_rgdp_growth(SPF_LEVEL_FILE)

spf_df = pd.concat([exp_gdp_grow_1y, exp_infl_1y], axis=1).sort_index()

macro_df = pd.DataFrame({
    "rgdp_pw":       rgdp_pw,
    "rpc_pw":        rpc_pw,
    "hours_pw":      hours_pw,
    "real_wages_pw": real_wages_pw,
    "cpi_index_q":   cpi,
    "gs1":           gs1
}).sort_index()

macro_df.index = pd.PeriodIndex(macro_df.index, freq="Q").to_timestamp("Q")
spf_df.index   = pd.PeriodIndex(spf_df.index,   freq="Q").to_timestamp("Q")

common_idx = macro_df.index.intersection(spf_df.index)
df_all = pd.concat([macro_df.loc[common_idx], spf_df.loc[common_idx]], axis=1)

print(df_all.shape, df_all.index.min(), df_all.index.max())
df_all.to_csv(TABLES_DIR / "macro_with_spf_expectations.csv")

df_all = df_all.dropna()

# =========================
# 8) Compute wedges
# =========================

wedges_df = compute_wedges(df_all)
wedges_df.to_csv(TABLES_DIR / "wedges_consolidated.csv")
print(f"Saved {TABLES_DIR / 'wedges_consolidated.csv'}")

# =========================
# 9) Informational deficiency table (linear vs non-linear projections)
# =========================

wedge_stat_cols = ["wedge1_stat", "wedge2_stat", "wedge3_stat", "wedge4_stat", "wedge5_stat"]
model_names = {
    "wedge1_stat": "Euler equation",
    "wedge2_stat": "Keynesian consumption",
    "wedge3_stat": "Permanent income",
    "wedge4_stat": "Forward-looking Taylor rule",
    "wedge5_stat": "Intratemporal substitution",
}

u_cols = [c for c in resid_df.columns if c.startswith("u_")]

wstat = wedges_df[wedge_stat_cols].copy()
wstat = wstat.reset_index().rename(columns={"index": DATE_COL})
wstat[DATE_COL] = pd.PeriodIndex(pd.to_datetime(wstat[DATE_COL]), freq="Q").to_timestamp("Q")

resid_tmp = resid_df[[DATE_COL] + u_cols].copy()
resid_tmp[DATE_COL] = pd.PeriodIndex(pd.to_datetime(resid_tmp[DATE_COL]), freq="Q").to_timestamp("Q")

proj_df = pd.merge(wstat, resid_tmp, on=DATE_COL, how="inner").dropna()

rows_out = []
for wcol in wedge_stat_cols:
    y = proj_df[wcol].to_numpy(dtype=float)
    X_proj = proj_df[u_cols].to_numpy(dtype=float)

    X_lin = sm.add_constant(X_proj)
    m_lin = sm.OLS(y, X_lin).fit()
    info_def_lin = 1.0 - float(m_lin.rsquared)

    scaler = StandardScaler()
    Xs_nn = scaler.fit_transform(X_proj)
    mlp = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=5000,
        random_state=0,
    )
    mlp.fit(Xs_nn, y)
    yhat = mlp.predict(Xs_nn)
    sse = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    R2_nl = 1.0 - sse / tss
    info_def_nl = 1.0 - R2_nl

    rows_out.append(
        {
            "model": model_names[wcol],
            "linear_projection": info_def_lin,
            "nonlinear_projection": info_def_nl,
        }
    )

info_table = pd.DataFrame(rows_out).set_index("model")
print(info_table.round({"linear_projection": 4, "nonlinear_projection": 2}))

info_table.to_csv(TABLES_DIR / "extension_table.csv")
print(f"Saved {TABLES_DIR / 'extension_table.csv'}")
