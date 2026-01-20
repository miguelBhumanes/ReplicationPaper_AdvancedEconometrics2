'''
Extension to the replication paper
'''

# =========================
# 0) Imports + file path
# =========================
import numpy as np
import pandas as pd
from aux_VAR import var
import statsmodels.api as sm

PATH = "2025-12-QD.csv"

DATE_COL = "sasdate"

# =========================
# 1) Read the 3 metadata rows
#    Row 0: column names
#    Row 1: factors flags (0/1)
#    Row 2: transformation codes (1..7)
# =========================
meta = pd.read_csv(PATH, header=None, nrows=3)
meta

header = meta.iloc[0].tolist()
factors_row = meta.iloc[1].tolist()
transform_row = meta.iloc[2].tolist()

# Build series lists (exclude date column)
series = header[1:]

use_in_factors = pd.Series(factors_row[1:], index=series).astype(float).astype(int)
transform_code = pd.Series(transform_row[1:], index=series).astype(float).astype(int)

print("Number of series:", len(series))
print("Series flagged for factors:", int(use_in_factors.sum()))
transform_code.value_counts().sort_index()

# =========================
# 2) Read the actual data block
# =========================
df_raw = pd.read_csv(PATH, skiprows=3, header=None)
df_raw.columns = header

# Parse and sort dates
df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], errors="coerce")
df_raw = df_raw.sort_values(DATE_COL).reset_index(drop=True)

# =========================
# 3) Apply McCracken/Ng transform codes
#    1: x
#    2: Δx
#    3: Δ²x
#    4: log(x)
#    5: Δlog(x)
#    6: Δ²log(x)
#    7: Δ( x/x(-1) - 1 )
# =========================
def transform_series(x: pd.Series, code: int) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)

    if code == 1:
        return x
    elif code == 2:
        return x.diff(1)
    elif code == 3:
        return x.diff(1).diff(1)
    elif code == 4:
        return np.log(x.where(x > 0))
    elif code == 5:
        return np.log(x.where(x > 0)).diff(1)
    elif code == 6:
        return np.log(x.where(x > 0)).diff(1).diff(1)
    elif code == 7:
        g = (x / x.shift(1)) - 1.0
        return g.diff(1)
    else:
        # unknown code: do nothing
        return x

out = {DATE_COL: df_raw[DATE_COL]}

for col in series:
    code = int(transform_code[col])
    out[col] = transform_series(df_raw[col], code)

df_tr = pd.DataFrame(out).copy()   # .copy() optional but nice

df_tr.head()

# =========================
# 4) Keep only the series flagged for factor extraction
# =========================
factor_cols = use_in_factors[use_in_factors == 1].index.tolist()

Z = df_tr[[DATE_COL] + factor_cols].copy()

# Drop the first 2 columns because they are using differeces and have nans
Z2 = Z.iloc[2:].reset_index(drop=True)

# =========================
# 5) Make the T x N matrix and standardize (mean 0, sd 1)
# =========================
dates = Z2[DATE_COL].copy()
X = Z2[factor_cols].to_numpy(dtype=float)

# standardize with NaNs allowed
mu = np.nanmean(X, axis=0)
sd = np.nanstd(X, axis=0, ddof=0)
sd = np.where(sd == 0, 1.0, sd)
Xs = (X - mu) / sd

T, N = Xs.shape

# =========================
# 6) EM-PCA to impute missing values + extract factors
# =========================
def em_pca(Xs: np.ndarray, k: int, max_iter=300, tol=1e-6, seed=0, verbose=False):
    """
    Xs: standardized (T,N) with NaNs
    k: number of PCs/factors
    Returns: F (T,k), Lambda (N,k), Xhat (T,N), Ximp (T,N)
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


# =========================
# 7) Bai–Ng information criteria to pick k
# =========================
def bai_ng_ic(Vk: float, k: int, N: int, T: int):
    C2 = min(N, T)
    Vk = max(Vk, 1e-12)

    pen1 = ((N + T) / (N * T)) * np.log((N * T) / (N + T))
    pen2 = ((N + T) / (N * T)) * np.log(C2)
    pen3 = (np.log(C2) / C2)

    ICp1 = np.log(Vk) + k * pen1
    ICp2 = np.log(Vk) + k * pen2
    ICp3 = np.log(Vk) + k * pen3
    return ICp1, ICp2, ICp3

kmax = 20
rows = []

for k in range(0, kmax + 1):
    Fk, Lk, Xhatk, Ximpk = em_pca(Xs, k=k, max_iter=300, tol=1e-6, verbose=False)
    Vk = np.mean((Ximpk - Xhatk) ** 2)

    ICp1, ICp2, ICp3 = bai_ng_ic(Vk, k, N=N, T=T)
    rows.append({"k": k, "Vk": Vk, "ICp1": ICp1, "ICp2": ICp2, "ICp3": ICp3})

ic_table = pd.DataFrame(rows)
ic_table.head()

crit = "ICp2"
best_k = int(ic_table.loc[ic_table[crit].idxmin(), "k"])
best_k

# Inspect the top candidates
ic_table.sort_values(crit).head(10)

# =========================
# 8) Re-run EM-PCA with best_k to get final factors
# =========================
F, Lam, Xhat, Ximp = em_pca(Xs, k=best_k, max_iter=500, tol=1e-6, verbose=True)

print("Final factors shape:", F.shape)
print("Final loadings shape:", Lam.shape)
print("Residual variance:", np.mean((Ximp - Xhat) ** 2))

# Put factors into a DataFrame with dates
factors_df = pd.DataFrame(F, columns=[f"F{i+1}" for i in range(F.shape[1])])
factors_df.insert(0, DATE_COL, dates.values)

factors_df.head()

# Save for VAR use
factors_df.to_csv("fred_qd_factors.csv", index=False)
print("Saved factors to fred_qd_factors.csv")

# =========================
# 9) VAR on factors: choose p by AIC/BIC (use BIC if they disagree) and save residuals
# =========================

# Build VAR input matrix (T x n_factors)
factor_cols_var = [c for c in factors_df.columns if c.startswith("F")]
Y = factors_df[factor_cols_var].to_numpy(dtype=float)

T_var, n_var = Y.shape
print("VAR Y shape:", Y.shape)

def var_ic_using_aux_var(Y, p, intercept=True):
    """
    Uses aux_VAR.var(Y,p) output to compute AIC/BIC.
    """
    U, B = var(Y, p, intercept=intercept)   # <- your imported aux_VAR.var
    U_eff = U[p:, :]
    T_eff = U_eff.shape[0]

    Sigma = (U_eff.T @ U_eff) / T_eff
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        logdet = np.inf

    const = 1 if intercept else 0
    k_params = n_var * (n_var * p + const)  # total params in the system

    aic = logdet + (2 * k_params) / T_eff
    bic = logdet + (np.log(T_eff) * k_params) / T_eff
    return aic, bic

pmax = 12  # quarterly default; adjust if needed
rows_p = []
for p in range(1, pmax + 1):
    aic_p, bic_p = var_ic_using_aux_var(Y, p, intercept=True)
    rows_p.append({"p": p, "AIC": aic_p, "BIC": bic_p})

ic_p_table = pd.DataFrame(rows_p).sort_values("p").reset_index(drop=True)
print(ic_p_table)

best_p_aic = int(ic_p_table.loc[ic_p_table["AIC"].idxmin(), "p"])
best_p_bic = int(ic_p_table.loc[ic_p_table["BIC"].idxmin(), "p"])
p_chosen = best_p_aic

# Fit chosen VAR and extract reduced-form residual series
U, B = var(Y, p_chosen, intercept=True)

# Make padding explicit as missing (safer than zeros)
U_out = U.astype(float).copy()
U_out[:p_chosen, :] = np.nan

resid_df = pd.DataFrame(U_out, columns=[f"u_{c}" for c in factor_cols_var])
resid_df.insert(0, DATE_COL, factors_df[DATE_COL].values)

# Drop the first lags observations (Nans because no residuals were obtained for those rows)
resid_df = resid_df.dropna()

# Saving dataset
resid_df.to_csv("var_residuals_factors.csv", index=False)
print("Saved residuals to var_residuals_factors.csv")

# =========================
# 10) Writing different models in terms of wedges and computing those wedges
# =========================

START = "1960-01-01"
END   = "2025-12-31"

# =========================
# 11) FRED downloader (no API key)
#    Uses fredgraph CSV endpoint
# =========================
def fred_download(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df.set_index("date")[series_id].sort_index()
    return s.loc[START:END]

# =========================
# 12) Pull FRED series  (worker-based denominators)
# =========================
rgdp  = fred_download("GDPC1")       # quarterly real GDP (SAAR)
rpc = fred_download("PCECC96")       # quarterlt real consumption
hoanbs_m = fred_download("HOANBS")   # monthly hours index
wasc_m   = fred_download("WASCUR")   # monthly nominal wages & salary accruals (SAAR)
cpi_m  = fred_download("CPIAUCSL")   # monthly CPI index
emp_m  = fred_download("CE16OV")     # monthly employment level (persons)
gs1_m  = fred_download("GS1")        # monthly 1-year Treasury constant maturity rate (%)

# =========================
# 13) Convert monthly CPI & employment to quarterly (average within quarter)
# =========================
hoanbs = hoanbs_m.resample("QE").mean()
wasc = wasc_m.resample("QE").mean()
cpi = cpi_m.resample("QE").mean()
emp = emp_m.resample("QE").mean()
gs1 = gs1_m.resample("QE").mean()

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

common_idx = rgdp.index.intersection(rpc.index).intersection(hoanbs.index).intersection(wasc.index) \
    .intersection(cpi.index).intersection(emp.index).intersection(gs1.index)

rgdp   = rgdp.loc[common_idx]
rpc    = rpc.loc[common_idx]
hoanbs = hoanbs.loc[common_idx]
wasc   = wasc.loc[common_idx]
cpi    = cpi.loc[common_idx]
emp    = emp.loc[common_idx]
gs1    = gs1.loc[common_idx]


# =========================
# 14) Construct per-worker
# =========================
# Real GDP per worker (SAAR dollars per employed person)
rgdp_pw = rgdp / emp

# Real consumption per worker (SAAR dollars per employed person)
rpc_pw = rpc / emp

# Hours worked per worker proxy: hours index divided by employment level
hours_pw = hoanbs / emp

# Real wages per worker proxy:
# (nominal wages SAAR) / CPI / employment
# Note: CPI is an index -> this is a consistent deflated series up to scale.
real_wages_pw = (wasc / cpi) / emp

# Keep CPI index itself too (quarterly average)
cpi_index_q = cpi.copy()


# =========================
# 15) SPF expectations (local XLSX) + merge with macro on common dates
# =========================

SPF_INFL_FILE  = "Inflation.xlsx"
SPF_LEVEL_FILE = "Median_RGDP_Growth.xlsx"

spf_inf = pd.read_excel(SPF_INFL_FILE)
spf_inf.columns = [str(c).strip().upper() for c in spf_inf.columns]
spf_inf["DATE"] = pd.PeriodIndex.from_fields(
    year=spf_inf["YEAR"].astype(int),
    quarter=spf_inf["QUARTER"].astype(int),
    freq="Q"
).to_timestamp("Q")
exp_infl_1y = spf_inf.set_index("DATE")["INFCPI1YR"].rename("exp_infl_1y")

spf_rgdp = pd.read_excel(SPF_LEVEL_FILE, sheet_name="Median_Growth")
spf_rgdp.columns = [str(c).strip().upper() for c in spf_rgdp.columns]
spf_rgdp["DATE"] = pd.PeriodIndex.from_fields(
    year=spf_rgdp["YEAR"].astype(int),
    quarter=spf_rgdp["QUARTER"].astype(int),
    freq="Q"
).to_timestamp("Q")

g3 = spf_rgdp["DRGDP3"] / 400.0
g4 = spf_rgdp["DRGDP4"] / 400.0
g5 = spf_rgdp["DRGDP5"] / 400.0
g6 = spf_rgdp["DRGDP6"] / 400.0

exp_gdp_grow_1y = 100.0 * ((1 + g3) * (1 + g4) * (1 + g5) * (1 + g6) - 1.0)
exp_gdp_grow_1y.index = spf_rgdp["DATE"]
exp_gdp_grow_1y = exp_gdp_grow_1y.rename("exp_gdp_grow_1y")

spf_df = pd.concat([exp_gdp_grow_1y, exp_infl_1y], axis=1).sort_index()

macro_df = pd.DataFrame({
    "rgdp_pw": rgdp_pw,
    "rpc_pw": rpc_pw,
    "hours_pw": hours_pw,
    "real_wages_pw": real_wages_pw,
    "cpi_index_q": cpi_index_q,
    "gs1": gs1
}).sort_index()

# align indices to quarter-end timestamps and merge on intersection
macro_df.index = pd.PeriodIndex(macro_df.index, freq="Q").to_timestamp("Q")
spf_df.index   = pd.PeriodIndex(spf_df.index,   freq="Q").to_timestamp("Q")

common_idx = macro_df.index.intersection(spf_df.index)
df_all = pd.concat([macro_df.loc[common_idx], spf_df.loc[common_idx]], axis=1)

print(df_all.shape, df_all.index.min(), df_all.index.max())
df_all.to_csv("macro_with_spf_expectations.csv")

# Dropping NAs for the computations
df_all = df_all.dropna()

# =========================
# 16) OBTAINING WEDGES
# =========================

'''
Model: Euler Equation
E Delta log C = 1/sigma (i - Epi - rho) -> Wedge1 = i - Epi - rho - E Delta log C
We proxy E Delta log C with the SPF expected output growth we have (E Delta log Y)
rho = 4% to be consistent with paper calibration of quarter beta = 0.99 (0.96 annual)
sigma = 1 calibrated in the paper
Wedge already for logs, so we just take differences of this wedge to make it stationary
Use per worker data for consumption
'''

# parameters per docstring
rho = 0.04
sigma = 1.0

# expected consumption log growth proxy: expected output growth (as provided)
E_dlogC = df_all["exp_gdp_grow_1y"]

wedge1 = df_all["gs1"] - df_all["exp_infl_1y"] - rho - sigma * E_dlogC
wedge1_stat = wedge1.diff()

'''
Model: Keynesian Consumption Function
C = c0 + c1Y where c0,c1 come from OLS on the sample
-> Wedge2 = c0 + c1Y - C
This wedge we take growth rates to make it stationary
Use per worker data for consumption and output
'''
C = df_all["rpc_pw"]
Y = df_all["rgdp_pw"]

X2 = sm.add_constant(Y)
m2 = sm.OLS(C, X2).fit()
C_hat2 = m2.predict(X2)

wedge2 = C_hat2 - C
wedge2_stat = wedge2.pct_change()

'''
Model: Permanent Income Hypothesis
Wedge3 = c0 (i - Epi) - C where c0 comes from OLS
This wedge we take growth rates
Use per worker data for consumption and output
'''
# real interest rate 
r_real = df_all["gs1"] - df_all["exp_infl_1y"]

# OLS to get c0: C = c0 * (i - Epi)   (no intercept so c0 is the slope)
m3 = sm.OLS(C, r_real).fit()
c0_pih = float(m3.params.iloc[0])

wedge3 = c0_pih * r_real - C
wedge3_stat = wedge3.pct_change()

'''
Model: Forward looking Taylor Rule
Wedge4 = i - rho - i0 - i1 Eygrowth - i2 Einflation 
This wedge just take differences to make it stationary
rho = 4% as before
i0,i1,i2 fit with OLS in the sample
'''
y4 = df_all["gs1"] - rho
X4 = sm.add_constant(
    pd.DataFrame(
        {
            "Eygrowth": df_all["exp_gdp_grow_1y"],
            "Einflation": df_all["exp_infl_1y"],
        },
        index=df_all.index,
    )
)
m4 = sm.OLS(y4, X4).fit()
y4_hat = m4.predict(X4)

wedge4 = y4 - y4_hat
wedge4_stat = wedge4.diff()

'''
Model: Intratemporal Substitution Optimality
sigma log C + frisch log N  = log W/P -> Wedge 5 = log W/P - sigma log C - frisch log N
to match paper: sigma = 1 frisch = 3.98
Just take differences of this wedge to make it stationary
Use per worker data for consumption and N (hours) and wages
'''
frisch = 3.98
sigma = 1.0

log_wp = np.log(df_all["real_wages_pw"])
log_c  = np.log(df_all["rpc_pw"])
log_n  = np.log(df_all["hours_pw"])

wedge5 = log_wp - sigma * log_c - frisch * log_n
wedge5_stat = wedge5.diff()

# Consolidated time-indexed dataframe with all wedges
wedges_df = pd.DataFrame(
    {
        "wedge1": wedge1,
        "wedge1_stat": wedge1_stat,
        "wedge2": wedge2,
        "wedge2_stat": wedge2_stat,
        "wedge3": wedge3,
        "wedge3_stat": wedge3_stat,
        "wedge4": wedge4,
        "wedge4_stat": wedge4_stat,
        "wedge5": wedge5,
        "wedge5_stat": wedge5_stat,
    },
    index=df_all.index,
).sort_index()

# save
wedges_df.to_csv("wedges_consolidated.csv")

# =========================
# 17) Informational deficiency table (linear vs non-linear projections)
# =========================
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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
    X = proj_df[u_cols].to_numpy(dtype=float)

    m_lin = sm.OLS(y, X).fit()
    info_def_lin = 1.0 - float(m_lin.rsquared)

    scaler = StandardScaler()
    Xs_nn = scaler.fit_transform(X)
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
    mse = float(np.mean((y - yhat) ** 2))
    mse_mean = float(np.mean((y - y.mean()) ** 2)) + 1e-12
    score_nl = 100.0 * (1.0 - mse / mse_mean)

    rows_out.append(
        {
            "model": model_names[wcol],
            "linear_projection": info_def_lin,
            "nonlinear_projection": score_nl,
        }
    )

info_table = pd.DataFrame(rows_out).set_index("model")
print(info_table.round({"linear_projection": 4, "nonlinear_projection": 2}))

info_table.to_csv("extension_table.csv")
print("Saved extension_table.csv")


