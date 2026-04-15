"""
Macroeconomic wedge construction for the extension analysis.

Five wedges derived from standard macro models (Euler equation,
Keynesian consumption function, Permanent Income Hypothesis,
forward-looking Taylor rule, intratemporal substitution optimality).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_wedges(df_all: pd.DataFrame, rho: float = 0.04, sigma: float = 1.0, frisch: float = 3.98) -> pd.DataFrame:
    """
    Compute the five macroeconomic wedges and their stationary transformations.

    Parameters
    ----------
    df_all : pd.DataFrame
        Merged macro + SPF expectations DataFrame. Must contain columns:
        'gs1', 'exp_infl_1y', 'exp_gdp_grow_1y', 'rpc_pw', 'rgdp_pw',
        'hours_pw', 'real_wages_pw'.
    rho : float
        Discount rate (annualised). Default 0.04 (matches beta=0.99 quarterly).
    sigma : float
        Inverse intertemporal elasticity of substitution. Default 1.
    frisch : float
        Frisch elasticity of labour supply. Default 3.98 (paper calibration).

    Returns
    -------
    pd.DataFrame
        Columns: wedge1..wedge5 (levels) and wedge1_stat..wedge5_stat
        (stationary transformations), indexed by date.
    """

    # ---- Wedge 1: Euler Equation ----
    # Wedge1 = i - Epi - rho - sigma * E Delta log C
    # Proxy E Delta log C ≈ expected output growth
    E_dlogC = df_all["exp_gdp_grow_1y"]
    wedge1 = df_all["gs1"] - df_all["exp_infl_1y"] - rho - sigma * E_dlogC
    wedge1_stat = wedge1.diff()

    # ---- Wedge 2: Keynesian Consumption Function ----
    # C = c0 + c1 Y  (OLS); Wedge2 = C_hat - C; stationary via pct_change
    C = df_all["rpc_pw"]
    Y = df_all["rgdp_pw"]
    X2 = sm.add_constant(Y)
    m2 = sm.OLS(C, X2).fit()
    C_hat2 = m2.predict(X2)
    wedge2 = C_hat2 - C
    wedge2_stat = wedge2.pct_change()

    # ---- Wedge 3: Permanent Income Hypothesis ----
    # C = c0 * (i - Epi)  (OLS, no intercept); stationary via pct_change
    r_real = df_all["gs1"] - df_all["exp_infl_1y"]
    m3 = sm.OLS(C, r_real).fit()
    c0_pih = float(m3.params.iloc[0])
    wedge3 = c0_pih * r_real - C
    wedge3_stat = wedge3.pct_change()

    # ---- Wedge 4: Forward-looking Taylor Rule ----
    # i - rho = i0 + i1 E_y + i2 E_pi  (OLS); stationary via diff
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

    # ---- Wedge 5: Intratemporal Substitution Optimality ----
    # sigma log C + frisch log N = log W/P; stationary via diff
    log_wp = np.log(df_all["real_wages_pw"])
    log_c  = np.log(df_all["rpc_pw"])
    log_n  = np.log(df_all["hours_pw"])
    wedge5 = log_wp - sigma * log_c - frisch * log_n
    wedge5_stat = wedge5.diff()

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

    return wedges_df
