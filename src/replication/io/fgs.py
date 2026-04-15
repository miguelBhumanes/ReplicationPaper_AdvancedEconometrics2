"""
Loader for the Forni, Gambetti, Sala (FGS) replication dataset.
"""

import numpy as np
import pandas as pd


def load_and_transform_fgs(path: str) -> pd.DataFrame:
    """
    Load the FGS data file and transform variables to stationarity.

    The raw file is whitespace-separated, date-indexed, with columns:
        TFP, Y, C, I, H, pi, FFR, W

    Transformations applied:
        dlog_TFP_100 = 100 * Δ log(TFP)
        dlog_Y_100   = 100 * Δ log(Y)
        dlog_C_100   = 100 * Δ log(C)
        dlog_I_100   = 100 * Δ log(I)
        H_100        = H  (levels, already stationary)
        pi_100       = pi
        ffr          = FFR
        dlog_W_100   = W  (already in log-difference form in the raw file)

    Parameters
    ----------
    path : str or Path
        Path to the FGS data text file.

    Returns
    -------
    pd.DataFrame
        Transformed, NaN-dropped DataFrame with the 8 columns listed above.
    """
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
