"""
Loaders for Philadelphia Fed Survey of Professional Forecasters (SPF) data.
"""

import pandas as pd


def load_spf_inflation(path) -> pd.Series:
    """
    Load 1-year-ahead median CPI inflation expectations from the SPF XLSX.

    Expected columns (case-insensitive): YEAR, QUARTER, INFCPI1YR

    Parameters
    ----------
    path : str or Path
        Path to the Inflation.xlsx file.

    Returns
    -------
    pd.Series
        Quarter-end-indexed series named 'exp_infl_1y'.
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["DATE"] = pd.PeriodIndex.from_fields(
        year=df["YEAR"].astype(int),
        quarter=df["QUARTER"].astype(int),
        freq="Q"
    ).to_timestamp("Q")
    return df.set_index("DATE")["INFCPI1YR"].rename("exp_infl_1y")


def load_spf_rgdp_growth(path) -> pd.Series:
    """
    Load 1-year-ahead median real GDP growth expectations from the SPF XLSX.

    Computes the annualised growth rate as:
        100 * [(1+g3)(1+g4)(1+g5)(1+g6) - 1]
    where g3–g6 are the quarterly growth rates (DRGDP3–DRGDP6) divided by 400.

    Expected sheet name: 'Median_Growth'
    Expected columns (case-insensitive): YEAR, QUARTER, DRGDP3, DRGDP4, DRGDP5, DRGDP6

    Parameters
    ----------
    path : str or Path
        Path to the Median_RGDP_Growth.xlsx file.

    Returns
    -------
    pd.Series
        Quarter-end-indexed series named 'exp_gdp_grow_1y'.
    """
    df = pd.read_excel(path, sheet_name="Median_Growth")
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["DATE"] = pd.PeriodIndex.from_fields(
        year=df["YEAR"].astype(int),
        quarter=df["QUARTER"].astype(int),
        freq="Q"
    ).to_timestamp("Q")

    g3 = df["DRGDP3"] / 400.0
    g4 = df["DRGDP4"] / 400.0
    g5 = df["DRGDP5"] / 400.0
    g6 = df["DRGDP6"] / 400.0

    exp_grow = 100.0 * ((1 + g3) * (1 + g4) * (1 + g5) * (1 + g6) - 1.0)
    exp_grow.index = df["DATE"]
    return exp_grow.rename("exp_gdp_grow_1y")
