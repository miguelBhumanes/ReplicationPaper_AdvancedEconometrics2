"""
Loader for the McCracken & Ng FRED-QD dataset.

The CSV has three header rows followed by data:
  Row 0: column names (first column is 'sasdate')
  Row 1: factor-inclusion flags (0 or 1 per series)
  Row 2: McCracken-Ng transformation codes (1–7)

Transformation codes:
  1: level
  2: first difference
  3: second difference
  4: log
  5: first difference of log
  6: second difference of log
  7: first difference of (x/x(-1) - 1)
"""

import numpy as np
import pandas as pd

DATE_COL = "sasdate"


def transform_series(x: pd.Series, code: int) -> pd.Series:
    """Apply a McCracken-Ng transformation code to a series."""
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
        return x


def load_fred_qd(path) -> tuple:
    """
    Load and transform the FRED-QD CSV.

    Parameters
    ----------
    path : str or Path
        Path to the FRED-QD CSV file.

    Returns
    -------
    df_tr : pd.DataFrame
        Full transformed DataFrame (all series + date column).
    use_in_factors : pd.Series
        Integer series (0/1) indicating which columns to use for factor extraction.
    factor_cols : list of str
        Column names flagged for factor extraction (use_in_factors == 1).
    """
    meta = pd.read_csv(path, header=None, nrows=3)

    header = meta.iloc[0].tolist()
    factors_row = meta.iloc[1].tolist()
    transform_row = meta.iloc[2].tolist()

    series = header[1:]  # exclude date column

    use_in_factors = pd.Series(factors_row[1:], index=series).astype(float).astype(int)
    transform_code = pd.Series(transform_row[1:], index=series).astype(float).astype(int)

    df_raw = pd.read_csv(path, skiprows=3, header=None)
    df_raw.columns = header

    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], errors="coerce")
    df_raw = df_raw.sort_values(DATE_COL).reset_index(drop=True)

    out = {DATE_COL: df_raw[DATE_COL]}
    for col in series:
        code = int(transform_code[col])
        out[col] = transform_series(df_raw[col], code)

    df_tr = pd.DataFrame(out).copy()

    factor_cols = use_in_factors[use_in_factors == 1].index.tolist()

    return df_tr, use_in_factors, factor_cols
