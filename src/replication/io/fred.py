"""
Lightweight FRED data downloader (no API key required).
Uses the public fredgraph CSV endpoint.
"""

import pandas as pd


def fred_download(series_id: str, start: str = "1960-01-01", end: str = "2025-12-31") -> pd.Series:
    """
    Download a FRED series as a pandas Series indexed by date.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. "GDPC1").
    start : str
        Start date in ISO format (inclusive).
    end : str
        End date in ISO format (inclusive).

    Returns
    -------
    pd.Series
        Date-indexed series, numeric, sorted ascending, trimmed to [start, end].
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df.set_index("date")[series_id].sort_index()
    return s.loc[start:end]
