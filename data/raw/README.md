# Raw Data

These files are **not tracked** in git. Place them here before running the pipeline.

| File | Description | Source |
|---|---|---|
| `2025-12-QD.csv` | McCracken & Ng FRED-QD quarterly macro dataset. Three header rows: column names, factor-inclusion flags (0/1), McCracken–Ng transform codes (1–7). First data column is `sasdate`. | [FRED-QD](https://research.stlouisfed.org/econ/mccracken/fred-databases/) — download the "Quarterly" version |
| `fgs-data.txt` | Forni, Gambetti, Sala replication dataset. Whitespace-separated, date-indexed, columns: `TFP Y C I H pi FFR W`. | Provided by Forni, Gambetti & Sala (replication files for their paper) |
| `Inflation.xlsx` | Philadelphia Fed Survey of Professional Forecasters — median 1-year CPI inflation expectations. Columns: `YEAR QUARTER INFCPI1YR`. | [Philadelphia Fed SPF](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters) |
| `Median_RGDP_Growth.xlsx` | Philadelphia Fed SPF — median real GDP growth expectations. Sheet: `Median_Growth`. Columns: `YEAR QUARTER DRGDP3 DRGDP4 DRGDP5 DRGDP6`. | [Philadelphia Fed SPF](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters) |

> **FRED macro data** (`GDPC1`, `PCECC96`, `HOANBS`, `WASCUR`, `CPIAUCSL`, `CE16OV`, `GS1`) used in `scripts/run_extension.py` are pulled live over HTTP — no local file needed.
>
> **FRED DSGE calibration data** (`GDPC1`, `PCECC96`, `GPDIC1`) used in `scripts/run_dsge_solution.py` require a `FRED_API_KEY` environment variable (see `.env.example`).
