# Replication Paper

Code to replicate and extend the main empirical and DSGE results of Forni, Gambetti & Sala.

## Repository layout

```
.
├── src/replication/         # installable Python package
│   ├── var.py               # VAR estimation, Cholesky, IRFs, FEVD, spectral VD
│   ├── bvar.py              # BVAR (diffuse prior), Table 7 utilities, Figure 7 helpers
│   ├── factors.py           # EM-PCA, Bai-Ng information criteria
│   ├── wedges.py            # macroeconomic wedge construction
│   ├── dsge/
│   │   ├── lre.py           # generic LRE model utilities (Sims/GENSYS)
│   │   └── solver.py        # QZ + Newton solver for x_t = F x_{t-1} + G u_t
│   └── io/
│       ├── paths.py         # central path registry (RAW_DATA_DIR, TABLES_DIR, …)
│       ├── fgs.py           # Forni-Gambetti-Sala data loader
│       ├── fredqd.py        # McCracken-Ng FRED-QD loader + transforms
│       ├── spf.py           # Philadelphia Fed SPF loaders
│       └── fred.py          # lightweight FRED HTTP downloader
│
├── scripts/                 # entry-point scripts (thin; all logic lives in src/)
│   ├── run_section_41.py    # Section 4.1 — Example 1 Monte Carlo
│   ├── run_section_42.py    # Section 4.2 — Example 2 Monte Carlo
│   ├── run_dsge_solution.py # DSGE model solution → data/processed/dsge_solution.npz
│   ├── run_section_43_and_5.py  # Section 4.3 + 5 — Table 7 & Figure 7
│   └── run_extension.py     # Extension — factor model, wedges, info deficiency
│
├── data/
│   ├── raw/                 # gitignored; place raw inputs here (see data/raw/README.md)
│   └── processed/           # gitignored; intermediate artifacts (e.g. dsge_solution.npz)
│
├── results/
│   ├── figures/             # gitignored; output figures (PNG + PDF)
│   └── tables/              # gitignored; output tables (CSV)
│
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                  # registers `replication` as importable package
```

## Data

Raw data files are **not tracked** in this repository. See `data/raw/README.md`
for a full description of each file and where to download it. In brief:

| File | Description |
|---|---|
| `data/raw/2025-12-QD.csv` | McCracken & Ng FRED-QD quarterly macro dataset |
| `data/raw/fgs-data.txt` | Forni, Gambetti & Sala replication dataset |
| `data/raw/Inflation.xlsx` | Philadelphia Fed SPF 1-year CPI inflation expectations |
| `data/raw/Median_RGDP_Growth.xlsx` | Philadelphia Fed SPF 1-year real GDP growth expectations |

The extension script also pulls FRED series (`GDPC1`, `PCECC96`, `HOANBS`,
`WASCUR`, `CPIAUCSL`, `CE16OV`, `GS1`) live over HTTP — no local file needed.

The DSGE solver fetches `GDPC1`, `PCECC96`, `GPDIC1` via the FRED API and
requires a key:

```bash
cp .env.example .env
# edit .env and fill in FRED_API_KEY
export FRED_API_KEY=your_key_here
```

## Running the pipeline

Scripts are independent but should be run in this order (section 4.3 reads
the DSGE solution produced in step 3):

```bash
python scripts/run_section_41.py
python scripts/run_section_42.py
python scripts/run_dsge_solution.py        # requires FRED_API_KEY
python scripts/run_section_43_and_5.py    # requires dsge_solution.npz + fgs-data.txt
python scripts/run_extension.py           # requires FRED-QD CSV + SPF XLSX files
```

Figures are saved to `results/figures/` (PNG + PDF).
Tables are saved to `results/tables/` (CSV).

## Security note

An earlier version of this repository contained a hardcoded FRED API key in
`DSGESol.py`. **That key has been removed** and replaced with the
`FRED_API_KEY` environment variable. If you obtained a copy before this
change, please revoke the old key at https://fredaccount.stlouisfed.org/apikeys
and generate a new one.
