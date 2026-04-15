"""
Central path registry.

All scripts resolve data and output paths through these constants so the
pipeline can be run from any working directory.
"""

from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError(
        "Cannot locate repo root: no pyproject.toml found in any parent directory."
    )


REPO_ROOT = _find_repo_root()

RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create output dirs on first import so scripts never have to mkdir themselves.
for _d in (PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
