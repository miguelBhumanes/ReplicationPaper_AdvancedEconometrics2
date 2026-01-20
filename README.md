# Replication Paper

Code and data to replicate and extend the main empirical and DSGE results.

## Core Scripts
- `Section41.py`
- `Section42.py`
- `section43&5.py`
- `extension.py`

Main replication and extension routines. Each corresponding to the section they are named after. 

## Auxiliary Modules
- `aux_VAR.py`
- `aux_DSGE.py`
- `DSGESol.py`

Helper functions and DSGE model solution.

## Data Inputs
- `2025-12-QD.csv` (McCraken & Ng FRED-MD dataset)
- `fgs-data.txt` (Replication Data provided by Forni, Gambetti, Sala)
- `Median_RGDP_Growth.xlsx` (SPF Expectations Data from FRED Philadelphia - Output Growth)
- `Inflation.xlsx` (SPF Expectations Data from FRED Philadelphia - Inflation)

Raw and processed datasets used in estimation.

## Outputs: Figures
- `example1_figure1.pdf`, `example1_figure1.png`
- `example2_figure2.pdf`, `example2_figure2.png`
- `figure7_replication.pdf`, `figure7_replication.png`

Replication and extension figures.

## Outputs: Tables
- `example1_table1_fevd.csv`
- `example1_table2_spectral_vd.csv`
- `example1_table3_deficiency.csv`
- `example2_table4_deficiency.csv`
- `extension_table.csv`
- `table7_replication.csv`
- `var_residuals_factors.csv`
- `fred_qd_factors.csv` 
- `macro_with_spf_expectations.csv`

Numerical results and intermediate outputs.

## Environment
- `requirements.txt`

Python dependencies.
