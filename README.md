# PoseRefineLab

PoseRefineLab is a research codebase for protein-ligand pose refinement, benchmark analysis, and failure-mode auditing. The repository combines the current SAEB docking pipeline, benchmark/report generation utilities, and a compact QM negative-result study used to test whether xTB rescoring improves final pose ranking.

The public repository is intentionally scoped to the code and artifacts that support the current technical conclusions: stable benchmark execution, explicit search-vs-selection diagnosis, and reproducible report packages under `reports/`.

## Repository At A Glance

- `src/` — main docking, refinement, scoring, and benchmark code
- `scripts/` — report generation, audits, and benchmark helpers
- `reports/` — curated docking and QM result packages for external review
- `docs/` — current project status and technical notes
- `quantum/` — QM/xTB utilities used in the pilot rescoring study

## Current Technical Position

- `SOCM` is slightly better on aggregate docking accuracy
- `FK-SMC + SOCM` is more stable across seeds
- hard failures are primarily `search-limited`, not just reranking failures
- MMFF outlier contamination is handled by auto-disable safeguards
- QM rescoring is retained as a pilot negative result, not a claimed improvement

## Selected Figures

![Docking stability comparison](reports/docking/stability_comparison.png)

![Docking benchmark summary](reports/docking/summary_metrics.png)

![H2 VQE bond-length scan](reports/quantum/h2_vqe_scan.png)

## Installation

Use Python 3.10+.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Rebuilding The Reports

From the repository root:

```bash
python scripts/build_reports.py
```

This regenerates:

- `reports/docking/`
- `reports/quantum/`

when the required benchmark CSV inputs and QM summary inputs are available locally.

## Key Entry Points

- `python src/run_benchmark.py --help`
- `python src/run_astex10_fksmc_socm.py --help`
- `python scripts/search_selection_gap_audit.py --help`

## Notes

- Intermediate Kaggle outputs, experimental scratch results, archives, and legacy submission folders are intentionally excluded from GitHub.
- The repository is meant to present the current working code and the current report packages, not every historical artifact.
