# MaxFlow

MaxFlow is the working repository for our AI docking / pose-refinement project and its associated QM negative-result study.

This repository is now organized around the code that is still relevant, the final evidence packages, and the scripts required to regenerate them.

## What This Repo Contains

- `src/`
  - main SAEB/MaxFlow implementation
  - refinement, scoring, MMFF safeguards, and benchmark logic
- `scripts/`
  - benchmark utilities
  - audit scripts
  - `build_workshop_evidence.py` to rebuild the final evidence packages
- `docs/`
  - current project status and technical notes
- `deliverables/workshop_evidence/`
  - final proof documents, figures, and core tables for:
    - docking project
    - QM negative-result project
- `quantum/`
  - QM/xTB-related scripts and lightweight notes used to build the negative-result package

## Current Project Position

The docking project currently supports a stability-focused claim more strongly than an accuracy-superiority claim.

Supported by the current evidence:

- `SOCM` remains slightly better on aggregate docking accuracy
- `FK-SMC + SOCM` is materially more stable across seeds
- hard failures are now diagnosed as primarily `search-limited`
- MMFF outlier contamination is handled by auto-disable safeguards

The QM line is kept as a documented pilot negative result:

- the xTB rescoring workflow works end-to-end
- but ligand-only and pocket-cluster rescoring did not improve pose ranking on the tested cases

## Key Deliverables

Docking evidence package:

- `deliverables/workshop_evidence/docking_project/proof_document.md`
- `deliverables/workshop_evidence/docking_project/stability_comparison.png`
- `deliverables/workshop_evidence/docking_project/gap_audit_targets.png`
- `deliverables/workshop_evidence/docking_project/summary_metrics.png`

Quantum evidence package:

- `deliverables/workshop_evidence/quantum_project/proof_document.md`
- `deliverables/workshop_evidence/quantum_project/qm_rmsd_comparison.png`
- `deliverables/workshop_evidence/quantum_project/qm_delta_vs_selected.png`

## Rebuilding The Evidence Packages

From the repository root:

```bash
python scripts/build_workshop_evidence.py
```

This regenerates:

- `deliverables/workshop_evidence/docking_project/`
- `deliverables/workshop_evidence/quantum_project/`

from the retained final CSV inputs and QM summary files.

## Benchmark Entry Points

Main benchmark CLI:

```bash
python src/run_benchmark.py --help
```

Astex-10 FK-SMC + SOCM wrapper:

```bash
python src/run_astex10_fksmc_socm.py --help
```

Search-vs-selection audit:

```bash
python scripts/search_selection_gap_audit.py --help
```

## Notes

- Intermediate Kaggle outputs, experimental scratch results, archives, and legacy submission folders are intentionally excluded from GitHub.
- The repository is meant to present the current working code and the final evidence package, not every historical artifact.
