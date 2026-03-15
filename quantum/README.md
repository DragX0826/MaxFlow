# Quantum Folder

This folder centralizes the local quantum-method side project and its proof artifacts.

Contents:

- `scripts/quantum_h2_vqe_demo.py`
  Runs the local H2 VQE demo and writes CSV, figures, and a short summary.
- `scripts/generate_two_track_proof_pdf.py`
  Builds the one-page PDF that summarizes both project routes:
  the Kaggle docking benchmark route and the local quantum proof route.
- `outputs/quantum_h2_vqe_run2/`
  Generated outputs from the latest local quantum run.
- `outputs/two_track_proof_sheet.pdf`
  One-page proof sheet combining the Kaggle main route and the quantum demo route.

Project split:

- Main route:
  Kaggle docking benchmark for pose prediction, stability, efficiency, and ranking claims.
- Quantum route:
  Small local H2 VQE proof-of-concept used as application/interview evidence for quantum-method exposure.

Notes:

- The current quantum run uses the fallback mode because PySCF was unavailable on this Windows machine.
- The fallback still produces a usable proof artifact: a standard H2 tutorial Hamiltonian VQE point plus a UFF scan.
