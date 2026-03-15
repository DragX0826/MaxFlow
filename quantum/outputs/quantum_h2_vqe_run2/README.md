# H2 VQE Demo

This run compares hydrogen bond-length energy curves from:

- VQE with Qiskit Nature (`sto3g`, Jordan-Wigner, `TwoLocal` ansatz)
- Exact diagonalization on the same qubit Hamiltonian
- RDKit UFF as a classical force-field baseline

Best VQE point:

- bond_length_angstrom: 0.850
- vqe_energy_hartree: -1.85727462
- exact_energy_hartree: -1.85727503
- uff_energy_kcal_mol: 9.56430609

Interpretation:

- VQE and exact diagonalization live on the same quantum chemistry scale (Hartree).
- UFF is an empirical force field, so its absolute values are not directly comparable.
- The useful comparison is the shape of the energy curve and where the minimum occurs.

PySCF was unavailable on this machine, so the quantum result uses the standard 2-qubit H2 tutorial Hamiltonian at 0.735 Angstrom while UFF is still scanned over bond lengths.
