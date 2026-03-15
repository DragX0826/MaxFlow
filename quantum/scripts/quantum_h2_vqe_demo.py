#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from qiskit.quantum_info import SparsePauliOp


def build_h2_mol(distance_angstrom: float) -> Chem.Mol:
    mol = Chem.MolFromSmiles("[H][H]")
    if mol is None:
        raise RuntimeError("Failed to build H2 molecule from SMILES.")
    mol = Chem.AddHs(Chem.RemoveHs(mol), addCoords=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, (0.0, 0.0, float(distance_angstrom)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def compute_uff_energy(distance_angstrom: float) -> float:
    mol = build_h2_mol(distance_angstrom)
    ff = AllChem.UFFGetMoleculeForceField(mol)
    if ff is None:
        raise RuntimeError("UFF is unavailable for H2 in the current RDKit build.")
    return float(ff.CalcEnergy())


def compute_vqe_energy(distance_angstrom: float, maxiter: int) -> tuple[float, float]:
    from qiskit.primitives import StatevectorEstimator
    from qiskit.circuit.library import TwoLocal
    from qiskit_algorithms import NumPyMinimumEigensolver, VQE
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.units import DistanceUnit

    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance_angstrom}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    mapper = JordanWignerMapper()

    solver_exact = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
    result_exact = solver_exact.solve(problem)
    exact_energy = float(result_exact.total_energies[0].real)

    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    ansatz = TwoLocal(
        qubit_op.num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        entanglement="full",
        reps=2,
    )
    optimizer = SLSQP(maxiter=maxiter)
    solver_vqe = GroundStateEigensolver(
        mapper,
        VQE(
            estimator=StatevectorEstimator(),
            ansatz=ansatz,
            optimizer=optimizer,
        ),
    )
    result_vqe = solver_vqe.solve(problem)
    vqe_energy = float(result_vqe.total_energies[0].real)
    return vqe_energy, exact_energy


def compute_vqe_energy_fallback(maxiter: int) -> tuple[float, float, float]:
    from qiskit.primitives import StatevectorEstimator
    from qiskit.circuit.library import TwoLocal
    from qiskit_algorithms import NumPyMinimumEigensolver, VQE
    from qiskit_algorithms.optimizers import SLSQP

    bond_length = 0.735
    hamiltonian = SparsePauliOp.from_list(
        [
            ("II", -1.052373245772859),
            ("IZ", 0.39793742484318045),
            ("ZI", -0.39793742484318045),
            ("ZZ", -0.01128010425623538),
            ("XX", 0.18093119978423156),
        ]
    )
    exact_solver = NumPyMinimumEigensolver()
    exact_result = exact_solver.compute_minimum_eigenvalue(hamiltonian)
    ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=2)
    vqe_solver = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=SLSQP(maxiter=maxiter),
    )
    vqe_result = vqe_solver.compute_minimum_eigenvalue(hamiltonian)
    return (
        bond_length,
        float(np.real(vqe_result.eigenvalue)),
        float(np.real(exact_result.eigenvalue)),
    )


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    best_idx = df["vqe_energy"].idxmin()
    best_row = df.loc[best_idx]
    md = f"""# H2 VQE Demo

This run compares hydrogen bond-length energy curves from:

- VQE with Qiskit Nature (`sto3g`, Jordan-Wigner, `TwoLocal` ansatz)
- Exact diagonalization on the same qubit Hamiltonian
- RDKit UFF as a classical force-field baseline

Best VQE point:

- bond_length_angstrom: {best_row["bond_length_angstrom"]:.3f}
- vqe_energy_hartree: {best_row["vqe_energy"]:.8f}
- exact_energy_hartree: {best_row["exact_energy"]:.8f}
- uff_energy_kcal_mol: {best_row["uff_energy"]:.8f}

Interpretation:

- VQE and exact diagonalization live on the same quantum chemistry scale (Hartree).
- UFF is an empirical force field, so its absolute values are not directly comparable.
- The useful comparison is the shape of the energy curve and where the minimum occurs.
"""
    (out_dir / "README.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small H2 VQE vs UFF comparison.")
    parser.add_argument("--out_dir", type=Path, default=Path("results/quantum_h2_vqe"))
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--min_bond", type=float, default=0.3)
    parser.add_argument("--max_bond", type=float, default=2.5)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bond_lengths = np.linspace(args.min_bond, args.max_bond, args.points)
    rows = []
    use_fallback = False
    try:
        import pyscf  # noqa: F401

        for bond_length in bond_lengths:
            print(f"[run] bond_length={bond_length:.3f} A", flush=True)
            vqe_energy, exact_energy = compute_vqe_energy(float(bond_length), args.maxiter)
            uff_energy = compute_uff_energy(float(bond_length))
            rows.append(
                {
                    "bond_length_angstrom": float(bond_length),
                    "vqe_energy": vqe_energy,
                    "exact_energy": exact_energy,
                    "uff_energy": uff_energy,
                    "abs_vqe_minus_exact": abs(vqe_energy - exact_energy),
                    "mode": "full_scan",
                }
            )
    except Exception as exc:
        use_fallback = True
        bond_length, vqe_energy, exact_energy = compute_vqe_energy_fallback(args.maxiter)
        print(f"[fallback] PySCF unavailable ({exc}). Using standard H2 tutorial Hamiltonian.", flush=True)
        nearest_idx = int(np.argmin(np.abs(bond_lengths - bond_length)))
        for local_bond in bond_lengths:
            current_idx = len(rows)
            is_quantum_point = current_idx == nearest_idx
            rows.append(
                {
                    "bond_length_angstrom": float(local_bond),
                    "vqe_energy": vqe_energy if is_quantum_point else np.nan,
                    "exact_energy": exact_energy if is_quantum_point else np.nan,
                    "uff_energy": compute_uff_energy(float(local_bond)),
                    "abs_vqe_minus_exact": abs(vqe_energy - exact_energy) if is_quantum_point else np.nan,
                    "mode": "fallback_single_point",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "energy_scan.csv", index=False)

    plt.figure(figsize=(8, 5))
    valid_quantum = df["vqe_energy"].notna()
    if valid_quantum.sum() > 1:
        plt.plot(df["bond_length_angstrom"], df["vqe_energy"], marker="o", label="VQE (Hartree)")
        plt.plot(df["bond_length_angstrom"], df["exact_energy"], marker="s", label="Exact (Hartree)")
        plt.xlabel("H-H bond length (Angstrom)")
        plt.ylabel("Energy (Hartree)")
        plt.title("H2 energy curve from VQE")
    else:
        qrow = df[valid_quantum].iloc[0]
        plt.bar(["VQE", "Exact"], [qrow["vqe_energy"], qrow["exact_energy"]], color=["tab:blue", "tab:orange"])
        plt.ylabel("Energy (Hartree)")
        plt.title(f"H2 single-point energy at {qrow['bond_length_angstrom']:.3f} A")
    plt.legend() if valid_quantum.sum() > 1 else None
    plt.tight_layout()
    plt.savefig(args.out_dir / "h2_vqe_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["bond_length_angstrom"], df["uff_energy"], marker="^", color="tab:red", label="UFF (kcal/mol)")
    plt.xlabel("H-H bond length (Angstrom)")
    plt.ylabel("Energy (kcal/mol)")
    plt.title("H2 energy curve from UFF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "h2_uff_curve.png", dpi=200)
    plt.close()

    write_summary(df, args.out_dir)
    if use_fallback:
        note = (
            "PySCF was unavailable on this machine, so the quantum result uses the standard "
            "2-qubit H2 tutorial Hamiltonian at 0.735 Angstrom while UFF is still scanned over bond lengths.\n"
        )
        with (args.out_dir / "README.md").open("a", encoding="utf-8") as handle:
            handle.write("\n" + note)
    print(f"[done] outputs written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
