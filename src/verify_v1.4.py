import torch
import numpy as np
from saeb.core.dynamics import torus_flow_velocity, FiberBundle, build_fiber_bundle
from rdkit import Chem
from rdkit.Chem import AllChem

def test_torus_wrap():
    print("Testing torus_flow_velocity...")
    theta_pred = torch.tensor([0.1, 0.1, 3.1])
    theta_native = torch.tensor([0.2, 6.2, -3.1]) # 6.2 approx 2pi, -3.1 approx -pi
    t = 0.5
    v = torus_flow_velocity(theta_pred, theta_native, t)
    print(f"Velocities: {v}")
    # Expected: 0.1, ~-0.1, ~0.1 (shortest arc)
    # 6.2 - 0.1 = 6.1 -> wraps to -0.18 approx
    # -3.1 - 3.1 = -6.2 -> wraps to 0.08 approx
    
def test_fiber_bundle_to():
    print("\nTesting FiberBundle.to()...")
    fb = FiberBundle(
        n_atoms=10,
        rotatable_bonds=[(0, 1)],
        downstream_masks=torch.zeros((1, 10)),
        fragment_labels=torch.zeros(10),
        pivot_atoms=[0]
    )
    print(f"Initial device: {fb.downstream_masks.device}")
    if torch.cuda.is_available():
        fb.to("cuda")
        print(f"New device: {fb.downstream_masks.device}")
    else:
        print("CUDA not available, skipping migration test.")

def test_rdkit_topology():
    print("\nTesting RDKit topology detection...")
    # Create a simple flexible molecule: Butane
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    pos = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    
    fb = build_fiber_bundle(pos, mol=mol)
    print(f"Atoms: {fb.n_atoms}")
    print(f"Rotatable Bonds found: {fb.rotatable_bonds}")
    print(f"Masks shape: {fb.downstream_masks.shape}")
    if len(fb.rotatable_bonds) > 0:
        print(f"First mask sum: {fb.downstream_masks[0].sum().item()}")

if __name__ == "__main__":
    test_torus_wrap()
    test_fiber_bundle_to()
    test_rdkit_topology()
