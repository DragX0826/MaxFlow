import os
import torch
import numpy as np
import logging
from pathlib import Path
from saeb.experiment.config import SimulationConfig
from saeb.utils.pdb_io import RealPDBFeaturizer
from saeb.experiment.suite import SAEBFlowRefinement, kabsch_rmsd
from rdkit import Chem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAEB-DiffDock")

def load_diffdock_poses(diffdock_output_dir: str, pdb_id: str):
    """
    DiffDock output format:
    diffdock_output/{pdb_id}/rank{1..40}.sdf
    """
    poses = []
    pose_dir = Path(diffdock_output_dir) / pdb_id
    if not pose_dir.exists():
        logger.warning(f"Pose directory {pose_dir} not found.")
        return None
        
    sdf_files = sorted(pose_dir.glob("rank*.sdf"))
    for sdf_file in sdf_files:
        mol = Chem.SDMolSupplier(str(sdf_file), removeHs=True)[0]
        if mol is not None:
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
            poses.append(pos)
    
    if not poses:
        return None
    return torch.stack(poses)

def run_benchmark(pdb_ids: list, diffdock_dir: str, pdb_dir: str, device: str = "cuda:0", steps: int = 200):
    """
    Main benchmark loop for Path A: Refinement Engine.
    """
    results = []
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    for pdb_id in pdb_ids:
        logger.info(f"== [{pdb_id}] Processing ==")
        
        config = SimulationConfig(pdb_id=pdb_id, target_name="DiffDock_Refine", pdb_dir=pdb_dir, steps=steps)
        featurizer = RealPDBFeaturizer(config=config)
        featurizer.device = device
        
        try:
            pos_P, x_P, q_P, (p_center, pos_native), x_L_native, q_L_native, lig_template = featurizer.parse(pdb_id)
        except Exception as e:
            logger.error(f"  [Skip] Error parsing PDB: {e}")
            continue
            
        diffdock_poses = load_diffdock_poses(diffdock_dir, pdb_id)
        if diffdock_poses is None:
            continue
            
        K = diffdock_poses.shape[0]
        # Calculate initial metrics
        rmsd_before = kabsch_rmsd(diffdock_poses.to(device), pos_native.to(device))
        best_before = rmsd_before.min().item()
        
        # Pocket anchor for guidance
        dist_to_pocket = torch.cdist(pos_P.unsqueeze(0).to(device), p_center.unsqueeze(0).unsqueeze(0).to(device))[0, :, 0]
        topk_idx = dist_to_pocket.argsort()[:20]
        pocket_anchor = pos_P[topk_idx].mean(dim=0).to(device)
        
        # Run Refinement
        refiner = SAEBFlowRefinement(config)
        logger.info(f"  [Refine] Polishing {K} candidates...")
        
        out = refiner.refine(
            pos_L_init=diffdock_poses,
            pos_P=pos_P.to(device),
            x_P=x_P.to(device),
            q_P=q_P.to(device),
            x_L=x_L_native.to(device),
            q_L=q_L_native.to(device),
            pocket_anchor=pocket_anchor,
            device=device,
            mol_template=lig_template,
            steps=steps
        )
        
        refined_poses = out["refined_poses"]
        rmsd_after = kabsch_rmsd(refined_poses.to(device), pos_native.to(device))
        best_after = rmsd_after.min().item()
        
        logger.info(f"  Result: {best_before:.2f}A -> {best_after:.2f}A")
        results.append({
            "pdb_id": pdb_id,
            "before": best_before,
            "after": best_after,
            "delta": best_before - best_after
        })
        
    return results

if __name__ == "__main__":
    # Example usage
    # run_benchmark(["1aq1", "1b8o"], "./diffdock_results", "./pdb_data")
    pass
