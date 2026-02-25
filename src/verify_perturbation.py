import torch
import math
import logging
from saeb.experiment.config import SimulationConfig
from saeb.utils.pdb_io import RealPDBFeaturizer
from saeb.experiment.suite import SAEBFlowRefinement, kabsch_rmsd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyPerturbation")

def run_verify():
    pdb_id = "1aq1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = SimulationConfig(pdb_id=pdb_id, target_name="Verified_Target", steps=200)
    featurizer = RealPDBFeaturizer(config=config)
    featurizer.device = device
    
    logger.info(f"Loading {pdb_id}...")
    pos_P, x_P, q_P, (p_center, pos_native), x_L_native, q_L_native, lig_template = featurizer.parse(pdb_id)
    
    # 1. Create perturbed poses (Crystal + 2.0A noise)
    pos_native = pos_native.to(device)
    K = 16 # Test-time compute: 16 candidates
    noises = torch.randn(K, *pos_native.shape, device=device) * 1.5 
    pos_perturbed = pos_native.unsqueeze(0) + noises # (K, N, 3)
    
    initial_rmsd = kabsch_rmsd(pos_perturbed, pos_native).min().item()
    logger.info(f"Initial Best Perturbed RMSD: {initial_rmsd:.2f}A")
    
    # 2. Compute pocket anchor
    dist_to_pocket = torch.cdist(pos_P.unsqueeze(0), p_center.unsqueeze(0).unsqueeze(0))[0, :, 0]
    topk_idx = dist_to_pocket.argsort()[:20]
    pocket_anchor = pos_P[topk_idx].mean(dim=0).to(device)
    
    # 3. Refine
    refiner = SAEBFlowRefinement(config)
    logger.info(f"Starting Refinement ({K} clones, 200 steps)...")
    
    out = refiner.refine(
        pos_L_init=pos_perturbed,
        pos_P=pos_P.to(device),
        x_P=x_P.to(device),
        q_P=q_P.to(device),
        x_L=x_L_native.to(device),
        q_L=q_L_native.to(device),
        pocket_anchor=pocket_anchor,
        device=device,
        mol_template=lig_template, # Enabled MMFF
        steps=200
    )
    
    refined_poses = out["refined_poses"]
    rmsd_after = kabsch_rmsd(refined_poses, pos_native)
    final_rmsd = rmsd_after.min().item()
    
    logger.info(f"Final Best Refined RMSD: {final_rmsd:.2f}A")
    logger.info(f"Improvement: {initial_rmsd - final_rmsd:.2f}A")
    
    if final_rmsd < 1.0:
        logger.info("SUCCESS: Perturbation recovered to sub-1A.")
    else:
        logger.warning("FAILED: Could not recover to sub-1A accurately.")

if __name__ == "__main__":
    run_verify()
