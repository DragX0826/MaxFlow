import torch
import math
import logging
from saeb.experiment.config import SimulationConfig
from saeb.utils.pdb_io import RealPDBFeaturizer
from saeb.experiment.suite import SAEBFlowRefinement, kabsch_rmsd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyInducedFit")

def run_verify():
    pdb_id = "1aq1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = SimulationConfig(pdb_id=pdb_id, target_name="InducedFit_Test", steps=200)
    featurizer = RealPDBFeaturizer(config=config)
    featurizer.device = device
    
    logger.info(f"Loading {pdb_id}...")
    pos_P, x_P, q_P, (p_center, pos_native), x_L_native, q_L_native, lig_template = featurizer.parse(pdb_id)
    
    # 1. Create perturbed ligand pose (2.17A noise)
    pos_native = pos_native.to(device)
    K = 16
    lig_noise = torch.randn(K, *pos_native.shape, device=device) * 1.5
    pos_L_init = pos_native.unsqueeze(0) + lig_noise
    
    # 2. Perturb Protein Pocket (Induced Fit Scenario)
    # Move nearby protein atoms by 0.5A to simulate "off-crystal" state
    dist_to_lig = torch.cdist(pos_P.unsqueeze(0).to(device), pos_native.unsqueeze(0))[0].min(dim=1)[0]
    pocket_mask = dist_to_pocket = dist_to_lig < 4.0
    pos_P_perturbed = pos_P.clone().to(device)
    pos_P_perturbed[pocket_mask] += torch.randn_like(pos_P_perturbed[pocket_mask]) * 0.5
    
    pocket_anchor = pos_P[dist_to_lig.argsort()[:20]].mean(dim=0).to(device)
    
    # 3. Refine - RIGID Baseline
    refiner = SAEBFlowRefinement(config)
    logger.info("--- [Baseline: Rigid Receptor] ---")
    out_rigid = refiner.refine(
        pos_L_init=pos_L_init,
        pos_P=pos_P_perturbed,
        x_P=x_P.to(device),
        q_P=q_P.to(device),
        x_L=x_L_native.to(device),
        q_L=q_L_native.to(device),
        pocket_anchor=pocket_anchor,
        device=device,
        mol_template=lig_template,
        allow_flexible_receptor=False,
        steps=200
    )
    rmsd_rigid = kabsch_rmsd(out_rigid["refined_poses"], pos_native).min().item()
    logger.info(f"Rigid Best RMSD: {rmsd_rigid:.2f}A")
    
    # 4. Refine - FLEXIBLE (Induced Fit)
    logger.info("--- [Proposed: Flexible Receptor] ---")
    out_flex = refiner.refine(
        pos_L_init=pos_L_init,
        pos_P=pos_P_perturbed,
        x_P=x_P.to(device),
        q_P=q_P.to(device),
        x_L=x_L_native.to(device),
        q_L=q_L_native.to(device),
        pocket_anchor=pocket_anchor,
        device=device,
        mol_template=lig_template,
        allow_flexible_receptor=True,
        steps=200
    )
    rmsd_flex = kabsch_rmsd(out_flex["refined_poses"], pos_native).min().item()
    logger.info(f"Flexible Best RMSD: {rmsd_flex:.2f}A")
    
    logger.info(f"Induced Fit Benefit: {rmsd_rigid - rmsd_flex:.2f}A")
    
    if rmsd_flex < rmsd_rigid:
        logger.info("SUCCESS: Flexible receptor improved docking accuracy.")
    else:
        logger.warning("FAILED: Flexible receptor did not show benefit here.")

if __name__ == "__main__":
    run_verify()
