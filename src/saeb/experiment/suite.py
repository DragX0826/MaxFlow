import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import os

from ..physics.engine import PhysicsEngine
from ..physics.config import ForceFieldParameters
from ..reporting.visualizer import PublicationVisualizer
from ..utils.pdb_io import RealPDBFeaturizer, save_points_as_pdb
from ..core.model import SAEBFlowBackbone, RectifiedFlow
from ..core.innovations import ShortcutFlowLoss, pat_step, langevin_noise
from .config import SimulationConfig

logger = logging.getLogger("SAEB-Flow.experiment.suite")


# ── Kabsch-aligned RMSD ──────────────────────────────────────────────────────

def kabsch_rmsd(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Kabsch-algorithm RMSD (optimal rotation + translation alignment).
    
    Args:
        pred : (B, N, 3)
        ref  : (N, 3) or (1, N, 3) — reference (crystal) pose
    Returns:
        rmsd : (B,) — per-sample aligned RMSD in Angstroms
    """
    if ref.dim() == 2:
        ref = ref.unsqueeze(0)   # (1, N, 3)
    ref = ref.expand(pred.size(0), -1, -1)   # (B, N, 3)

    # Centre both
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    ref_c  = ref  - ref.mean(dim=1, keepdim=True)

    # Covariance: H = pred^T @ ref  (B, 3, 3)
    H = pred_c.transpose(1, 2) @ ref_c

    try:
        U, S, Vh = torch.linalg.svd(H)
        # Correct reflection: det(Vh^T U^T)
        d = torch.linalg.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2))
        sign_mat = torch.diag_embed(torch.stack(
            [torch.ones_like(d), torch.ones_like(d), d], dim=-1))
        R = Vh.transpose(-1, -2) @ sign_mat @ U.transpose(-1, -2)
        pred_rot = pred_c @ R.transpose(-1, -2)
    except Exception:
        # Fallback: translation-only if SVD fails (e.g. collinear atoms)
        pred_rot = pred_c

    sq_diff = (pred_rot - ref_c).pow(2).sum(dim=-1)   # (B, N)
    return torch.sqrt(sq_diff.mean(dim=-1).clamp(min=0.0))  # (B,)


# ── Experiment class ──────────────────────────────────────────────────────────

class SAEBFlowExperiment:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.featurizer = RealPDBFeaturizer(config=config)
        ff_params = ForceFieldParameters(no_physics=config.no_physics, no_hsa=config.no_hsa)
        self.phys = PhysicsEngine(ff_params)
        self.visualizer = PublicationVisualizer()

    def run(self, device=None):
        logger.info(f"SAEB-Flow | {self.config.target_name} ({self.config.pdb_id})")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        # Set device on featurizer immediately so ESM and all tensors are on right GPU
        self.featurizer.device = device
        self.phys = self.phys.to(device)

        # ── Data ────────────────────────────────────────────────────────────
        pos_P, x_P, q_P, (p_center, pos_native), x_L_native, q_L_native, lig_template = \
            self.featurizer.parse(self.config.pdb_id)

        if pos_native is None or pos_native.shape[0] == 0:
            raise ValueError(f"No native ligand atoms found for {self.config.pdb_id}")

        q_L_native = q_L_native.to(device)
        B = self.config.batch_size
        N = pos_native.shape[0]

        # ── Ligand ensemble initialisation ──────────────────────────────────
        # Spread clones uniformly around pocket centre
        pos_L = nn.Parameter(
            p_center.unsqueeze(0).expand(B, N, -1).clone() +
            torch.randn(B, N, 3, device=device) * 3.0
        )
        x_L = nn.Parameter(x_L_native.unsqueeze(0).expand(B, -1, -1).clone())

        # ── Model ───────────────────────────────────────────────────────────
        backbone = SAEBFlowBackbone(167, 64, num_layers=2).to(device)
        model    = RectifiedFlow(backbone).to(device)

        cbsf_loss_fn = ShortcutFlowLoss(lambda_x1=1.0, lambda_conf=0.01).to(device)

        # Optimiser: positions + model
        opt = torch.optim.AdamW(
            [pos_L, x_L] + list(model.parameters()),
            lr=self.config.lr, weight_decay=1e-4
        )
        # Warm-up (100 steps) then cosine decay
        total_steps = self.config.steps
        warmup_steps = min(50, total_steps // 6)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * prog))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        # Pre-expand protein tensors (shared across batch)
        pos_P_b = pos_P.unsqueeze(0).expand(B, -1, -1)
        x_P_b   = x_P.unsqueeze(0).expand(B, -1, -1)
        q_L_b   = q_L_native.unsqueeze(0).expand(B, -1)

        is_train = (self.config.mode == "train")

        history_E    = []
        history_RMSD = []          # min over ensemble
        history_FM   = []
        history_CosSim = []        # Flow alignment with Physics force

        # PAT State (Magma Inspired)
        alpha_ema = torch.zeros(B, N, 1, device=device) 
        beta_ema  = 0.9    # Magma momentum
        tau       = 2.0    # Tempered sigmoid scale

        # ── Precompute pocket residue mask for guided exploration ────────────
        # Finds protein atoms within 6Å of the pocket centre — used as magnet
        with torch.no_grad():
            dist_to_pocket = torch.cdist(pos_P.unsqueeze(0), p_center.unsqueeze(0).unsqueeze(0))[0, :, 0]
            pocket_mask = (dist_to_pocket < 6.0)  # (M,) bool
            
            # Robust Anchor (Bug fix): Ensure at least 20 atoms are used
            if pocket_mask.sum() < 20:
                # Fallback to Top-K closest atoms if 6A radius is too sparse
                k = min(20, len(dist_to_pocket))
                topk_idx = torch.topk(dist_to_pocket, k=k, largest=False).indices
                pocket_anchor = pos_P[topk_idx].mean(dim=0)
            else:
                pocket_anchor = pos_P[pocket_mask].mean(dim=0)

        prev_pos_L = prev_latent = None

        for step in range(total_steps):
            # t in (0, 1] — avoid t=0 where conditional field is undefined
            t   = (step + 1) / max(total_steps, 1)
            t_t = torch.full((B,), t, device=device)

            opt.zero_grad()

            # ── Forward ─────────────────────────────────────────────────────
            out = model(
                x_L=x_L, x_P=x_P_b,
                pos_L=pos_L, pos_P=pos_P_b,
                t=t_t,
                prev_pos_L=prev_pos_L,
                prev_latent=prev_latent,
            )
            v_pred     = out['v_pred']      # (B, N, 3)
            x1_pred    = out['x1_pred']     # (B, N, 3)
            confidence = out['confidence']  # (B, N, 1)

            # Store latent for recycling
            prev_pos_L  = pos_L.detach().clone()
            prev_latent = out['latent'].detach().clone()

            # ── Loss ────────────────────────────────────────────────────────
            dt = 1.0 / total_steps
            if is_train:
                # Conditional vector field: v*(xt|x1) = (x1 - xt)/(1 - t)
                v_target = (pos_native.unsqueeze(0) - pos_L.detach()) / (1.0 - t + 1e-6)
                loss_fm = cbsf_loss_fn(
                    v_pred.view(B * N, 3), x1_pred.view(B * N, 3), confidence.view(B * N, 1),
                    v_target.view(B * N, 3), pos_native, B, N
                )
            else:
                # Inference: self-consistency (no pos_native used)
                euler_pos = (pos_L.detach() + v_pred * dt)
                loss_fm = cbsf_loss_fn.inference_loss(
                    v_pred.view(B * N, 3), x1_pred.view(B * N, 3),
                    confidence.view(B * N, 1), euler_pos, B, N
                )

            # pos_L.requires_grad_(True) is already set (it's an nn.Parameter)
            raw_energy, e_hard, alpha, energy_clamped = self.phys.compute_energy(
                pos_L, pos_P, q_L_b, q_P, x_L, x_P, t
            )
            # Issue 8 Fix: Gradient Isolation
            # Detach raw_energy for loss_phys to prevent it from contributing to pos_L.grad.
            # We want f_phys to be the EXPLICIT and ONLY physical guide in the PAT step.
            loss_phys = raw_energy.mean().detach() * 0.01

            # Pocket Guidance (decays to 0 at t=0.5)
            lig_centroid = pos_L.mean(dim=1)
            pocket_dist  = (lig_centroid - pocket_anchor.unsqueeze(0)).pow(2).sum(-1)
            guidance_weight = max(0.1 * (1.0 - t * 2.0), 0.0)
            loss_pocket = guidance_weight * pocket_dist.mean()

            # Compute physical force BEFORE backward to reuse graph
            f_phys = -torch.autograd.grad(
                raw_energy.sum(), pos_L, retain_graph=False, create_graph=False
            )[0].detach()  # detach so it doesn't interfere with backward

            # ── Backward & Optimizer Step
            # Bug Fix 17: In inference, only optimize pos_L via physical forces & guidance.
            # DO NOT backward loss_fm as it leads to zero-velocity trivial solutions in self-consistency.
            if is_train:
                (loss_fm + loss_phys + loss_pocket).backward()
            else:
                (loss_phys + loss_pocket).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            scheduler.step()

            # ── Metrics (before position overwrite) ────────────────────────
            history_E.append(energy_clamped.mean().item())
            history_FM.append(loss_fm.item())

            log_every = max(total_steps // 10, 1)
            if step % log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    rmsd = kabsch_rmsd(pos_L.detach(), pos_native)
                    r_min, r_mean = rmsd.min().item(), rmsd.mean().item()
                history_RMSD.append(r_min)

            # ── Langevin Temperature Annealing ─────────────────────────────
            # Bug Fix: Use exponential decay for smoother transition and residual exploration
            decay_rate = 5.0
            T_curr = self.config.temp_start * math.exp(-decay_rate * step / total_steps) + 1e-4

            # ── PAT: Physics-Adaptive Trust (Magma Inspired) ───────────────
            # Atom-wise CosSim → Tempered Sigmoid → EMA smoothing
            with torch.no_grad():
                c_i = F.cosine_similarity(v_pred.detach(), f_phys, dim=-1).unsqueeze(-1)
                alpha_tilt = torch.sigmoid(c_i / tau)
                
                # Confidence-weighted trust: lower confidence => more physics trust
                # We blend alpha_tilt with (1 - confidence)
                # This ensures that when the neural model is unsure, physics takes over.
                alpha_tilt = 0.5 * alpha_tilt + 0.5 * (1.0 - confidence.detach())
                
                alpha_ema = beta_ema * alpha_ema + (1.0 - beta_ema) * alpha_tilt
                history_CosSim.append(c_i.mean().item())

            # ── PAT + Langevin Combined Position Update ────────────────────
            with torch.no_grad():
                # Bug Fix: Pass x_L for mass-weighted noise
                noise   = langevin_noise(pos_L.shape, T_curr, dt, device, x_L=x_L)
                pos_new = pat_step(pos_L, v_pred.detach(), f_phys, alpha_ema, confidence.detach(), dt)
                pos_L.data.copy_(pos_new + noise)

            # ── Alpha Hardening (called once per step) ─────────────────────
            with torch.no_grad():
                self.phys.update_alpha(f_phys.norm(dim=-1).mean().item())

            # ── Log ────────────────────────────────────────────────────────
            if step % log_every == 0 or step == total_steps - 1:
                # Calculate average force norm for logging
                f_norm_avg = f_phys.norm(dim=-1).mean().item()
                logger.info(
                    f"  [{step+1:4d}/{total_steps}] "
                    f"E={history_E[-1]:8.1f}  F_phys={f_norm_avg:.4f}  "
                    f"CosSim={history_CosSim[-1]:.3f}  α={alpha:.2f}  "
                    f"T={T_curr:.3f}  "
                    f"RMSD_min={history_RMSD[-1]:.2f}A  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # ── Final evaluation ─────────────────────────────────────────────────
        with torch.no_grad():
            final_rmsd = kabsch_rmsd(pos_L.detach(), pos_native)
            best_idx   = final_rmsd.argmin()
            best_rmsd  = final_rmsd[best_idx].item()
            best_pos   = pos_L[best_idx].detach().cpu().numpy()

        print(f"\n{'='*55}")
        print(f" {self.config.pdb_id:8s}  best={best_rmsd:.2f}A  "
              f"mean={final_rmsd.mean():.2f}A  E={history_E[-1]:.1f}")
        print(self.visualizer.interpreter.interpret_energy_trend(history_E))
        print(f"{'='*55}\n")

        # Save best pose
        os.makedirs("results", exist_ok=True)
        save_points_as_pdb(best_pos, f"results/{self.config.pdb_id}_best.pdb")

        # Per-target plots
        self.visualizer.plot_convergence_dynamics(
            history_E, filename=f"conv_{self.config.pdb_id}.pdf"
        )
        self.visualizer.plot_rmsd_convergence(
            history_RMSD, filename=f"rmsd_{self.config.pdb_id}.pdf"
        )
        self.visualizer.plot_alignment_trends(
            history_CosSim, filename=f"align_{self.config.pdb_id}.pdf"
        )

        return {
            "pdb_id":        self.config.pdb_id,
            "best_rmsd":     best_rmsd,
            "mean_rmsd":     final_rmsd.mean().item(),
            "final_energy":  history_E[-1],
            "mean_cossim":   np.mean(history_CosSim) if history_CosSim else 0.0,
            "steps":         total_steps,
        }
