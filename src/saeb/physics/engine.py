import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .config import ForceFieldParameters

logger = logging.getLogger("SAEB-Flow.physics.engine")

class PhysicsEngine(nn.Module):
    """
    Differentiable Molecular Mechanics Engine.
    Implements standard LJ 12-6, Coulombic electrostatics, and Pauli repulsion.
    Refactored as nn.Module for DataParallel device consistency.
    """
    def __init__(self, ff_params: ForceFieldParameters):
        super().__init__()
        self.params = ff_params
        self.register_buffer("prot_radii_map", torch.tensor([1.7, 1.55, 1.52, 1.8], dtype=torch.float32))
        self.register_buffer("vdw_radii", ff_params.vdw_radii)
        self.register_buffer("epsilon", ff_params.epsilon)
        self.register_buffer("standard_valencies", ff_params.standard_valencies)
        self.register_buffer("current_alpha_buffer", torch.tensor([ff_params.softness_start], dtype=torch.float32))
        
        self.hardening_rate = 0.1
        self.max_force_ema = 1.0

    @property
    def current_alpha(self):
        return self.current_alpha_buffer.item()
    
    @current_alpha.setter
    def current_alpha(self, value):
        self.current_alpha_buffer.fill_(value)

    def reset_state(self):
        self.current_alpha = self.params.softness_start
        self.max_force_ema = 1.0

    def update_alpha(self, force_magnitude):
        """Adapt softness based on gradient magnitude (EMA-normalized)."""
        with torch.no_grad():
            self.max_force_ema = 0.99 * self.max_force_ema + 0.01 * force_magnitude
            norm_force = torch.clamp(
                torch.tensor(force_magnitude / (self.max_force_ema + 1e-8), 
                             device=self.current_alpha_buffer.device), 0.0, 1.0)
            decay = self.hardening_rate * torch.sigmoid(5.0 * (0.2 - norm_force))
            self.current_alpha = self.current_alpha * (1.0 - decay.item())
            self.current_alpha = max(self.current_alpha, 0.5)

    def soft_clip_vector(self, v, max_norm=10.0):
        """Direction-preserving soft clip for stability."""
        norm = v.norm(dim=-1, keepdim=True)
        scale = (max_norm * torch.tanh(norm / max_norm)) / (norm + 1e-6)
        return v * scale

    def _ensure_batch(self, pos_L, pos_P, q_L, q_P, x_P):
        """Normalize all inputs to have a batch dimension and broadcast."""
        if pos_P.dim() == 2: pos_P = pos_P.unsqueeze(0)
        if q_P.dim() == 1: q_P = q_P.unsqueeze(0)
        if q_L.dim() == 1: q_L = q_L.unsqueeze(0)
        if x_P.dim() == 2: x_P = x_P.unsqueeze(0)
        
        B = pos_L.shape[0]
        if pos_P.shape[0] == 1 and B > 1: pos_P = pos_P.expand(B, -1, -1)
        if q_P.shape[0] == 1 and B > 1: q_P = q_P.expand(B, -1)
        if q_L.shape[0] == 1 and B > 1: q_L = q_L.expand(B, -1)
        if x_P.shape[0] == 1 and B > 1: x_P = x_P.expand(B, -1, -1)
        return pos_L, pos_P, q_L, q_P, x_P

    def compute_energy(self, pos_L, pos_P, q_L, q_P, x_L, x_P, step_progress=0.0):
        """
        Compute intermolecular interaction energy.
        
        No COM subtraction — operates on raw coordinates to preserve 
        absolute distance semantics required by force field potentials.
        """
        if hasattr(self.params, 'no_physics') and self.params.no_physics:
            zero = torch.zeros(pos_L.shape[0], device=pos_L.device)
            return zero, zero, self.current_alpha, zero
            
        with torch.amp.autocast('cuda', enabled=False):
            pos_L = pos_L.float()
            pos_P = pos_P.float()
            q_L = q_L.float()
            q_P = q_P.float()
            x_L = x_L.float()
            x_P = x_P.float()
            
            pos_L, pos_P, q_L, q_P, x_P = self._ensure_batch(pos_L, pos_P, q_L, q_P, x_P)
            B = pos_L.shape[0]

            # --- Pairwise distances (no COM shift) ---
            diff = pos_L.unsqueeze(2) - pos_P.unsqueeze(1)  # (B, N_L, N_P, 3)
            dist_sq = diff.pow(2).sum(dim=-1)                # (B, N_L, N_P)
            dist = torch.sqrt(dist_sq + 1e-8)
            
            # --- Radii ---
            type_probs_L = x_L[..., :9]
            radii_L = type_probs_L @ self.vdw_radii[:9].float()
            radii_P = x_P[..., :4] @ self.prot_radii_map
            sigma_ij = radii_L.unsqueeze(-1) + radii_P.unsqueeze(1)  # (B, N_L, N_P)
            
            # --- Softened distance for early-stage stability ---
            soft_dist_sq = dist_sq + self.current_alpha * sigma_ij.pow(2) + 1e-8
            soft_dist = torch.sqrt(soft_dist_sq)
            
            # --- Coulombic Electrostatics ---
            dielectric = 4.0
            q_L_exp = q_L.unsqueeze(2)   # (B, N_L, 1)
            q_P_exp = q_P.unsqueeze(1)   # (B, 1, N_P)
            e_elec = (332.06 * q_L_exp * q_P_exp) / (dielectric * soft_dist)
            
            # --- Standard LJ 12-6: 4ε[(σ/r)^12 - (σ/r)^6] ---
            eps_L = torch.relu(type_probs_L @ self.epsilon[:9].float())
            eps_ij = torch.sqrt(eps_L.unsqueeze(-1) * 0.15 + 1e-8)
            
            ratio = sigma_ij / (soft_dist + 1e-8)
            ratio = torch.clamp(ratio, max=5.0) 
            
            ratio6 = ratio.pow(6)
            ratio12 = ratio6.pow(2)
            e_vdw_raw = 4.0 * eps_ij * (ratio12 - ratio6)
            
            # --- Bug 1 Fix: Smooth Clamping for Gradients ---
            # Gradient path: use softplus to prevent dead zones while capping extremes
            # Clipping floor at -10 with softplus mapping
            e_vdw_grad = F.softplus(e_vdw_raw - (-10.0)) + (-10.0)
            # Clipping ceiling at 500 with softplus mapping
            e_vdw_grad = -F.softplus(-(e_vdw_grad - 500.0)) + 500.0
            
            # --- Distance cutoff ---
            CUTOFF = 12.0
            dist_mask = torch.sigmoid((CUTOFF - dist) * 2.0)
            
            e_inter_grad = (e_elec + e_vdw_grad) * dist_mask
            e_soft = e_inter_grad.sum(dim=(1, 2))

            # --- Hydrophobic Solvation Approximation (Moved inside autocast) ---
            e_hsa = torch.zeros(B, device=pos_L.device)
            if hasattr(self.params, 'no_hsa') and not self.params.no_hsa:
                is_C_L = type_probs_L[..., 0]     # carbon probability
                is_C_P = x_P[..., 0]
                mask_cc = is_C_L.unsqueeze(2) * is_C_P.unsqueeze(1)
                hsa_term = 1.0 / (1.0 + (dist / 4.0).pow(4))
                e_hsa = (-0.5 * mask_cc * hsa_term * dist_mask).sum(dim=(1, 2))

            # --- Pauli Repulsion ---
            t_gate = torch.sigmoid(torch.tensor((step_progress - 0.2) * 10.0, device=pos_L.device))
            overlap = torch.relu(sigma_ij * 0.6 - dist)
            e_pauli = t_gate * 100.0 * overlap.pow(2).sum(dim=(1, 2))
            
            # --- Ghost repulsion ---
            if step_progress < 0.15:
                e_ghost = 500.0 * torch.relu(0.5 - dist).pow(2).sum(dim=(1, 2))
            else:
                e_ghost = torch.zeros(B, device=pos_L.device)

            # --- Final Combining (Differentiable) ---
            e_raw = e_soft + 5.0 * e_hsa + e_pauli + e_ghost
            
            # --- NaN safety ---
            if torch.isnan(e_raw).any():
                logger.warning("NaN in raw energy")
                e_raw = torch.nan_to_num(e_raw, nan=0.0)

        # Log energy is clamped for metrics reporting stability
        # Use a harder clamp for the log to stay in -500..5000 range
        log_vdw = torch.clamp(e_vdw_raw, min=-10.0, max=500.0)
        log_soft = ((e_elec + log_vdw) * dist_mask).sum(dim=(1, 2)) + 5.0 * e_hsa
        
        e_soft_final = torch.clamp(log_soft, min=-500.0, max=5000.0)
        e_hard_final = torch.clamp(e_pauli + e_ghost, max=10000.0)
        log_energy   = torch.clamp(e_soft_final + e_hard_final, max=1e6)
        
        return e_raw, e_hard_final, self.current_alpha, log_energy

    def calculate_valency_loss(self, pos_L, x_L):
        """Valency constraint loss."""
        B, N, _ = pos_L.shape
        dist = torch.cdist(pos_L, pos_L) + torch.eye(N, device=pos_L.device).unsqueeze(0) * 10
        neighbors = torch.sigmoid((1.8 - dist) * 10.0).sum(dim=-1)
        type_probs = x_L[..., :9]
        target_valency = type_probs @ self.standard_valencies
        diff_val = neighbors - target_valency
        penalty_over = torch.where(diff_val > 0, 5.0 * diff_val.pow(2), diff_val.pow(2))
        return penalty_over.mean(dim=1)

    def calculate_internal_geometry_score(self, pos_L, target_dist=None):
        """Internal geometry constraint."""
        B, N, _ = pos_L.shape
        dist = torch.cdist(pos_L, pos_L)
        eye = torch.eye(N, device=dist.device).unsqueeze(0)
        mask = (eye < 0.5)
        
        if target_dist is not None:
            if target_dist.dim() == 2:
                target_dist = target_dist.unsqueeze(0)
            expansion_error = F.huber_loss(dist, target_dist, delta=1.0, reduction='none')
            return 100.0 * (expansion_error * mask).sum(dim=(1, 2))
        else:
            clash = torch.clamp(1.2 - dist, min=0.0).pow(2)
            stretch = torch.clamp(dist - 1.8, min=0.0).pow(2)
            return 100.0 * ((clash + stretch) * mask).sum(dim=(1, 2))
