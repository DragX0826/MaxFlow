import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

class ShortcutFlowLoss(nn.Module):
    """
    CBSF Loss for training:
      L_fm:   Huber( v_pred, v_target )          — flow matching objective
      L_x1:   confidence-weighted Huber( x1_pred, x1_target ) — shortcut supervision
      L_conf: entropy regularisation             — prevents confidence collapse
    
    pos_native is ONLY used in train mode. In inference, call .inference_loss() instead.
    """
    def __init__(self, lambda_x1: float = 1.0, lambda_conf: float = 0.01):
        super().__init__()
        self.lambda_x1 = lambda_x1
        self.lambda_conf = lambda_conf

    def forward(self, v_pred, x1_pred, confidence, v_target, x1_target, B, N):
        """Train-mode loss. x1_target = pos_native expanded to batch."""
        v_pred = v_pred.view(B, N, 3)
        v_target = v_target.view(B, N, 3)
        l_fm = F.huber_loss(v_pred, v_target, delta=1.0)

        # Fix: Ensure x1_target is handled correctly regardless of input shape
        if x1_target.dim() == 2: # (N, 3)
            x1_target = x1_target.view(1, N, 3).expand(B, -1, -1)
        elif x1_target.dim() == 3: # (B or 1, N, 3)
            if x1_target.size(0) == 1:
                x1_target = x1_target.expand(B, -1, -1)
        
        x1_pred = x1_pred.view(B, N, 3)
        conf = confidence.view(B, N, 1)
        # Confidence gates the shortcut loss — high confidence → strong x1 supervision
        l_x1 = (conf * F.huber_loss(x1_pred, x1_target, delta=2.0, reduction='none')).mean()

        # Entropy regularisation: prevent confidence from saturating to 0 or 1
        l_conf = -self.lambda_conf * torch.mean(
            conf * torch.log(conf + 1e-8) + (1 - conf) * torch.log(1 - conf + 1e-8)
        )
        return l_fm + self.lambda_x1 * l_x1 + l_conf

    def inference_loss(self, v_pred, x1_pred, confidence, euler_pos, B, N):
        """
        Inference-mode self-consistency loss (no pos_native needed).
        v_pred and x1_pred should agree with the Euler step.
        """
        v_pred = v_pred.view(B, N, 3)
        x1_pred = x1_pred.view(B, N, 3)
        euler_pos = euler_pos.view(B, N, 3).detach()
        # x1_pred should point to where Euler step lands
        l_self = F.huber_loss(x1_pred, euler_pos, delta=2.0)
        # Velocity magnitude regularisation (prevent runaway)
        l_reg = 0.01 * v_pred.pow(2).mean()
        return l_self + l_reg

def pat_step(pos_L, v_pred, f_phys, alpha_ema, confidence, dt):
    """
    Physics-Adaptive Trust (PAT) step.
    Blends neural flow and physics force based on EMA-smoothed trust.
    
    Args:
        pos_L: (B, N, 3)
        v_pred: (B, N, 3)  - Neural direction (Å/step)
        f_phys: (B, N, 3)  - Physical force (-grad E)
        alpha_ema: (B, N, 1) - EMA trust weights (includes CosSim + Confidence)
        confidence: (B, N, 1) - Neural confidence (0 to 1)
        dt: step size
    """
    # Fix Bug 3: Force/Flow Scale Matching
    # f_phys can be huge or tiny. We scale it to match v_pred magnitude
    # This prevents physics from drowning out the model gradients or being ignored.
    v_norm = v_pred.norm(dim=-1, keepdim=True).detach()
    # Bug Fix 18: Add floor to v_norm for f_phys scaling.
    # Ensures physics guidance continues even if neural model is stagnant.
    target_norm = torch.clamp(v_norm, min=0.02) 
    f_norm = f_phys.norm(dim=-1, keepdim=True)
    f_phys_scaled = f_phys / (f_norm + 1e-8) * target_norm
    
    velocity = (1.0 - alpha_ema) * v_pred + alpha_ema * f_phys_scaled
    
    # Fix Bug 2: Confidence Contradiction
    # Instead of velocity * confidence (which zeros out step if conf=0),
    # we use a softer scaling that allows for physics-guided exploration even when unsure.
    step_scale = 0.5 + 0.5 * confidence.detach()  # Range [0.5, 1.0]
    
    new_pos = pos_L + velocity * (step_scale * dt)
    return new_pos

def langevin_noise(pos_shape, temperature, dt, device, x_L=None):
    """
    Generates Langevin stochastic noise: sqrt(2*T*dt/m) * N(0,1)
    Supports mass-weighting based on atom types.
    """
    if temperature <= 1e-6:
        return torch.zeros(pos_shape, device=device)
    
    noise = torch.randn(pos_shape, device=device)
    
    # Default mass = 1.0 (Hydrogen or generic)
    mass = torch.ones((pos_shape[0], pos_shape[1], 1), device=device)
    
    if x_L is not None:
        # x_L has [C, N, O, S, F, P, Cl, Br, I] at indices 0-8
        types = x_L[..., :9] # (B, N, 9)
        # Atomic masses: C=12, N=14, O=16, S=32, F=19, P=31, Cl=35, Br=80, I=127
        mass_coeffs = torch.tensor([12.0, 14.0, 16.0, 32.0, 19.0, 31.0, 35.0, 80.0, 127.0], device=device)
        mass = (types * mass_coeffs).sum(dim=-1, keepdim=True)
        # Issue 7 Fix: Mass Protection
        # Soft probabilities might lead to mass < 12.0 (Carbon); 
        # clamp to Carbon-scale to prevent noise from exploding on light/uncertain atoms.
        mass = torch.clamp(mass, min=12.0) 
    
    # Fluctuating-Dissipation Theorem: scale = sqrt(2 * T * dt / mass)
    scale = torch.sqrt(torch.tensor(2.0 * temperature * dt, device=device) / mass)
    return scale * noise

def shortcut_step(pos_L, v_pred, x1_pred, confidence, t, dt):
    """
    Legacy CBSF shortcut step (compatibility shim for PAT).
    """
    # Simply use pat_step but with f_phys = 0 and alpha_ema = 0
    # Or just re-implement the simple version:
    conf = confidence
    euler_step = pos_L + v_pred * dt
    new_pos = (1.0 - conf) * euler_step + conf * x1_pred
    return new_pos


def run_with_recycling(model, recycling_encoder, pos_L, x_L, x_P, pos_P, h_P, t_flow, n_recycle=3):
    """Iterative recycling wrapper for SAEB-Flow."""
    B, N, _ = pos_L.shape
    correction = None
    
    for i in range(n_recycle):
        # In a real implementation, hooks or side-channels are used to inject correction
        # For the refactor, we simulate the iterative step
        out = model(x_L=x_L, x_P=x_P, pos_L=pos_L, pos_P=pos_P, t=t_flow)
        
        if 'v_pred' in out:
             dt = 1.0 / n_recycle
             # Ensure t_flow is passed as a float
             t_val = t_flow[0].item() if torch.is_tensor(t_flow) else t_flow
             pos_L = shortcut_step(pos_L, out['v_pred'].view(B, N, 3), 
                                  out.get('x1_pred', pos_L).view(B, N, 3), 
                                  out.get('confidence', torch.zeros_like(pos_L[..., :1])), 
                                  t_val, dt)
        
        if i < n_recycle - 1 and 'latent' in out:
             # prev_latent is used to generate the recycling signal
             pass 

    return out

def integrate_innovations(config, backbone, device):
    """Wires together all innovation components."""
    from .model import ShortcutFlowHead, RecyclingEncoder # Local imports to avoid circularity
    
    innovations = {
        'recycling_encoder': RecyclingEncoder(hidden_dim=64).to(device),
        'shortcut_loss_fn': ShortcutFlowLoss().to(device),
        'phpd_scheduler': None # Placeholder
    }
    return innovations
