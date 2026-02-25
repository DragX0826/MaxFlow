import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from ..utils.esm import get_esm_model

logger = logging.getLogger("SAEB-Flow.core.model")

# --- 0. GEOMETRIC VECTOR PERCEPTRONS (GVP) Core ---

class GVP(nn.Module):
    """
    Geometric Vector Perceptron (Jing et al. 2021).
    Handles (s, V) pairs where s is scalar and V is vector features.
    Guarantees SE(3) equivariance.
    """
    def __init__(self, in_dims, out_dims, vector_gate=True):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        
        self.v_proj = nn.Linear(self.vi, self.vo, bias=False)
        self.s_proj = nn.Sequential(
            nn.Linear(self.si + self.vo, self.so),
            nn.SiLU(),
            nn.Linear(self.so, self.so)
        )
        if vector_gate:
            self.gate_proj = nn.Linear(self.si + self.vo, self.vo)
        
    def forward(self, s, V):
        """
        s: (B, N, si)
        V: (B, N, vi, 3)
        """
        # Vector transformation
        V_out = self.v_proj(V.transpose(-1, -2)).transpose(-1, -2) # (B, N, vo, 3)
        v_norm = torch.norm(V_out, dim=-1, keepdim=False)  # (B, N, vo)
        
        # Scalar transformation with norm-based awareness
        s_combined = torch.cat([s, v_norm], dim=-1)
        s_out = self.s_proj(s_combined)
        
        # Vector gating for non-linearity
        if self.vector_gate:
            gate = torch.sigmoid(self.gate_proj(s_combined)).unsqueeze(-1)
            V_out = V_out * gate
            
        return s_out, V_out


class SinusoidalTimeEmbedding(nn.Module):
    """
    Multi-frequency sinusoidal time embedding.
    Maps scalar t in [0,1] to a rich feature vector, following DDPM/DiT conventions.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, t):
        """t: (B,) -> (B, dim)"""
        args = t.unsqueeze(-1) * self.freqs * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class StructureSequenceEncoder(nn.Module):
    """
    Bridges ESM-2 sequence embeddings with 3D structural context.
    Expects x_P with first 4 dims = atom one-hots, remaining = ESM features.
    """
    def __init__(self, esm_model_name="esm2_t33_650M_UR50D", hidden_dim=64):
        super().__init__()
        self.esm_dim = 1280
        self.adapter = nn.Sequential(
            nn.Linear(self.esm_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # Fallback for non-ESM features (e.g., one-hot only)
        self.fallback = nn.Sequential(
            nn.Linear(25, hidden_dim),
            nn.SiLU()
        )

    def forward(self, x_P):
        x_P = x_P.float()
        feat_dim = x_P.size(-1)
        if feat_dim > 25:
            # ESM features present: skip first 4 atom one-hots
            esm_slice = x_P[..., 4:4+self.esm_dim]
            if esm_slice.size(-1) < self.esm_dim:
                esm_slice = F.pad(esm_slice, (0, self.esm_dim - esm_slice.size(-1)))
            return self.adapter(esm_slice)
        else:
            # Fallback: use raw one-hot features
            if feat_dim < 25:
                x_P = F.pad(x_P, (0, 25 - feat_dim))
            return self.fallback(x_P[..., :25])


class GVPAdapter(nn.Module):
    """
    Equivariant Cross-Attention with Vector support.
    Uses relative vectors (pos_L - pos_P) to populate vector channels.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gvp = GVP((hidden_dim, 1), (hidden_dim, 1)) # Scalar + 1 Vector (relative pos)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x_L, x_P, pos_L, pos_P):
        """
        x_L: (B, N_L, D), x_P: (B, N_P, D)
        pos_L: (B, N_L, 3), pos_P: (B, N_P, 3)
        """
        # Compute pairwise distance bias
        diff = pos_L.unsqueeze(2) - pos_P.unsqueeze(1) # (B, N_L, N_P, 3)
        dist = torch.norm(diff, dim=-1)               # (B, N_L, N_P)
        
        # 1. Scalar Attention
        q = self.q_proj(x_L)
        k = self.k_proj(x_P)
        v = self.v_proj(x_P)
        
        attn_bias = -1e9 * (dist > 10.0).float()
        scores = torch.bmm(q, k.transpose(-1, -2)) / self.scale + attn_bias
        probs = F.softmax(scores, dim=-1)
        
        # 2. Vector Aggregation (The Equivariant Part)
        # Aggregated context vector V = sum(weights * relative_directions)
        # We normalize direction to maintain scale stability
        direction = diff / (dist.unsqueeze(-1) + 1e-6)
        V_agg = torch.einsum('bnp, bnpk -> bnk', probs, direction).unsqueeze(2) # (B, N_L, 1, 3)
        
        # 3. Scalar Context
        s_context = torch.bmm(probs, v)
        
        # 4. Refine with GVP
        s_out, V_out = self.gvp(x_L + s_context, V_agg)
        return s_out, V_out, probs


# --- 2. INNOVATIONS (CBSF) ---

class EquivariantFlowHead(nn.Module):
    """
    Equivariant Head using GVP.
    Predicts velocity (v_pred) and endpoint (x1_pred) as vector features.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gvp = GVP((hidden_dim, 1), (hidden_dim, 2)) # Out 2 vectors
        self.conf_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h, V_in, pos_L=None):
        """h: (B, N, D), V_in: (B, N, 1, 3), pos_L: (B, N, 3)"""
        s_out, V_out = self.gvp(h, V_in) # V_out: (B, N, 2, 3)
        
        v_pred = V_out[:, :, 0, :]
        x1_delta = V_out[:, :, 1, :]
        x1_pred = (pos_L + x1_delta) if pos_L is not None else x1_delta
        
        conf = self.conf_proj(s_out)
        return {'v_pred': v_pred, 'x1_pred': x1_pred, 'confidence': conf, 'latent': s_out}


class RecyclingEncoder(nn.Module):
    """AF2-style recycling: encodes pairwise distances of previous pose."""
    def __init__(self, hidden_dim, num_rbf=16):
        super().__init__()
        self.dist_embed = nn.Linear(num_rbf, hidden_dim)
        centers = torch.linspace(0, 20, num_rbf)
        self.register_buffer("rbf_centers", centers)

    def forward(self, prev_pos_L, prev_latent):
        dist = torch.norm(prev_pos_L.unsqueeze(2) - prev_pos_L.unsqueeze(1), dim=-1)
        rbf = torch.exp(-0.5 * (dist.unsqueeze(-1) - self.rbf_centers).pow(2))
        h_recycling = self.dist_embed(rbf.mean(dim=2))
        return prev_latent + h_recycling


# --- 3. BACKBONE ---

class PermutationInvariantBlock(nn.Module):
    """Self-Attention + FFN block for unordered atom sets."""
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        h = self.ff(x)
        return self.norm2(x + h)


class SAEBFlowBackbone(nn.Module):
    """
    Master Architecture: Perception -> Cross-Attention -> Self-Attention -> Policy Head.
    
    All outputs are shaped (B, N, *) — no flattening.
    """
    def __init__(self, node_in_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(node_in_dim, hidden_dim)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.perception = StructureSequenceEncoder(hidden_dim=hidden_dim)
        self.cross_attn = GVPAdapter(hidden_dim)
        self.recycling = RecyclingEncoder(hidden_dim)
        self.reasoning = nn.ModuleList([
            PermutationInvariantBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.head = EquivariantFlowHead(hidden_dim)

    def forward(self, x_L, x_P, pos_L, pos_P, t, prev_pos_L=None, prev_latent=None):
        """
        Args:
            x_L: (B, N_L, D_L) ligand features
            x_P: (B, N_P, D_P) protein features  
            pos_L: (B, N_L, 3) ligand positions
            pos_P: (B, N_P, 3) protein positions
            t: (B,) time in [0, 1]
            prev_pos_L: optional (B, N_L, 3) for recycling
            prev_latent: optional (B, N_L, D) for recycling
        Returns:
            dict with 'v_pred' (B,N,3), 'x1_pred' (B,N,3), 'confidence' (B,N,1), 'latent' (B,N,D)
        """
        B, N, _ = pos_L.shape
        
        # 1. Embed ligand features + time conditioning
        h = self.embedding(x_L)                    # (B, N, D)
        t_emb = self.time_embed(t)                 # (B, D)
        h = h + t_emb.unsqueeze(1)                 # broadcast to all atoms
        
        # 2. Recycling injection (if available)
        if prev_pos_L is not None and prev_latent is not None:
            h = self.recycling(prev_pos_L, h)
        
        # 3. Protein perception + cross-attention (Equivariant)
        h_P = self.perception(x_P)                 # (B, N_P, D)
        h, V, _ = self.cross_attn(h, h_P, pos_L, pos_P) # h: (B,N,D), V: (B,N,1,3)
        
        # 4. Self-attention reasoning (permutation invariant on scalars)
        # V remains equivariant as it is not touched by LayerNorm/Linear on its x,y,z dims
        for layer in self.reasoning:
            h = layer(h)
        
        # 5. Equivariant head — outputs are naturally SE(3) equivariant
        out = self.head(h, V, pos_L=pos_L)
        out['latent'] = h  # for recycling
        return out


class RectifiedFlow(nn.Module):
    """Rectified Flow (Liu et al., 2023) wrapper."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, **kwargs):
        return self.backbone(**kwargs)
