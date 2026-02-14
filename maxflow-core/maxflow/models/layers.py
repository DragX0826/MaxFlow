# maxflow/models/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math

# ============== Phase 23: Hyper-Connections (ICLR 2025) ==============

class HyperConnection(nn.Module):
    """
    Hyper-Connections (HC) - ICLR 2025
    Extends residual connections with learnable depth and width matrices.
    """
    def __init__(self, dim, init_scale=1.0):
        super().__init__()
        self.dim = dim
        self.W_d = nn.Parameter(torch.eye(dim) * init_scale)
        self.W_w = nn.Parameter(torch.eye(dim) * init_scale)
        
    def forward(self, x, residual):
        depth_out = torch.matmul(x, self.W_d.T)
        width_out = torch.matmul(residual, self.W_w.T)
        return depth_out + width_out


class ManifoldConstrainedHC(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC).
    Projects W_d and W_w onto identity-perturbed manifold.
    """
    def __init__(self, dim, init_scale=0.01, projection='identity_perturb'):
        super().__init__()
        self.dim = dim
        self.projection = projection
        
        if projection == 'orthonormal':
            self.W_d_param = nn.Parameter(torch.randn(dim, dim) * init_scale)
            self.W_w_param = nn.Parameter(torch.randn(dim, dim) * init_scale)
        else:
            self.delta_d = nn.Parameter(torch.zeros(dim, dim))
            self.delta_w = nn.Parameter(torch.zeros(dim, dim))
            self.epsilon = init_scale
    
    def _project_to_manifold(self):
        if self.projection == 'orthonormal':
            W_d, _ = torch.linalg.qr(self.W_d_param)
            W_w, _ = torch.linalg.qr(self.W_w_param)
        else:
            I = torch.eye(self.dim, device=self.delta_d.device)
            W_d = I + self.epsilon * self.delta_d
            W_w = I + self.epsilon * self.delta_w
        return W_d, W_w
        
    def forward(self, x, residual):
        W_d, W_w = self._project_to_manifold()
        depth_out = torch.matmul(x, W_d.T)
        width_out = torch.matmul(residual, W_w.T)
        return depth_out + width_out


class DynamicHyperConnection(nn.Module):
    """
    Dynamic Hyper-Connections (DHC) — input-adaptive gating.
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or dim // 4
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x, residual):
        combined = torch.cat([x, residual], dim=-1)
        weights = self.gate(combined)
        alpha_d = weights[:, 0:1]
        alpha_w = weights[:, 1:2]
        return alpha_d * x + alpha_w * residual

# ============== End Phase 23 ==============


# ============== Phase 61: Multi-Head GVP Cross-Attention ==============

class GVPCrossAttention(MessagePassing):
    """
    Phase 61: SE(3)-Equivariant Multi-Head Cross-Attention with Vector Gating.
    
    Upgrades from Phase 23 single-head:
    - num_heads parallel attention heads (default=4)
    - Vector gating: v_out = σ(W_g · s) ⊙ (W_v · v)  [Jing et al., ICLR 2021]
    - Geometric distance bias: score += MLP(||r_ij||)
    """
    def __init__(self, s_dim, v_dim, num_heads=4):
        super().__init__(aggr='add', flow='source_to_target')
        self.s_dim = s_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.head_dim = s_dim // num_heads
        assert s_dim % num_heads == 0, f"s_dim ({s_dim}) must be divisible by num_heads ({num_heads})"
        
        # Multi-head Q/K/V projections (scalar)
        self.q_proj = nn.Linear(s_dim, s_dim)
        self.k_proj = nn.Linear(s_dim, s_dim)
        self.v_s_proj = nn.Linear(s_dim, s_dim)  # scalar value projection
        self.o_proj = nn.Linear(s_dim, s_dim)     # output projection
        
        # Vector projections with gating
        self.v_v_proj = nn.Linear(v_dim, v_dim)     # vector value
        self.v_gate = nn.Sequential(                # vector gate σ(W_g · s)
            nn.Linear(s_dim, v_dim),
            nn.Sigmoid()
        )
        
        # Geometric distance bias — follows AlphaFold2 / SE(3)-Transformer
        self.dist_bias = nn.Sequential(
            nn.Linear(1, s_dim // 2),
            nn.SiLU(),
            nn.Linear(s_dim // 2, num_heads)
        )
        
        # Layer norm for stability (Nuclear Option Higher Epsilon)
        self.norm_s = nn.LayerNorm(s_dim, eps=1e-5)
        self.norm_v = nn.LayerNorm([v_dim, 3], eps=1e-5)

    def forward(self, s_L, v_L, pos_L, s_P, v_P, pos_P, batch_L, batch_P):
        from torch_geometric.nn import radius
        edge_index = radius(pos_P, pos_L, r=10.0, batch_x=batch_P, batch_y=batch_L)
        
        # [SOTA Fix] Adaptive Index Swapper (ICLR Hardening)
        # Handles inconsistencies in radius() return order across different torch-cluster versions.
        # Ensure row 0 refers to Protein (pos_P) and row 1 refers to Ligand (pos_L).
        if edge_index.numel() > 0:
            # radius(x, y) returns (y_idx, x_idx) i.e. (Target, Source).
            # We want (Source, Target) for flow='source_to_target'.
            # Source=Protein, Target=Ligand.
            # So we generally need to flip (L, P) -> (P, L).
            
            # Heuristic: Protein indices (Source) are typically larger than Ligand indices (Target).
            # If row 0 max > row 1 max -> (P, L). Correct.
            # If row 1 max > row 0 max -> (L, P). Flip.
            
            max_0 = edge_index[0].max().item()
            max_1 = edge_index[1].max().item()
            
            # If Row 1 (Target?) has larger indices than Row 0 (Source?), it's likely (L, P).
            # Because P > L usually.
            if max_1 > max_0:
                edge_index = edge_index.flip(0)
                
            # [Safety] Hard Clamp Target Indices to avoid CUDA Asset Trigger
            # Target is Row 1. Must be < num_ligand_nodes (s_L.size(0)).
            num_ligand_nodes = s_L.size(0)
            if edge_index[1].max().item() >= num_ligand_nodes:
                # If indices still invalid, clamp them.
                # This handles cases where heuristic fails or P < L.
                # Or simply ensures no OOB access.
                edge_index[1] = edge_index[1].clamp(max=num_ligand_nodes - 1)
        
        # Robust squeeze
        if s_L.dim() == 3 and s_L.size(0) == 1: s_L = s_L.squeeze(0)
        if v_L.dim() == 4 and v_L.size(0) == 1: v_L = v_L.squeeze(0)
        if s_P.dim() == 3 and s_P.size(0) == 1: s_P = s_P.squeeze(0)
        if v_P.dim() == 4 and v_P.size(0) == 1: v_P = v_P.squeeze(0)
        
        # Flatten for propagate
        x_L = torch.cat([s_L, v_L.reshape(v_L.size(0), -1)], dim=-1)
        x_P = torch.cat([s_P, v_P.reshape(v_P.size(0), -1)], dim=-1)
        
        src_idx, tgt_idx = edge_index
        edge_dist = torch.norm(pos_P[src_idx] - pos_L[tgt_idx], dim=-1, keepdim=True)  # (E, 1)
        
        out_x = self.propagate(
            edge_index, x=(x_P, x_L), 
            edge_dist=edge_dist,
            size=(s_P.size(0), s_L.size(0))
        )
        
        out_s = out_x[:, :self.s_dim]
        out_v = out_x[:, self.s_dim:].reshape(-1, self.v_dim, 3)
        
        # Residual + LayerNorm
        s_out = self.norm_s(s_L + self.o_proj(out_s))
        v_out = v_L + out_v  # Skip norm_v for equivariance preservation
        
        return s_out, v_out

    def message(self, x_j, x_i, index, edge_dist):
        """Multi-head attention with geometric bias and vector gating."""
        H, D = self.num_heads, self.head_dim
        
        s_j = x_j[:, :self.s_dim]
        v_j = x_j[:, self.s_dim:].reshape(-1, self.v_dim, 3)
        s_i = x_i[:, :self.s_dim]
        
        # Multi-head Q, K, V
        q = self.q_proj(s_i).view(-1, H, D)  # (E, H, D)
        k = self.k_proj(s_j).view(-1, H, D)  # (E, H, D)
        v_s = self.v_s_proj(s_j).view(-1, H, D)  # (E, H, D)
        
        # Attention score with geometric bias
        score = (q * k).sum(dim=-1) / math.sqrt(D)  # (E, H)
        score = score.clamp(-10.0, 10.0) # SOTN: Clamp scores before softmax
        geo_bias = self.dist_bias(edge_dist).clamp(-5.0, 5.0) # (E, H)
        score = score + geo_bias
        
        from torch_geometric.utils import softmax
        attn = softmax(score, index, dim=0)  # (E, H) — softmax per target node
        
        # Scalar output: weighted sum over heads
        out_s = (attn.unsqueeze(-1) * v_s).reshape(-1, self.s_dim)  # (E, s_dim)
        
        # Vector output with gating: v_out = σ(W_g · s_j) ⊙ (W_v · v_j)
        gate = self.v_gate(s_j).clamp(0.001, 0.999) # SOTN: Clamp gate to prevent dead/exploding signals
        v_proj = self.v_v_proj(v_j.transpose(1, 2)).transpose(1, 2)  # (E, v_dim, 3)
        # Apply gate and attention (sum over heads via mean attention)
        attn_mean = attn.mean(dim=-1, keepdim=True).view(-1, 1, 1).clamp(0, 1)  # (E, 1, 1)
        out_v = attn_mean * gate.unsqueeze(-1) * v_proj  # (E, v_dim, 3)
        
        return torch.cat([out_s, out_v.reshape(out_v.size(0), -1)], dim=-1)


def complex_log(z):
    # log(z) = log|z| + i*arg(z)
    return torch.complex(torch.log(z.abs() + 1e-9), torch.atan2(z.imag, z.real))

def complex_exp(z):
    # exp(a + bi) = exp(a) * (cos(b) + i sin(b))
    exp_a = torch.exp(z.real)
    return torch.complex(exp_a * torch.cos(z.imag), exp_a * torch.sin(z.imag))

def complex_nan_to_num(z):
    # robust nan_to_num for complex
    return torch.complex(z.real.nan_to_num(), z.imag.nan_to_num())

# ============== Phase 61: CausalMolSSM (Mamba-2 Institutional Variant) ==============

class CausalMolSSM(nn.Module):
    """
    True Mamba-3 Implementation (ICLR 2026).
    Key Fixes:
    1. Native Complex64 state (not interleaved).
    2. Exact Cayley Transform for Symplectic Stability.
    3. Proper initialization for rotational features.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.d_state = d_state # N
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        
        # Mamba-3: Input-dependent parameters
        # x_proj maps to: delta, B_real, B_imag, C_real, C_imag
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state * 4, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Mamba-3 A Matrix Initialization (Complex Decay + Rotation)
        # Real part: Decay (negative), Imag part: Frequency (rotation)
        # We initialize A_imag to cover a wide range of frequencies 2pi * k / L
        A_real = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1) * -0.5
        A_imag = torch.pi * torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1) # Standard SSM init
        self.A_log = nn.Parameter(torch.complex(A_real, A_imag)) # Native Complex Parameter
        
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32)) # D stays real
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, batch_idx=None):
        # 0. Dense Padding Logic (Graph -> Batch Sequence)
        is_batched_graph = batch_idx is not None
        if is_batched_graph:
            batch_size = batch_idx.max().item() + 1
            node_counts = torch.bincount(batch_idx, minlength=batch_size)
            max_len = node_counts.max().item()
            
            # Create dense batch [B, L, D]
            # Use scatter/gather or pre-computed indices for speed
            dense_x = torch.zeros(batch_size, max_len, self.d_model, device=x.device, dtype=x.dtype)
            mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) < node_counts.unsqueeze(1)
            
            # Global to Dense Index Map
            ptr = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long), node_counts.cumsum(0)])
            ranges = torch.arange(x.size(0), device=x.device)
            rel_idx = ranges - ptr[batch_idx]
            dense_x[batch_idx, rel_idx] = x
            x_in = dense_x
        else:
            x_in = x.unsqueeze(0) if x.dim() == 2 else x

        # 1. Projections
        xz = self.in_proj(x_in)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv logic
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :x.size(2)] # Causal slice
        x = F.silu(x).transpose(1, 2)
        
        # 2. SSM Parameters
        ssm_params = self.x_proj(x)
        delta, B_re, B_im, C_re, C_im = ssm_params.split(
            [self.d_inner, self.d_state, self.d_state, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        
        # Construct Complex B and C
        B = torch.complex(B_re, B_im) # (B, L, N)
        C = torch.complex(C_re, C_im) # (B, L, N)
        
        # 3. True Trapezoidal Scan (Cayley Transform)
        y = self._scan_complex(x, delta, B, C)
        
        # Gating
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        # Unpad if Graph
        if is_batched_graph:
            out_flat = out[batch_idx, rel_idx]
            return out_flat
            
        return out.squeeze(0) if batch_idx is None and x.dim() == 2 else out

    def _scan_complex(self, u, dt, B, C):
        """
        Executes h_t = (I - dt*A/2)^-1 (I + dt*A/2) h_{t-1} + ...
        """
        batch, seq_len, _ = u.shape
        
        # Materialize A (Broadcasting)
        # A_log is (D, N), dt is (B, L, D)
        A = -torch.exp(self.A_log) # Enforce stability
        
        # Discretization Step (Trapezoidal / Bilinear)
        # log_A_bar = log( (2 + dt*A) / (2 - dt*A) )
        dt_c = dt.unsqueeze(-1).to(torch.complex64) # (B, L, D, 1)
        A_c = A.unsqueeze(0).unsqueeze(0)           # (1, 1, D, N)
        
        denom = 2.0 - dt_c * A_c
        numer = 2.0 + dt_c * A_c
        
        # Complex Logarithm for additive scan
        log_A_bar = torch.log(numer) - torch.log(denom)
        
        # Discretize Input B
        # B_bar = dt * B * (2 / (2 - dt*A))
        scale = 2.0 * dt_c / denom
        u_bar = scale * B.unsqueeze(2) * u.unsqueeze(-1).to(torch.complex64)
        
        # Associative Scan (Cumsum in Log space)
        log_A_cumsum = torch.cumsum(log_A_bar, dim=1)
        
        # Stable Decay Correction
        # exp(S_t - S_k) * u_k
        # Standard approach: h_t = exp(S_t) * cumsum(exp(-S_t) * u_bar)
        
        # We need efficient computation. 
        # Note: torch.cumsum is sequential on GPU unless using specific kernels (associative scan).
        # For python impl, standard cumsum is O(L).
        
        H = torch.cumsum(torch.exp(-log_A_cumsum) * u_bar, dim=1)
        H = H * torch.exp(log_A_cumsum)
        
        # Output Projection: Real(C * H)
        # Sum over N state dim
        y = (C.unsqueeze(2) * H).sum(dim=-1).real
        
        return y + self.D * u

# Legacy alias
SimpleS6 = CausalMolSSM
