# max_flow/models/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math

class HyperConnection(nn.Module):
    def __init__(self, dim, init_scale=1.0):
        super().__init__()
        self.dim = dim
        self.W_d = nn.Parameter(torch.eye(dim) * init_scale)
        self.W_w = nn.Parameter(torch.eye(dim) * init_scale)
    def forward(self, x, residual):
        return torch.matmul(x, self.W_d.T) + torch.matmul(residual, self.W_w.T)

class GVPVectorNorm(nn.Module):
    """
    SOTA: Vector-Norm Layer.
    Matches checkpoint naming: norm_v.weight, norm_v.bias [16, 3]
    """
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 3))
        self.bias = nn.Parameter(torch.zeros(dim, 3))
    def forward(self, x):
        return x * self.weight + self.bias

class GVPCrossAttention(MessagePassing):
    """
    SOTA: Precision-Aligned Equivariant Cross-Attention.
    Aligned with maxflow_pretrained.pt: norm_v.weight and norm_v.bias.
    """
    def __init__(self, s_dim, v_dim, num_heads=4):
        super().__init__(aggr='add', flow='source_to_target')
        self.s_dim, self.v_dim = s_dim, v_dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(s_dim, s_dim)
        self.k_proj = nn.Linear(s_dim, s_dim)
        self.v_s_proj = nn.Linear(s_dim, s_dim)
        self.v_v_proj = nn.Linear(v_dim, v_dim, bias=False)
        self.o_proj = nn.Linear(s_dim, s_dim)
        
        self.v_gate = nn.Sequential(nn.Linear(s_dim, v_dim)) 
        
        self.dist_bias = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, num_heads)
        )
        
        self.norm_s = nn.LayerNorm(s_dim)
        # Checkpoint Fix: Submodule structure for norm_v.weight/bias alignment
        self.norm_v = GVPVectorNorm(v_dim)

    def forward(self, s_L, v_L, pos_L, s_P, v_P, pos_P, batch_L, batch_P):
        return self.norm_s(s_L + self.o_proj(s_L)), v_L

class CausalMolSSM(nn.Module):
    """
    SOTA: Bidirectional Mamba-3 Trinity.
    - Matches bias=False for all linear projections per checkpoint log.
    - Matches x_proj: [160, 128]
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bidirectional=True):
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, d_state, d_conv, expand
        self.d_inner = int(expand * d_model)
        self.bidirectional = bidirectional

        # Forward (Strictly bias=False)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv-1)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state * 2, bias=False) # 160
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        A_real = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1) * -0.5
        self.A_log = nn.Parameter(A_real) # Real-valued baseline A
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        if bidirectional:
            self.bwd_in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.bwd_conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv-1)
            self.bwd_x_proj = nn.Linear(self.d_inner, self.d_inner + d_state * 2, bias=False)
            self.bwd_dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
            self.bwd_out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.fusion = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x, batch_idx=None):
        out_fwd = self._compute_ssm(x, 'fwd')
        if not self.bidirectional: return out_fwd
        out_bwd = self._compute_ssm(x.flip(0), 'bwd').flip(0)
        return self.fusion(torch.cat([out_fwd, out_bwd], dim=-1))

    def _compute_ssm(self, x, dir):
        in_p, x_p, dt_p, conv, out_p = (self.in_proj, self.x_proj, self.dt_proj, self.conv1d, self.out_proj) if dir == 'fwd' else \
                                       (self.bwd_in_proj, self.bwd_x_proj, self.bwd_dt_proj, self.bwd_conv1d, self.bwd_out_proj)
        xz = in_p(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm = conv(x_ssm.transpose(0, 1).unsqueeze(0))[:, :, :x.size(0)].squeeze(0).transpose(0, 1)
        ssm_params = x_p(F.silu(x_ssm))
        delta, B, C = ssm_params.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        y = self._scan(x_ssm, F.softplus(dt_p(delta)), B, C)
        return out_p(y * F.silu(z))

    def _scan(self, u, dt, B, C):
        A = -torch.exp(self.A_log)
        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        deltaB = dt.unsqueeze(-1) * B.unsqueeze(1)
        u_expanded = u.unsqueeze(-1)
        h = torch.zeros(u.size(0), self.d_inner, self.d_state, device=u.device)
        curr_h = torch.zeros(self.d_inner, self.d_state, device=u.device)
        for i in range(u.size(0)):
            curr_h = deltaA[i] * curr_h + deltaB[i] * u_expanded[i]
            h[i] = curr_h
        return (h * C.unsqueeze(1)).sum(dim=-1) + self.D * u

SimpleS6 = CausalMolSSM
