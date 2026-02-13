# max_flow/models/backbone.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from max_flow.models.layers import GVPCrossAttention, SimpleS6 
from max_flow.utils.constants import NUM_ATOM_TYPES
from max_flow.utils.scatter import robust_scatter_mean

class GlobalContextBlock(nn.Module):
    """
    Wrapper for Global Context mixing using Mamba-3 (Selective Scan).
    Provides O(N) linear-time global dependency modeling.
    """
    def __init__(self, d_model):
        super().__init__()
        # SOTA Fix: Align d_state=16 and bidirectional mapping with maxflow_pretrained.pt
        self.mamba = SimpleS6(d_model, d_state=16) 
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, batch_idx=None):
        # Mamba-3 handles batch-aware padding/unpadding internally via batch_idx
        x_out = self.mamba(x, batch_idx=batch_idx)
        return self.norm(x + x_out)

class GVPEncoder(nn.Module):
    """
    Enhanced GVP-like Encoder using distance-based edge features.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        # Edge dim is 1 (distance)
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, edge_dim=1)) 
        
        self.s_emb = nn.Linear(in_channels, hidden_channels)
        
        # Learnable Fractional Filter (SOTA 2.3)
        from max_flow.utils.fractional_ops import LearnableFractionalFilter
        self.frac_filter = LearnableFractionalFilter(hidden_channels, window=5)
        
    def forward(self, x, edge_index, pos):
        s = self.s_emb(x).clamp(-10.0, 10.0)
        
        # 1. Compute Distances (Stable)
        row, col = edge_index
        diff = pos[row] - pos[col]
        dist = torch.sqrt(torch.sum(diff**2, dim=-1, keepdim=True) + 1e-12)
        
        # 2. Fractional Filtering
        num_nodes = s.size(0)
        num_edges = edge_index.size(1)
        K = num_edges // num_nodes if num_nodes > 0 and num_edges % num_nodes == 0 else 0
        
        if K > 0 and K >= self.frac_filter.window:
             dist_reshaped = dist.view(num_nodes, K, 1)
             dist_filtered, _ = self.frac_filter(dist_reshaped[:, :self.frac_filter.window], s)
             dist = dist + dist_filtered.repeat_interleave(K, dim=0).view(-1, 1)

        # 3. GVP Convolution
        for conv in self.convs:
            s_out = conv(s, edge_index, edge_attr=dist)
            s = (s + s_out).clamp(-20.0, 20.0)
            s = torch.relu(s)
        
        v = torch.zeros(s.size(0), 16, 3, device=x.device)
        return s, v

class MotifPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.s_pool = nn.Linear(channels, channels)
        self.v_pool = nn.Linear(16, 16)

    def forward(self, s, v, motif_batch):
        motif_batch = motif_batch.view(-1)
        s_motif = robust_scatter_mean(s, motif_batch, dim=0)
        v_motif = robust_scatter_mean(v, motif_batch, dim=0) 
        return self.s_pool(s_motif), self.v_pool(v_motif.transpose(1, 2)).transpose(1, 2)

# --- Auxiliary Heads ---
class WaterHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, s): return self.mlp(s)

class ChargeHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2, 1), nn.Tanh())
    def forward(self, s): return self.mlp(s)

class FlexibilityHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2, 1), nn.Softplus())
    def forward(self, s): return self.mlp(s)

class ChiralAwarenessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2, 1), nn.Tanh())
    def forward(self, s): return self.mlp(s)

class ConceptHead(nn.Module):
    def __init__(self, hidden_dim, num_concepts=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_concepts))
    def forward(self, s): return self.mlp(s)

# --- Main Backbone ---
class CrossGVP(nn.Module):
    """
    MaxFlow Engine Backbone: SE(3)-Equivariant GVP + Mamba-3 Trinity.
    """
    def __init__(self, node_in_dim=167, hidden_dim=64, num_layers=3, num_atom_types=NUM_ATOM_TYPES):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.ligand_encoder = GVPEncoder(node_in_dim, hidden_dim, num_layers)
        self.protein_encoder = GVPEncoder(21, hidden_dim, num_layers) 
        
        # Global Context (Mamba-3)
        self.global_mixer = GlobalContextBlock(hidden_dim)
        
        # Cross Attention
        self.cross_layers = nn.ModuleList([
            GVPCrossAttention(hidden_dim, 16) for _ in range(num_layers)
        ])
        
        # Output Heads
        self.final_s = nn.Linear(hidden_dim, hidden_dim)
        self.final_v = nn.Linear(16, 1) # Trans Velocity
        self.motif_pooling = MotifPooling(hidden_dim)
        self.omega_v = nn.Linear(16, 1) # Rot Velocity
        self.atom_head = nn.Linear(hidden_dim, NUM_ATOM_TYPES) 
        self.conf_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        
        # Aux Heads
        self.water_head = WaterHead(hidden_dim)
        self.charge_head = ChargeHead(hidden_dim)
        self.flex_head = FlexibilityHead(hidden_dim)
        self.admet_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.chiral_head = ChiralAwarenessHead(hidden_dim)
        self.concept_head = ConceptHead(hidden_dim)
        
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.center_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        
        # VIB
        self.vib_mean = nn.Linear(hidden_dim, hidden_dim)
        self.vib_logvar = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t, data, return_latent=False, aux_tasks=True):
        device = data.x_L.device
        t_emb = self.time_mlp(t.view(-1, 1)) 
        
        # 1. Encode Ligand
        from max_flow.utils.geometry import radius_graph
        edge_index_L = radius_graph(data.pos_L, r=5.0, batch=data.x_L_batch if hasattr(data, 'x_L_batch') else None)
        s_L, v_L = self.ligand_encoder(data.x_L, edge_index_L, data.pos_L)
        
        # 2. Encode Protein
        edge_index_P = radius_graph(data.pos_P, r=5.0, batch=data.x_P_batch if hasattr(data, 'x_P_batch') else None)
        s_P, v_P = self.protein_encoder(data.x_P, edge_index_P, data.pos_P)
        
        # 3. Time & Anchor Embedding
        batch_idx = getattr(data, 'x_L_batch', getattr(data, 'batch', None))
        if batch_idx is not None:
            t_nodes = t_emb[batch_idx]
            center = data.pocket_center[batch_idx]
        else:
            t_nodes = t_emb.expand(s_L.size(0), -1)
            center = data.pocket_center.expand(s_L.size(0), -1)
            
        s_L = s_L + t_nodes
        dist_to_center = torch.norm(data.pos_L - center, dim=-1, keepdim=True)
        s_L = s_L + self.center_proj(dist_to_center)

        # 4. Global Context (Mamba-3)
        s_L = self.global_mixer(s_L, batch_idx=batch_idx)
        
        # 5. VIB (Information Bottleneck)
        if batch_idx is not None:
            batch_idx = batch_idx.view(-1)
            num_graphs = batch_idx.max().item() + 1
            s_L_sum = torch.zeros(num_graphs, s_L.size(-1), device=s_L.device)
            s_L_sum.index_add_(0, batch_idx, s_L)
            count = torch.zeros(num_graphs, 1, device=s_L.device)
            count.index_add_(0, batch_idx, torch.ones(s_L.size(0), 1, device=s_L.device))
            s_L_global = (s_L_sum / count.clamp(min=1e-6)).clamp(-50.0, 50.0)
        else:
            num_graphs = 1
            s_L_global = s_L.mean(dim=0, keepdim=True).clamp(-50.0, 50.0)
            
        mu = self.vib_mean(s_L_global).clamp(-10.0, 10.0)
        logvar = self.vib_logvar(s_L_global).clamp(-10, 10)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*logvar) if self.training else mu
        
        # Protein Global Latent for alignment
        if hasattr(data, 'x_P_batch'):
            s_P_sum = torch.zeros(num_graphs, s_P.size(-1), device=s_P.device)
            s_P_sum.index_add_(0, data.x_P_batch, s_P)
            count_P = torch.zeros(num_graphs, 1, device=s_P.device)
            count_P.index_add_(0, data.x_P_batch, torch.ones(s_P.size(0), 1, device=s_P.device))
            s_P_global = (s_P_sum / count_P.clamp(min=1e-6)).clamp(-50.0, 50.0)
        else:
            s_P_global = s_P.mean(dim=0, keepdim=True).repeat(num_graphs, 1)

        s_L = s_L + (z[batch_idx] if batch_idx is not None else z)

        # 7. Cross Interaction
        pos_P_ref = data.pos_P
        batch_P_ref = getattr(data, 'x_P_batch', None)
        for layer in self.cross_layers:
             s_L, v_L = layer(s_L, v_L, data.pos_L, s_P, v_P, pos_P_ref, batch_idx, batch_P_ref)

        # 8. Confidence Score
        confidence = self.conf_head(s_L_global).squeeze(-1)

        # 9. Output Logic
        if hasattr(data, 'atom_to_motif'):
            s_motif, v_motif = self.motif_pooling(s_L, v_L, data.atom_to_motif)
            v_trans = self.final_v(v_motif.transpose(1, 2)).squeeze(-1)
            v_rot = self.omega_v(v_motif.transpose(1, 2)).squeeze(-1)
            res = {'v_trans': v_trans, 'v_rot': v_rot}
        else:
            atom_vel = self.final_v(v_L.transpose(1, 2)).squeeze(-1)
            res = {'v_pred': atom_vel}
            
        # 10. Aux Tasks (Optional for speed)
        if aux_tasks:
            res['p_water'] = torch.nan_to_num(self.water_head(s_L))
            res['admet'] = torch.nan_to_num(self.admet_head(s_L_global)) # Global
            res['charge'] = torch.nan_to_num(self.charge_head(s_L))
            res['rmsf'] = torch.nan_to_num(self.flex_head(s_L))
            res['chiral'] = torch.nan_to_num(self.chiral_head(s_L))
            res['concept'] = torch.nan_to_num(self.concept_head(s_L_global)) # Global context
            res['atom_logits'] = torch.nan_to_num(self.atom_head(s_L))

        if return_latent:
             return res, confidence, kl_div, (s_L_global, s_P_global)
        return res, confidence, kl_div
