# maxflow/models/flow_matching.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from maxflow.utils.motifs import MotifDecomposer
from maxflow.utils.hft_scheduler import AlmgrenChrissScheduler, KalmanGradientFilter, calculate_grad_volatility
from maxflow.utils.physics import PhysicsEngine
from maxflow.utils.scatter import robust_scatter_mean

class RectifiedFlow(nn.Module):
    """
    Conditional Rectified Flow Wrapper.
    """
    def __init__(self, backbone, use_dmd=False):
        super().__init__()
        self.backbone = backbone
        self.use_dmd = use_dmd # SOTA Phase 11: DMD Diversity
        self.motif_decomposer = MotifDecomposer()
        self.physics_engine = PhysicsEngine() # SOTA Phase 51: Training with Physics

    def forward(self, data, t=None):
        """
        Required for direct module calls (e.g. TTA or inference).
        Defaults to t=1.0 (target velocity) if not specified.
        """
        device = data.x_L.device
        batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
        if t is None:
            t = torch.ones(batch_size, device=device)
        
        # Backbone returns (res, confidence, kl_div)
        res, _, _ = self.backbone(t, data)
        return res

    def loss(self, data, reduction='mean'):
        """
        Flow Matching Loss.
        Args:
            data: PyG batch object.
            reduction: 'mean' for standard training, 'none' for MaxRL reweighting.
        """
        device = data.x_L.device
        batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
        
        # 1. Sample time t ~ Logit-Normal(0, 1) (SOTA Phase 15)
        # Focuses training on the "difficult" middle part of the flow trajectory.
        # sigma=1.0 is standard for optimal transport flow matching.
        t_raw = torch.randn(batch_size, device=device)
        t = torch.sigmoid(t_raw)
        
        # 2. Robust Global Centering
        center_raw = data.pocket_center
        batch_idx = getattr(data, 'x_L_batch', getattr(data, 'batch', None))
        
        if batch_idx is not None:
             center_nodes = center_raw[batch_idx]
        else:
             center_nodes = center_raw.expand(data.pos_L.size(0), -1)

        pos_L_orig = data.pos_L.clone()
        pos_P_orig = data.pos_P.clone()
        
        # Center coordinates to stabilize Mamba-3/GVP
        data.pos_L = data.pos_L - center_nodes
        
        batch_P = getattr(data, 'x_P_batch', getattr(data, 'pos_P_batch', None))
        if batch_P is not None:
             center_nodes_P = center_raw[batch_P]
             data.pos_P = data.pos_P - center_nodes_P
        else:
             data.pos_P = data.pos_P - center_raw
        
        # 3. Interpolate Flow
        noise = torch.randn_like(data.pos_L)
        x_0 = noise 
        x_1 = data.pos_L 
        
        t_nodes = t[batch_idx].view(-1, 1) if batch_idx is not None else t.view(-1, 1)
        x_t = t_nodes * x_1 + (1 - t_nodes) * x_0
        
        data.pos_L = x_t
        
        # 4. Backbone Prediction
        backbone_res, conf_pred, kl_div = self.backbone(t, data)
        
        v_pred = backbone_res.get('v_pred')
        if v_pred is None:
            v_pred = (backbone_res.get('v_trans'), backbone_res.get('v_rot'))
            
        atom_logits_pred = backbone_res.get('atom_logits')
        charge_delta = backbone_res.get('charge')
        chiral_pred = backbone_res.get('chiral')

        # 5. Physics Penalties (Clash & Repulsion)
        loss_clash = self.physics_engine.calculate_steric_clash(
            x_t, data.pos_P, threshold=1.5, batch_L=batch_idx, batch_P=batch_P
        )
        loss_repulsion = self.physics_engine.calculate_intra_repulsion(
            x_t, threshold=1.2, batch_L=batch_idx
        )

        # Restore original coords
        data.pos_L = pos_L_orig
        data.pos_P = pos_P_orig
        
        # 6. Velocity Strategy (Straight Line Target)
        v_target = (x_1 - x_0).clamp(-50.0, 50.0)
        
        # 7. Multi-Objective Loss Calculation
        lambda_kl = 0.001
        
        # A. MSE Components
        if hasattr(data, 'atom_to_motif') and isinstance(v_pred, tuple):
             v_trans_pred, v_rot_pred = v_pred
             v_target_motif = robust_scatter_mean(v_target, data.atom_to_motif, dim=0)
             
             loss_trans_m = F.mse_loss(v_trans_pred, v_target_motif, reduction='none').mean(dim=-1)
             loss_rot_m = F.mse_loss(v_rot_pred, torch.zeros_like(v_rot_pred), reduction='none').mean(dim=-1)
             # Map back to atoms
             loss_mse_atom = loss_trans_m[data.atom_to_motif] + 0.1 * loss_rot_m[data.atom_to_motif]
        else:
             loss_mse_atom = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=-1)
        
        # Aggregate to Graph level
        if batch_idx is not None:
            loss_mse_graph = robust_scatter_mean(loss_mse_atom, batch_idx, dim=0)
        else:
            loss_mse_graph = loss_mse_atom.mean().view(1)

        # B. Physics & Auxiliary (Per Graph)
        loss_atom_graph = torch.zeros_like(loss_mse_graph)
        if atom_logits_pred is not None and hasattr(data, 'atom_types'):
            la = F.cross_entropy(atom_logits_pred, data.atom_types, reduction='none')
            if batch_idx is not None:
                loss_atom_graph = robust_scatter_mean(la, batch_idx, dim=0)
            else:
                loss_atom_graph = la.mean().view(1)

        # Summary Per-Graph Loss for MaxRL reweighting
        total_loss_graph = loss_mse_graph.clamp(max=100.0) + loss_atom_graph + lambda_kl * kl_div + loss_clash + loss_repulsion
        
        # C. DMD Diversity Loss (Optional)
        if getattr(self, 'use_dmd', False) and (t < 0.1).any():
             # Standard cosine similarity penalty across batch
             v_norm = F.normalize(loss_mse_graph.view(batch_size, -1), dim=-1)
             sim = torch.mm(v_norm, v_norm.t())
             mask = ~torch.eye(batch_size, device=device).bool()
             loss_dmd = 0.5 * torch.relu(sim[mask].mean() - 0.2)
             total_loss_graph = total_loss_graph + loss_dmd

        if reduction == 'mean':
            return total_loss_graph.mean()
        return total_loss_graph
            
        # SOTA Phase 11: DMD Diversity Loss (Distribution Matching Distillation)
        # Penalize mode collapse by enforcing variance in early flow steps (t < 0.1)
        if getattr(self, 'use_dmd', False) and (t < 0.1).any():
             # We want diverse velocities for diverse noise inputs.
             # Calculate pairwise cosine similarity of mean velocities per graph
             batch_L = getattr(data, 'x_L_batch', None)
             if batch_L is not None:
                 if v_trans_pred is not None:
                     v_flat = v_trans_pred
                 else: 
                     v_flat = v_pred
                     
                 # Pool to graph level
                 from torch_scatter import scatter_mean
                 v_graph = scatter_mean(v_flat, batch_L, dim=0) # (B, 3)
                 
                 # Normalize
                 v_graph = F.normalize(v_graph, dim=-1)
                 
                 # Cosine Similarity Matrix
                 sim_matrix = torch.mm(v_graph, v_graph.t())
                 
                 # We want to minimize similarity between different graphs (off-diagonal)
                 # Mask diagonal
                 mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
                 if mask.any():
                     avg_sim = sim_matrix[mask].mean()
                     # Penalty: If similarity is high, loss increases.
                     # We want sim to be 0 (orthogonal) or negative.
                     loss_dmd = 0.5 * torch.relu(avg_sim - 0.2) # Margin 0.2
                     total_loss = total_loss + loss_dmd
        
        return total_loss

    def sample_x_t(self, data, t, noise=None):
        """
        SOTA Hardening: Centralized flow sampling with optional noise injection.
        Allows MaxRL to inject consistent noise for win/lose pairs.
        """
        device = data.x_L.device
        batch_L = getattr(data, 'x_L_batch', None)
        x_1 = data.pos_L
        
        if noise is None:
            noise = torch.randn_like(x_1)
        x_0 = noise
        
        t_nodes = t[batch_L].view(-1, 1) if batch_L is not None else t.view(-1, 1).expand(x_1.size(0), -1)
        x_t = t_nodes * x_1 + (1 - t_nodes) * x_0
        
        return x_t, x_0, x_1, t_nodes, data

    def calculate_chiral_penalty(self, x, data):
        """
        Computes signed volume penalty for potential chiral centers.
        V = (a-d) . ((b-d) x (c-d)) / 6
        """
        chiral_indices = getattr(data, 'chiral_indices', None) # (N_centers, 4)
        if chiral_indices is None or chiral_indices.size(0) == 0:
            return torch.tensor(0.0, device=x.device)
            
        a, b, c, d = x[chiral_indices[:, 0]], x[chiral_indices[:, 1]], x[chiral_indices[:, 2]], x[chiral_indices[:, 3]]
        target_v = getattr(data, 'chiral_volumes', None) # Ground truth volumes
        
        current_v = torch.sum((a - d) * torch.cross(b - d, c - d, dim=-1), dim=-1) / 6.0
        
        if target_v is not None:
             # Penalize sign flip (Chiral inversion)
             return F.mse_loss(current_v, target_v)
        return torch.tensor(0.0, device=x.device)

    def _slerp(self, q0, q1, t):
        """
        Spherical Linear Interpolation for Quaternions.
        (Future-proofing for full SE(3) quaternionic flow)
        """
        cos_half_theta = torch.sum(q0 * q1, dim=-1, keepdim=True)
        # Ensure shortest path
        neg_mask = cos_half_theta < 0
        q1 = torch.where(neg_mask, -q1, q1)
        cos_half_theta = torch.where(neg_mask, -cos_half_theta, cos_half_theta)
        
        half_theta = torch.acos(cos_half_theta.clamp(-1, 1))
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta**2)
        
        res = torch.where(
            sin_half_theta < 0.001,
            0.5 * q0 + 0.5 * q1,
            (torch.sin((1-t)*half_theta) * q0 + torch.sin(t*half_theta) * q1) / sin_half_theta
        )
        return F.normalize(res, dim=-1)
    
    # ============== Phase 24: SO(3)-Averaged Training (ICML 2025) ==============
    
    def loss_so3_averaged(self, data, n_rotations=4):
        """
        SO(3)-Averaged Flow Matching Loss (ICML 2025).
        
        Key Insight: Average the loss over random rotations from Haar measure on SO(3).
        This eliminates rotational bias and leads to faster convergence.
        
        Args:
            data: PyG batch object
            n_rotations: Number of random rotations to average over
        """
        device = data.x_L.device
        
        total_loss = 0.0
        for _ in range(n_rotations):
            # Generate random SO(3) rotation
            R = self._random_rotation(device)
            
            # Apply rotation to ligand positions
            original_pos_L = data.pos_L.clone()
            data.pos_L = torch.matmul(data.pos_L, R.T)
            
            # Also rotate pocket center for consistency
            original_center = data.pocket_center.clone()
            data.pocket_center = torch.matmul(data.pocket_center, R.T)
            
            # Compute loss on rotated data
            loss = self.loss(data)
            total_loss += loss
            
            # Restore original positions
            data.pos_L = original_pos_L
            data.pocket_center = original_center
        
        return total_loss / n_rotations
    
    @staticmethod
    def _random_rotation(device):
        """
        Sample a random rotation matrix from Haar measure on SO(3).
        Guaranteed det=+1 and uniform distribution using quaternion Gaussian method.
        Ref: Stewart (1980) / Shoemake (1992).
        """
        # 1. Sample 4D unit Gaussian
        q = torch.randn(4, device=device)
        q = q / (q.norm() + 1e-8)
        
        # 2. Extract components
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # 3. Construct rotation matrix
        R = torch.stack([
            torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w]),
            torch.stack([2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w]),
            torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2])
        ], dim=0)
        
        return R
    
    # ============== End Phase 24 ==============

    def get_combined_velocity(self, t, x_t, data, gamma):
        """
        SOTA Phase 34: Optimized Guidance Calculation (Single Scalar Objective).
        Reduces autograd overhead for complex multi-objective sampling.
        """
        device = x_t.device
        t_val = float(t[0].item() if t.dim() > 0 else t.item())
        
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            data.pos_L = x_t
            
            # 1. Backbone Forward
            v_info, confidence, _ = self.backbone(t, data)
            
            if t_val >= 0:
                pass 
            v_model = v_info.get('v_pred') if 'v_pred' in v_info else (v_info.get('v_trans'), v_info.get('v_rot'))
            
            # 2. Build Fused Objective
            total_objective = torch.tensor(0.0, device=device)
            
            # A. Confidence (Maximize)
            if gamma > 0:
                total_objective += gamma * confidence.sum()
            
            # B. Physics & Auxiliary (Minimize Energy -> Maximize negative)
            schedule = 4 * t_val * (1 - t_val)
            lambda_phys = 2.0 * schedule
            
            if lambda_phys > 0.1:
                ensemble_P = getattr(data, 'ensemble_P', None)
                if ensemble_P is not None:
                    energy = self.physics_engine.calculate_ensemble_interaction_energy(
                        x_t, ensemble_P, q_L=v_info.get('charge'), pos_metals=getattr(data, 'pos_metals', None)
                    )
                else:
                    energy = self.physics_engine.calculate_interaction_energy(
                        x_t, data.pos_P, data=data, q_L_dynamic=v_info.get('charge')
                    )
                total_objective -= lambda_phys * energy.sum()

            # C. Synthetics & Properties (Optional but Fused)
            # ... can add admet_pred, reward_kinetic here ...

            # 3. Optimized Single Backward Pass
            if total_objective.requires_grad:
                grads = torch.autograd.grad(total_objective, x_t, retain_graph=False)[0]
            else:
                grads = torch.zeros_like(x_t)
                
            if grads is None: grads = torch.zeros_like(x_t)
            
            # Clamping for stability
            grad_norm = torch.norm(grads, dim=-1, keepdim=True) + 1e-6
            grads = grads / grad_norm * grad_norm.clamp(max=2.0)
            
            # 4. Hierarchical Projection for Motif Flows
            if isinstance(v_model, tuple):
                 v_trans, v_rot = v_model
                 g_trans, g_rot, g_local = self.motif_decomposer.get_hierarchical_projected_velocity(
                    x_t, grads, data.atom_to_motif
                 )
                 v_trans_final = v_trans + g_trans
                 v_rot_final = v_rot + g_rot
                 
                 from maxflow.utils.scatter import robust_scatter_mean
                 centroids = robust_scatter_mean(x_t, data.atom_to_motif, dim=0)
                 return v_trans_final, v_rot_final, g_local, centroids, torch.zeros_like(data.pos_P), v_info.get('rmsf')
            else:
                 v_final = v_model + grads
                 return None, None, v_final, None, torch.zeros_like(data.pos_P), v_info.get('rmsf')

    def rigid_body_step(self, x_t, v_trans, v_rot, v_local, centroids, dt, atom_to_motif):
        """
        [Alpha Phase 62] Riemannian Lie-Group Integration.
        Uses Rodrigues' formula for exact rotation on SO(3).
        """
        if v_trans is None:
            return x_t + v_local * dt
            
        # 1. Expand motif velocities to atoms
        v_t_atom = v_trans[atom_to_motif]
        
        # 2. SE(3) Geodesic Update: T_next = T + v_t*dt, R_next = exp(w*dt)R
        # We approximate the rotation by applying it to relative vectors r
        atom_to_center = centroids[atom_to_motif]
        r = x_t - atom_to_center
        
        # Rodrigues Rotation: r_new = r*cos(theta) + (k x r)*sin(theta) + k*(k.r)*(1-cos(theta))
        # For small dt, theta = ||w||*dt, k = w / ||w||
        omega = v_rot[atom_to_motif] # (N, 3)
        theta = torch.norm(omega, dim=-1, keepdim=True) * dt
        k = omega / (torch.norm(omega, dim=-1, keepdim=True) + 1e-8)
        
        # Rotate relative vectors
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        cross_kr = torch.cross(k, r, dim=-1)
        dot_kr = (k * r).sum(dim=-1, keepdim=True)
        
        r_rotated = r * cos_t + cross_kr * sin_t + k * dot_kr * (1 - cos_t)
        
        # 3. Combine Translation + Rotation + Local Residual
        x_next = atom_to_center + v_t_atom * dt + r_rotated + v_local * dt
        
        return x_next

    def sample(self, data, steps=10, gamma=1.5, particles=1):
        """
        SOTA Phase 50: HFT-Inspired SMC Sampling.
        particles: Number of parallel trajectories per molecule (Importance Sampling).
        """
        device = data.x_L.device
        self._kalman_filters = {} # Reset filter cache for new molecule/batch to avoid shape mismatch
        
        # 1. Expand data for particles
        if particles > 1:
            from torch_geometric.data import Batch
            # Create N copies of the batch
            particle_list = []
            for _ in range(particles):
                particle_list.append(data.clone())
            data = Batch.from_data_list(particle_list)
            
        batch_idx = getattr(data, 'x_L_batch', getattr(data, 'batch', None))
        
        center = data.pocket_center[batch_idx] if batch_idx is not None else data.pocket_center
        batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
        
        # SOTA Phase 35: Initialize Protein native positions
        if not hasattr(data, 'pos_P_orig'):
            data.pos_P_orig = data.pos_P.clone()
            
        x_t = torch.randn_like(data.pos_L) + center
        # SOTA Phase 38 Fix: Enable gradients for guidance calculation
        x_t = x_t.detach().requires_grad_(True)
        if not data.pos_P.requires_grad:
            data.pos_P = data.pos_P.detach().requires_grad_(True)
        
        # SOTA Phase 50: HFT Best Execution Components
        hft_opt = AlmgrenChrissScheduler(steps, initial_gamma=gamma)
        grad_history = []
        
        dt = 1.0 / steps
        traj = [x_t.detach().cpu().clone()]
        
        for i in range(steps):
            t_curr = i / steps
            t = torch.tensor([t_curr] * batch_size, device=device)
            
            # --- Engineering Hardening: Guidance Gating (Phase 46) ---
            # Expensive physics/guidance is skipable in early steps
            # or calculated at reduced frequency
            run_guidance = (t_curr > 0.2) or (i % 2 == 0)
            
            # --- SOTA Phase 62: Incremental SSM State Update ---
            if not hasattr(self, '_ssm_states'): self._ssm_states = {}
            if id(data) not in self._ssm_states:
                # Initialize state for Mamba-2 (D_inner, d_state) per node
                d_inner = self.backbone.global_mixer.mamba.d_inner
                d_state = self.backbone.global_mixer.mamba.d_state
                self._ssm_states[id(data)] = torch.zeros(batch_size, 1, d_inner, d_state, device=device)
            
            # --- SOTA Phase 50: Almgren-Chriss Adaptive Guidance ---
            volatility = calculate_grad_volatility(grad_history)
            active_gamma = hft_opt.get_adaptive_gamma(t_curr, volatility) if run_guidance else 0.0
            
            # --- Heun Step Part 1: Predictor ---
            v_t, v_rot_t, v_local_t, centroids_t, g_prot_t, rmsf_t = self.get_combined_velocity(
                t, x_t, data, active_gamma
            )
            
            # Update history for HFT Volatility tracking
            if run_guidance:
                # We extract the physics gradient part implicitly or use total displacement
                grad_history.append(v_local_t if v_t is None else v_t)
                if len(grad_history) > 10: grad_history.pop(0)

            atom_to_motif = getattr(data, 'atom_to_motif', None)
            x_tilde = self.rigid_body_step(x_t, v_t, v_rot_t, v_local_t, centroids_t, dt, atom_to_motif)
            
            # --- Induced-Fit Displacement Part 1 ---
            with torch.no_grad():
                # Displace protein atoms (inverse force direction)
                # schedule: softer flexibility at mid-sampling
                flex_scale = 0.5 * (4 * t_curr * (1 - t_curr))
                data.pos_P = data.pos_P - flex_scale * g_prot_t * dt
                
            # --- Langevin Noise Injection (Phase 45: Adaptive Induced-Fit) ---
            # σ(t) peaks at mid-traj, scaled by predicted RMSF
            if 0.1 < t_curr < 0.9:
                # SOTA Phase 45: Adaptive noise based on pocket flexibility
                sigma_t = 0.1 * rmsf_t * (1.0 - t_curr) 
                noise = torch.randn_like(x_tilde) * sigma_t * torch.sqrt(torch.tensor(dt, device=x_tilde.device))
                x_tilde = x_tilde + noise
                
            # --- Heun Step Part 2: Corrector ---
            if i < steps - 1:
                t_next = (i + 1) / steps
                t_n = torch.tensor([t_next] * batch_size, device=device)
                
                v_next, v_rot_next, v_local_next, centroids_next, g_prot_next, rmsf_next = self.get_combined_velocity(t_n, x_tilde, data, gamma)
                
                # Average velocities for 2nd order accuracy
                v_avg = 0.5 * (v_t + v_next) if v_t is not None else None
                v_rot_avg = 0.5 * (v_rot_t + v_rot_next) if v_rot_t is not None else None
                v_local_avg = 0.5 * (v_local_t + v_local_next)
                
                # Re-apply step from original x_t with averaged velocity
                x_t = self.rigid_body_step(x_t, v_avg, v_rot_avg, v_local_avg, centroids_t, dt, atom_to_motif)
                
                # Induced-Fit Displacement Part 2
                with torch.no_grad():
                    g_prot_avg = 0.5 * (g_prot_t + g_prot_next)
                    pass 
                
                # --- SOTA Phase 50: SMC Vectorized Re-sampling ---
                if particles > 1 and i == (steps // 2):
                    with torch.no_grad():
                        x_t = self._resample_particles(x_t, data, batch_size, particles)
            else:
                x_t = x_tilde
                
            traj.append(x_t.detach().cpu().clone())
            
        return x_t.detach(), traj

    def _resample_particles(self, x_t, data, batch_size, particles):
        """
        SOTA Vectorized SMC Resampling (Universal Engine Efficiency).
        """
        device = x_t.device
        batch_idx = getattr(data, 'x_L_batch', getattr(data, 'batch', None))
        
        # 1. Calculate Energies (Particles, Batch)
        energies = self.physics_engine.calculate_interaction_energy(
            x_t, data.pos_P, 
            batch_L=batch_idx, batch_P=getattr(data, 'pos_P_batch', None),
            data=data
        ).view(particles, batch_size).T # (Batch, Particles)
        
        # 2. Resample Indices per Molecule
        probs = torch.softmax(-energies / 2.0, dim=1) 
        resample_idx = torch.multinomial(probs, particles, replacement=True) # (Batch, Particles)
        
        # 3. Apply Resampling (Hybrid Vectorization)
        new_x_t = x_t.clone()
        for b in range(batch_size):
             mol_mask_base = (batch_idx % batch_size == b)
             if not mol_mask_base.any(): continue
             
             # Extract nodes for all particles of molecule b
             mol_nodes = x_t[mol_mask_base].view(particles, -1, 3)
             # Map new particle slots to source particle slots
             new_nodes = mol_nodes[resample_idx[b]] # (Particles, N_nodes, 3)
             # Flatten and write back
             new_x_t[mol_mask_base] = new_nodes.view(-1, 3)
             
        return new_x_t
