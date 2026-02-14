
import torch
import torch.nn as nn

# --- 1. Muon Optimizer (The 2025 Standard for Speed) ---
class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized Optimizer.
    Converges faster than AdamW for high-dimensional generative models.
    Reference: https://github.com/KellerJordan/Muon (Adapted for PyTorch)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None: continue
                
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Newton-Schulz Iteration for Orthogonalization
                # X_{k+1} = 1.5 * X_k - 0.5 * X_k * (X_k^T * X_k) or 1.5 X - 0.5 (X X^T) X
                if g.dim() > 1:
                    X = g.view(g.size(0), -1)
                    m, n = X.shape
                    if m > n:
                        # Orthogonalize columns
                        for _ in range(ns_steps):
                            M = X.t() @ X
                            X.mul_(1.5).addmm_(X, M, alpha=-0.5)
                    else:
                        # Orthogonalize rows
                        for _ in range(ns_steps):
                            M = X @ X.t()
                            X.mul_(1.5).addmm_(M, X, alpha=-0.5)
                    g = X.view_as(g)
                
                # Update weights
                p.add_(g, alpha=-lr)

# --- 2. GRPO-MaxRL Loss (Memory Efficient RL) ---
def compute_grpo_maxrl_loss(log_probs, rewards):
    """
    Combined GRPO + MaxRL Objective.
    Eliminates the need for a Critic model by using batch averages.
    
    Args:
        log_probs: (B, ) Log probabilities of the generated samples.
        rewards: (B, ) Rewards for each sample.
    """
    # 1. Compute Baseline (Group Mean)
    # If using multiple GPUs, this should sync across devices. 
    # For single GPU, it's just batch mean.
    baseline = rewards.mean()
    
    # 2. Advantage / Importance Weight
    # GRPO normalizes advantages: A = (r - mean) / std
    # MaxRL uses Ratio: W = r / mean
    
    # Hybrid SOTA approach:
    # Use Normalized Advantage for stability, but reweight by MaxRL logic for hard exploration
    
    # Simple MaxRL implementation with Group Baseline:
    weights = rewards / (baseline + 1e-6)
    
    # Clip for stability (prevents exploding gradients on lucky samples)
    weights = torch.clamp(weights, min=0.0, max=5.0)
    
    # 3. Policy Gradient
    # Loss = - E [ Weight * log_prob ]
    loss = -torch.mean(weights.detach() * log_probs)
    
    return loss, baseline.item()
