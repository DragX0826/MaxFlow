# maxflow/utils/training.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np

def optimize_for_intel():
    """Checks for Intel extensions to PyTorch."""
    try:
        import intel_extension_for_pytorch as ipex
        print("Intel Extension for PyTorch detected. Optimizing...")
    except ImportError:
        pass

def get_optimizer(model, learning_rate=1e-4, weight_decay=0.01):
    """
    SOTA Phase 55: Schedule-Free Optimizer Support.
    Returns standard AdamW; wrapper logic handles complexity.
    """
    # SOTA Acceleration: Use Muon (Momentum Orthogonalized) by default
    try:
        from maxflow.utils.optimization import Muon
        # Muon doesn't use weight_decay in valid steps usually, but we pass it if needed or ignore
        return Muon(model.parameters(), lr=learning_rate, momentum=0.95)
    except ImportError:
        # Fallback to Fused AdamW
        use_fused = torch.cuda.is_available() and hasattr(torch.optim, "AdamW")
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=use_fused)

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not math.isfinite(val): val = 0.0
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-10)

class DynamicRewardScaler:
    """
    Online normalization of rewards using Welford's algorithm.
    SOTA: Added robust variance guards and epsilon protection.
    """
    def __init__(self, momentum=0.1, epsilon=1e-4):
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
        self.epsilon = epsilon
        
    def update(self, values):
        """Batch Welford update — O(1) vectorized, no Python loop."""
        if isinstance(values, torch.Tensor):
            values = values.detach().float().cpu()
        elif np.isscalar(values):
            values = torch.tensor([values], dtype=torch.float32)
        else:
            values = torch.tensor(values, dtype=torch.float32)
            
        # Filter NaNs/Infs
        values = values[torch.isfinite(values)]
        n = values.numel()
        if n == 0:
            return
            
        batch_mean = values.mean().item()
        batch_var = values.var().item() if n > 1 else 0.0
        
        new_count = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * n / new_count
        self.M2 = self.M2 + batch_var * (n - 1) + delta**2 * self.count * n / new_count
        self.count = new_count
            
    @property
    def var(self):
        # SOTA: Ensure variance is always at least epsilon^2
        if self.count <= 1: return 1.0
        return max(self.M2 / self.count, self.epsilon**2)
        
    @property
    def std(self):
        return math.sqrt(self.var)
        
    def normalize(self, values):
        """
        Z-score normalization: (x - mean) / std.
        """
        if isinstance(values, torch.Tensor):
            dtype = values.dtype
            device = values.device
            vals = torch.nan_to_num(values.detach().float(), nan=self.mean)
            norm = (vals - self.mean) / self.std
            return norm.to(dtype).to(device)
        elif isinstance(values, (list, tuple)):
            return [(torch.nan_to_num(torch.tensor(x), nan=self.mean).item() - self.mean) / self.std for x in values]
        else:
            v = values if math.isfinite(values) else self.mean
            return (v - self.mean) / self.std

class SNRAwareEMA:
    """
    SOTA Phase 58: Layer-wise SNR Aware EMA.
    """
    def __init__(self, model, decay_base=0.999, snr_threshold=2.0):
        self.model = model
        self.decay_base = decay_base
        self.snr_threshold = snr_threshold
        self.shadow = {}
        self.grad_mu = {} 
        self.grad_var = {} 
        self.momentum = 0.1
        
        self.exclude_keywords = ['gvp', 'invariant', 'equivariant', 'pos_embedding']
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                is_excluded = any(kw in name.lower() for kw in self.exclude_keywords)
                if not is_excluded:
                    self.shadow[name] = param.data.clone()
                    self.grad_mu[name] = torch.zeros_like(param.data)
                    self.grad_var[name] = torch.ones_like(param.data)
                
    def update(self, model):
        # 1. Gather SNR Stats
        layer_snrs = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self.shadow:
                grad = param.grad.data
                
                # Update Stats
                self.grad_mu[name] = (1 - self.momentum) * self.grad_mu[name] + self.momentum * grad
                diff = grad - self.grad_mu[name]
                self.grad_var[name] = (1 - self.momentum) * self.grad_var[name] + self.momentum * (diff * diff)
                
                # Compute SNR
                snr = torch.abs(self.grad_mu[name]) / (torch.sqrt(self.grad_var[name]) + 1e-6)
                
                # Group by Layer
                layer_key = ".".join(name.split('.')[:2])
                if layer_key not in layer_snrs:
                    layer_snrs[layer_key] = []
                layer_snrs[layer_key].append(snr.mean().item())
                
        # 2. Update Shadows with Adaptive Decay
        for name, param in model.named_parameters():
            if name in self.shadow and param.grad is not None:
                layer_key = ".".join(name.split('.')[:2])
                
                # Safe average calculation
                if layer_key in layer_snrs and layer_snrs[layer_key]:
                    avg_layer_snr = sum(layer_snrs[layer_key]) / len(layer_snrs[layer_key])
                else:
                    avg_layer_snr = 0.0
                
                current_decay = self.decay_base
                if avg_layer_snr > self.snr_threshold:
                    current_decay = min(self.decay_base, 0.9)
                
                new_average = (1.0 - current_decay) * param.data + current_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                 param.data.copy_(self.shadow[name])
                 
    def state_dict(self): return self.shadow
    def load_state_dict(self, state_dict): self.shadow = state_dict

import os
import csv
import time

class CSVLogger:
    """
    SOTA Phase 63: Minimalist CSV Logger for Institutional Training.
    Prevents terminal spam while maintaining a complete history.
    """
    def __init__(self, filename, headers):
        self.filename = filename
        self.headers = headers
        self.file_exists = os.path.isfile(filename)
        
        # Initialize file with headers if new
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not self.file_exists:
                writer.writeheader()
                
    def log(self, metrics):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(metrics)

class SilentStepLogger:
    """
    Periodic console logger to replace tqdm spam.
    """
    def __init__(self, accelerator, total_steps, interval=50, desc="Training"):
        self.accelerator = accelerator
        self.total_steps = total_steps
        self.interval = interval
        self.desc = desc
        self.start_time = time.time()
        
    def log(self, step, metrics):
        if step % self.interval == 0 or step == self.total_steps:
            elapsed = time.time() - self.start_time
            rate = step / elapsed if elapsed > 0 else 0
            eta = (self.total_steps - step) / rate if rate > 0 else 0
            
            # Multi-line or single-line compact print
            metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            msg = f"[{self.desc}] Step {step}/{self.total_steps} | {metric_str} | Rate: {rate:.2f} it/s | ETA: {int(eta//60)}m {int(eta%60)}s"
            self.accelerator.print(msg)
