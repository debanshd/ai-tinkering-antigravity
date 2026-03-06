import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"

class BipolarSAE(nn.Module):
    def __init__(self, d_in=768, d_out=4096, k=32):
        super().__init__()
        self.enc = nn.Linear(d_in, d_out)
        self.dec = nn.Linear(d_out, d_in)
        self.k = k
        
    def forward(self, x):
        pre_acts = self.enc(x)
        # AbsTopK explicitly maintains the sign while enforcing sparsity
        abs_acts = torch.abs(pre_acts)
        val, idx = torch.topk(abs_acts, self.k, dim=-1)
        
        # Create mask of top-k absolute values
        mask = torch.zeros_like(pre_acts).scatter(-1, idx, 1.0)
        f = pre_acts * mask  # Preserves polarity (+ or -)
        return self.dec(f), f

class LinearController(nn.Module):
    def __init__(self, in_features=128):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)
        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x):
        return torch.clamp(self.linear(x), -2.0, 2.0), None
