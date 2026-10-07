import torch
import math
from jaxtyping import Float, Int
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device=None, dtype:torch.dtype=None):
        super().__init__()
        variance = 2 /(in_features + out_features)
        sigma = math.sqrt(variance)
        self.W: Float[torch.Tensor, "out_features in_features "] = torch.zeros(out_features, in_features, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(self.W, 0, variance, -3*sigma, 3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")
