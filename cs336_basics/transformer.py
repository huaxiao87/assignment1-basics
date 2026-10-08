import torch
import math
from jaxtyping import Float, Int
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device=None, dtype:torch.dtype=None):
        super().__init__()
        variance = 2 /(in_features + out_features)
        sigma = math.sqrt(variance)
        self.W: Float[torch.Tensor, "out_features in_features "] = torch.nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.W, 0, variance, -3*sigma, 3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device=None, dtype:torch.dtype=None):        
        super().__init__()
        self.embedding_matrix = torch.nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embedding_matrix, 0, 1, -3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[x]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g: Float[torch.Tensor, "d_model"] = torch.nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        torch.nn.init.constant_(self.g, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm_x = x / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return norm_x.to(in_dtype) * self.g

def silu( x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device:torch.device=None, dtype:torch.dtype=None):
        super().__init__()
        # # You should set d_ff to approximately 8/3×𝑑model in your implementation
        # d_ff = int(8/3 * d_model)
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pass