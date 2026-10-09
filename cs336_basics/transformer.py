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
        # You should set d_ff to approximately 8/3×𝑑model in your implementation
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        positions = torch.arange(0, max_seq_len, device=device)
        freqs = 1 / theta ** ((2 * torch.arange(0, d_k/2, device=device)) / d_k)
        angles = positions.outer(freqs)
        self.register_buffer("sin_cached", torch.sin(angles))
        self.register_buffer("cos_cached", torch.cos(angles))
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin_cached[token_positions]
        cos = self.cos_cached[token_positions]
        rearranged_x = rearrange(x, "... (pairs two) -> ... pairs two", two=2)
        # rotate each pair 
        # build a batch of 2x2 rotation matrices for each pair
        rotation_matrices = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
        # Rotate each pair
        rotated_x = einsum(rearranged_x, rotation_matrices, "... pairs j, ... pairs i j -> ... pairs i")
        # unrearrange
        return rearrange(rotated_x, "... pairs two -> ... (pairs two)")

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # subtracting the maximum value in the 𝑖-th dimension from all elements of the 𝑖-th dimension to avoid numerical stability issues
    x = x - x.max(dim=dim, keepdim=True).values
    # apply softmax to the 𝑖-th dimension of the input tensor.
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    d_k = Q.shape[-1]   
    scores = einsum(Q, K, "... Q d_k, ... K d_k -> ... Q K") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -float('inf'))
    return softmax(scores, dim=-1) @ V

class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device:torch.device=None, dtype:torch.dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.O = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build the boolean mask for the causal self-attention
        mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=0)
        # Split the input into multiple heads using einops
        Q = rearrange(self.Q(x), "... seq_len (num_heads head_dim)-> ... num_heads seq_len head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        K = rearrange(self.K(x), "... seq_len (num_heads head_dim)-> ... num_heads seq_len head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        V = rearrange(self.V(x), "... seq_len (num_heads head_dim)-> ... num_heads seq_len head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        return self.O(rearrange(scaled_dot_product_attention(Q, K, V, mask=mask), "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)", num_heads=self.num_heads, head_dim=self.head_dim))
