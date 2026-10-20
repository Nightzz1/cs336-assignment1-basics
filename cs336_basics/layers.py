"""Basic trainable layers and the position-wise feed-forward network."""

import torch

from .nn_utils import silu


class LinearLayer(torch.nn.Module):
    def __init__(self, in_features: int, 
                       out_features: int, 
                       device: torch.device | None = None, 
                       dtype: torch.dtype | None = None):
        super().__init__()
        std_dev = (2 / (in_features + out_features)) ** 0.5
        # leverage torch.nn.init.trunc_normal_ to initialize the weights with a truncated normal distribution
        # truncated at [-3\sigma, 3\sigma] to avoid extreme values
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=True)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std_dev, a=-3*std_dev, b=3*std_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('... i, o i -> ... o', x, self.weight)


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, num_embeddings: int, 
                       embedding_dim: int, 
                       device: torch.device | None = None, 
                       dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype), requires_grad=True)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNormLayer(torch.nn.Module):
    def __init__(self, d_model: int, 
                       eps: float = 1e-5, 
                       device: torch.device | None = None, 
                       dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        x_result = x_normalized * self.weight
        return x_result.to(in_type)


class SwiGLUFFNLayer(torch.nn.Module):
    def __init__(self, d_model: int, 
                       d_ff: int, 
                       device: torch.device | None = None, 
                       dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = LinearLayer(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = LinearLayer(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = LinearLayer(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = silu(x1)
        x3 = self.w3(x)
        x4 = x2 * x3
        return self.w2(x4)
