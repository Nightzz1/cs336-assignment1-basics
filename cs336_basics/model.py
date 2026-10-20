"""Transformer blocks and the complete language model."""

import torch

from .attention import MultiheadSelfAttentionWithRoPE
from .layers import EmbeddingLayer, LinearLayer, RMSNormLayer, SwiGLUFFNLayer


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, 
                       num_heads: int, 
                       d_ff: int, 
                       theta: float, 
                       max_seq_len: int,
                       device: torch.device | None = None,
                       dtype: torch.dtype | None = None):
        super().__init__()
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ln1 = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFFNLayer(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNormLayer(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_norm1 = self.ln1(x)
        attn_output = self.attn(x_norm1, token_positions)
        x = x + attn_output

        x_norm2 = self.ln2(x)
        ffn_output = self.ffn(x_norm2)
        x = x + ffn_output

        return x


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, 
                       context_length: int,
                       d_model: int, 
                       num_heads: int, 
                       d_ff: int, 
                       theta: float, 
                       num_layers: int,
                       device: torch.device | None = None,
                       dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = EmbeddingLayer(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype) for _ in range(num_layers)
        ])
        self.ln_final = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.lm_head = LinearLayer(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        token_positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        for block in self.layers:
            x = block(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
