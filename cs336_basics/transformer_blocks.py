import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = q.size(-1)
    # use torch.einsum for matrix multiplication and transpose
    scores = torch.einsum('... i d, ... j d -> ... i j', q, k) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = softmax(scores, dim=-1)
    output = torch.einsum('... i j, ... j k -> ... i k', attention_weights, v)
    return output

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
        x2 = x1 * sigmoid(x1)
        x3 = self.w3(x)
        x4 = x2 * x3
        return self.w2(x4)


class RotaryPositionalEmbeddingLayer(torch.nn.Module):
    def __init__(self, theta: float, 
                       d_k: int, 
                       max_seq_len: int, 
                       device: torch.device | None = None,
                       dtype: torch.dtype | None = None):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for rotary positional embeddings."
        self.d_k = d_k

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype).float() / d_k))

        positions = torch.arange(max_seq_len, device=device, dtype=dtype).float()

        angles = torch.einsum('p, d -> p d', positions, inv_freq)
        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        while token_positions.ndim < x.ndim - 1:
            token_positions = token_positions.unsqueeze(-2)

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)

        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        y = torch.stack([y1, y2], dim=-1)

        return y.flatten(-2)


class MultiheadSelfAttentionLayer(torch.nn.Module):
    def __init__(self, d_model: int, 
                       num_heads: int,
                       device: torch.device | None = None,
                       dtype: torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = LinearLayer(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = LinearLayer(d_model, d_model, device=device, dtype=dtype)

    # with casual masking, the mask should be a lower triangular matrix of shape (seq_len, seq_len) with 1s in the lower triangle and 0s elsewhere. The mask should be broadcastable to the shape of the attention scores.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)


        Q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        mask = torch.tril(torch.ones((x.size(1), x.size(1)), device=x.device)).unsqueeze(0).unsqueeze(0)

        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.output_proj(attn_output)
        return output

# apply the rotary positional embeddings to the Q and K matrices before computing attention
class MultiheadSelfAttentionWithRoPE(MultiheadSelfAttentionLayer):
    def __init__(self, d_model: int, 
                       num_heads: int,
                       theta: float,
                       max_seq_len: int,
                       device: torch.device | None = None,
                       dtype: torch.dtype | None = None):
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbeddingLayer(theta, self.d_k, max_seq_len, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert token_positions is not None, "token_positions must be provided for RoPE."
        batch_size = x.size(0)
        seq_len = x.size(1)

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        Q = self.rope(Q, token_positions).view(batch_size, self.num_heads, seq_len, self.d_k)
        K = self.rope(K, token_positions).view(batch_size, self.num_heads, seq_len, self.d_k)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0)

        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)

        output = self.output_proj(attn_output)
        return output

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


        

