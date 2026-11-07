import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        past_length = 0

        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = past_key.size(2)

            if past_key.device != k.device or past_value.device != v.device:
                raise ValueError("Cached keys and values must be on the same device as the input")

            if past_key.dtype != k.dtype or past_value.dtype != v.dtype:
                raise ValueError("Cached keys and values must have the same dtype as the input")

            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(
                T,
                k.size(2),
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=past_length + 1,
        )

        scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        out = attention_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)

        if use_cache:
            return out, (k, v)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)

    def forward(self, x, past_key_value=None, use_cache=False):
        attention_output = self.attention(
            self.norm1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        if use_cache:
            attention_output, present_key_value = attention_output

        x = x + attention_output
        x = x + self.ffn(self.norm2(x))

        if use_cache:
            return x, present_key_value

        return x


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len=32,
        d_model=128,
        num_heads=4,
        num_layers=4,
    ):
        super().__init__()

        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1 when using KV cache")

        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _validate_past_key_values(self, past_key_values, batch_size):
        if past_key_values is None:
            return 0

        if not isinstance(past_key_values, (tuple, list)):
            raise ValueError("past_key_values must be a tuple or list")

        if len(past_key_values) != len(self.blocks):
            raise ValueError(
                "past_key_values must contain one (key, value) pair per layer"
            )

        past_length = None

        for layer_index, layer_cache in enumerate(past_key_values):
            if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
                raise ValueError(
                    f"Cache for layer {layer_index} must be a (key, value) pair"
                )

            key, value = layer_cache

            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Cache for layer {layer_index} must contain tensors"
                )

            if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
                raise ValueError(
                    f"Cache for layer {layer_index} must have matching "
                    "[batch, heads, sequence, head_dim] shapes"
                )

            attention = self.blocks[layer_index].attention
            expected_prefix = (batch_size, attention.num_heads)

            if key.shape[:2] != expected_prefix or key.size(3) != attention.head_dim:
                raise ValueError(
                    f"Cache for layer {layer_index} has an incompatible shape"
                )

            if past_length is None:
                past_length = key.size(2)
            elif key.size(2) != past_length:
                raise ValueError("All layer caches must have the same sequence length")

        return 0 if past_length is None else past_length

    def _forward_impl(self, input_ids, past_key_values=None, use_cache=False):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        B, T = input_ids.shape

        if T == 0:
            raise ValueError("input_ids must contain at least one token")

        past_length = self._validate_past_key_values(
            past_key_values,
            batch_size=B,
        )
        total_length = past_length + T

        if total_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {total_length} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        token_emb = self.token_embedding(input_ids)

        positions = torch.arange(
            past_length,
            total_length,
            device=input_ids.device
        )

        position_emb = self.position_embedding(positions)
        x = token_emb + position_emb

        present_key_values = []

        for layer_index, block in enumerate(self.blocks):
            layer_past = (
                None
                if past_key_values is None
                else past_key_values[layer_index]
            )

            if use_cache:
                x, present_key_value = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                present_key_values.append(present_key_value)
            else:
                x = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=False,
                )

        x = self.norm(x)
        logits = self.lm_head(x)

        next_cache = tuple(present_key_values) if use_cache else None
        return logits, next_cache

    def forward(self, input_ids, targets=None):
        logits, _ = self._forward_impl(input_ids)

        loss = None

        if targets is not None:
            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )

        return logits, loss

    @torch.no_grad()
    def prefill(self, input_ids):
        """Process a prompt once and return its logits and per-layer KV cache."""
        return self._forward_impl(input_ids, use_cache=True)

    @torch.no_grad()
    def decode(self, input_ids, past_key_values):
        """Process one new token using a KV cache returned by prefill/decode."""
        if input_ids.ndim != 2 or input_ids.size(1) != 1:
            raise ValueError("decode expects input_ids with shape [batch, 1]")

        if past_key_values is None:
            raise ValueError("decode requires past_key_values from prefill or decode")

        return self._forward_impl(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
