from torch import nn
from constants import *

import torch
import math


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attention = MultiheadAttention()
        self.layer_norm_1 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.layer_norm_2 = LayerNormalization()

    def forward(self, x, encoder_mask):
        after_attn = self.multihead_attention(x, x, x, mask=encoder_mask) # (B, L, d_model) => (B, L, d_model)
        after_norm_1 = self.layer_norm_1(after_attn + x) # (B, L, d_model) => (B, L, d_model)
        after_ff = self.feed_forward(after_norm_1) # (B, L, d_model) => (B, L, d_ff) => (B, L, d_model)
        after_norm_2 = self.layer_norm_2(after_ff + after_norm_1) # (B, L, d_model) => (B, L, d_model)

        return after_norm_2


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_multihead_attention = MultiheadAttention()
        self.layer_norm1 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.layer_norm2 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.layer_norm3 = LayerNormalization()

    def forward(self, x, encoder_output, masked_attn_mask, attn_mask):
        after_masked_attn = self.masked_multihead_attention(x, x, x, mask=masked_attn_mask) # (B, L, d_model) => (B, L, d_model)
        after_norm_1 = self.layer_norm1(after_masked_attn + x) # (B, L, d_model) => (B, L, d_model)
        after_attn = self.multihead_attention(after_norm_1, encoder_output, encoder_output, mask=attn_mask) # (B, L, d_model) => (B, L, d_model)
        after_norm_2 = self.layer_norm2(after_attn + after_norm_1) # (B, L, d_model) => (B, L, d_model)
        after_ff = self.feed_forward(after_norm_2) # (B, L, d_model) => (B, L, d_ff) => (B, L, d_model)
        after_norm_3 = self.layer_norm3(after_ff + after_norm_2) # (B, L, d_model) => (B, L, d_model)

        return after_norm_3


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(batch_size, seq_len, num_heads, d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(batch_size, seq_len, num_heads, d_k) # (B, L, num_heads, d_k)
        v = self.w_k(v).view(batch_size, seq_len, num_heads, d_k) # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, d_model) # (B, L, num_heads, d_k) = (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_scores = self.dropout(attn_scores)
        attn_distribs = self.attn_softmax(attn_scores) # (B, num_heads, L, L)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x):
        x = self.relu(self.linear_1(x)) # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([seq_len, d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos/(10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, x):
        x *= math.sqrt(d_model) # (B, L, d_model)
        x += self.positional_encoding # (B, L, d_model)

        return x
