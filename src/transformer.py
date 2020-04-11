from torch import nn
from config import *

import torch
import math


class MultiheadAttention(nn.Module):
    def __init__(self):
        self.inf = 1e9

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax()

        # Final output linear transformation
        self.w_0 = nn.Parameter(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(batch_size, seq_len, num_heads, d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(batch_size, seq_len, num_heads, d_k) # (B, L, num_heads, d_k)
        v = self.w_k(v).view(batch_size, seq_len, num_heads, d_k) # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_scores = self.dropout(attn_scores)
        attn_distrib = self.attn_softmax(attn_scores) # (B, num_heads, L, L)
        attn_value = torch.matmul(attn_distrib, v) # (B, num_heads, L, d_k)

        return attn_value

class PositionalEncoder(nn.Module):
    def __init__(self):
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                k = i // 2
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos/(10000 ** (2 * k / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * k / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, x):
        x *= math.sqrt(self.d_model) # (B, L, d_model)
        x += self.positional_encoding # (B, L, d_model)

        return x