from torch import nn
from constants import *
from layers import *

import torch


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tar_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.tar_embedding = nn.Embedding(self.tar_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(d_model, self.tar_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, tar_input, encoder_mask, masked_attn_mask, attn_mask):
        src_embedded = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
        tar_embedded = self.tar_embedding(tar_input) # (B, L) => (B, L, d_model)
        src_positional_encoded = self.positional_encoder(src_embedded) # (B, L, d_model) => (B, L, d_model)
        tar_positional_encoded = self.positional_encoder(tar_embedded) # (B, L, d_model) => (B, L, d_model)

        encoder_output = self.encoder(src_positional_encoded, encoder_mask) # (B, L, d_model)
        decoder_output = self.decoder(tar_positional_encoded, encoder_output, masked_attn_mask, attn_mask) # (B, L, d_model)

        output = self.softmax(self.output_linear(decoder_output)) # (B, L, d_model) => # (B, L, tar_vocab_size)

        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])

    def forward(self, x, encoder_mask):
        for i in range(num_layers):
            x = self.layers[i](x, encoder_mask)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])

    def forward(self, x, encoder_output, masked_attn_mask, attn_mask):
        for i in range(num_layers):
            x = self.layers[i](x, encoder_output, masked_attn_mask, attn_mask)

        return x
