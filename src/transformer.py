from torch import nn
from constants import *
from layers import *

import torch


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, encoder_mask=None, decoder_mask=None):
        src_embedded = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
        trg_embedded = self.trg_embedding(trg_input) # (B, L) => (B, L, d_model)
        src_positional_encoded = self.positional_encoder(src_embedded) # (B, L, d_model) => (B, L, d_model)
        trg_positional_encoded = self.positional_encoder(trg_embedded) # (B, L, d_model) => (B, L, d_model)

        encoder_output = self.encoder(src_positional_encoded, encoder_mask) # (B, L, d_model)
        decoder_output = self.decoder(trg_positional_encoded, encoder_output, encoder_mask, decoder_mask) # (B, L, d_model)

        output = self.softmax(self.output_linear(decoder_output)) # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, encoder_mask):
        for i in range(num_layers):
            x = self.layers[i](x, encoder_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for i in range(num_layers):
            x = self.layers[i](x, encoder_output, encoder_mask, decoder_mask)

        return self.layer_norm(x)
