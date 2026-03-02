import torch
from torch import nn


class DummyTextCond(nn.Module):
    def __init__(self, token_sequence_length, d_channels):
        super().__init__()
        self.token_sequence_length = token_sequence_length
        self.d_channels = d_channels

        self.token_proj = nn.Linear(10, token_sequence_length * d_channels)
        self.token_norm = nn.LayerNorm(d_channels)

    def forward(self, label_vector):
        # label_vector is of size [B, 10]
        b = label_vector.size(0)

        tokens = self.token_proj(label_vector)
        tokens = tokens.reshape(b, self.token_sequence_length, self.d_channels)
        tokens = self.token_norm(tokens)

        return tokens
