import torch
from torch import nn


class ClassLabelConditioner(nn.Module):
    """Turns integer class labels into a short sequence of conditioning tokens."""

    def __init__(self, num_classes: int, token_count: int, d_channels: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.null_label = self.num_classes
        self.token_count = int(token_count)
        self.d_channels = int(d_channels)
        self.embedding = nn.Embedding(self.num_classes + 1, self.token_count * self.d_channels)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(dtype=torch.long).clamp(min=0, max=self.null_label)
        tokens = self.embedding(labels)
        return tokens.view(labels.shape[0], self.token_count, self.d_channels)
