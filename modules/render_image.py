import torch
import matplotlib.pyplot as plt
import math
from pathlib import Path


def render_image(tensor: torch.Tensor, title: str = None, name: str = "", save: bool = False):
    B, C, H, W = tensor.shape

    cols = math.ceil(math.sqrt(B))
    rows = math.ceil(B / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if B > 1 else [axes]

    for i in range(rows * cols):
        ax = axes[i]
        if i < B:
            img = tensor[i]
            if C == 1:
                img = img.squeeze(0)
                ax.imshow(img.cpu(), cmap='gray')
            elif C == 3:
                img = img.permute(1, 2, 0)
                ax.imshow(img.cpu())
            else:
                raise ValueError(f"Unsupported number of channels: {C}")
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)

    plt.tight_layout()
    if save:
        Path("media").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"media/{name}.png")
    plt.show()
