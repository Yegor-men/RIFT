import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.conditioning import ClassLabelConditioner
from modules.inference import (
    build_model_from_config,
    infer_sidecar_path,
    load_checkpoint_config,
    newest_checkpoint_path,
)
from modules.rift import RIFT
from modules.rift_diffusion import sample_noise
from modules.sampling import run_rift_sampling

__test__ = False  # This is a plotting script, not a pytest test module.

# CONFIG ===============================================================================================================

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = None  # If None, use newest MNIST_E*_rift.safetensors from MODEL_DIR.
CONDITIONER_PATH = None  # If None, inferred from MODEL_PATH.
CONFIG_PATH = None  # If None, inferred from MODEL_PATH.

DEVICE = "cuda"
SEED = 0
SAVE_FIGURES = False
FIGURE_DIR = Path("MNIST/media/tests")

SAMPLE_NUM_STEPS = 20
SAMPLE_STEP_SIZE = 0.05
SAMPLE_CFG_SCALE = 4.0
TRAJECTORY_STEP_COUNT = 11

SQUARE_SIZES = (14, 28, 64, 128)
ASPECT_SIZES = ()
FINAL_GRID_COUNT_PER_DIGIT = 10

RUN_SQUARE_TRAJECTORIES = True
RUN_ASPECT_RATIO_GRID = False


# CHECKPOINT LOADING ===================================================================================================

def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(device: torch.device) -> tuple[RIFT, ClassLabelConditioner, dict]:
    model_path = Path(MODEL_PATH) if MODEL_PATH else newest_checkpoint_path(MODEL_DIR, "MNIST")
    conditioner_path = Path(CONDITIONER_PATH) if CONDITIONER_PATH else infer_sidecar_path(
        model_path,
        "conditioner.safetensors",
    )
    config = load_checkpoint_config(model_path, CONFIG_PATH)
    model, conditioner, model_config = build_model_from_config(config, device)

    model.load_state_dict(load_file(str(model_path)))
    conditioner.load_state_dict(load_file(str(conditioner_path)))

    model.eval()
    conditioner.eval()

    print(f"Loaded model: {model_path}")
    print(f"Loaded conditioner: {conditioner_path}")
    print(f"Loaded config: {Path(CONFIG_PATH) if CONFIG_PATH else infer_sidecar_path(model_path, 'config.json')}")
    return model, conditioner, model_config


# LABELS / SAMPLING ====================================================================================================

def digit_labels(count_per_digit: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(10, dtype=torch.long, device=device)
    return labels.repeat_interleave(count_per_digit)


@torch.no_grad()
def sample_pixel_trace(
        model: RIFT,
        conditioner: ClassLabelConditioner,
        labels: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
):
    image_channels = int(model.c_channels)
    initial_noise = sample_noise((labels.shape[0], image_channels, height, width), device=device)
    positive_tokens = conditioner(labels)
    negative_tokens = conditioner(torch.full_like(labels, conditioner.num_classes))

    samples, _, image_trace, velocity_trace, time_trace = run_rift_sampling(
        model=model,
        initial_noise=initial_noise,
        positive_text_conditioning=positive_tokens,
        negative_text_conditioning=negative_tokens,
        num_steps=SAMPLE_NUM_STEPS,
        step_size=SAMPLE_STEP_SIZE,
        cfg_scale=SAMPLE_CFG_SCALE,
        device=device,
        return_trace=True,
    )

    return samples, image_trace, velocity_trace, time_trace


# PLOTTING =============================================================================================================

def maybe_savefig(name: str) -> None:
    if not SAVE_FIGURES:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=160)


def pick_trace_indices(trace_len: int, desired_count: int) -> list[int]:
    if trace_len <= desired_count:
        return list(range(trace_len))
    positions = torch.linspace(0, trace_len - 1, steps=desired_count)
    return sorted(set(int(round(pos.item())) for pos in positions))


def show_tensor_grid(tensor: torch.Tensor, title: str, name: str | None = None) -> None:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    batch, channels, _, _ = tensor.shape
    cols = math.ceil(math.sqrt(batch))
    rows = math.ceil(batch / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = axes.flatten() if batch > 1 else [axes]

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= batch:
            continue
        image = tensor[idx]
        if channels == 1:
            ax.imshow(image.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image.permute(1, 2, 0))

    fig.suptitle(title)
    plt.tight_layout()
    if name:
        maybe_savefig(name)
    plt.show()


def show_trace_grid(
        trace: list[torch.Tensor],
        title: str,
        name: str,
        labels: torch.Tensor,
        column_titles: list[str] | None = None,
) -> None:
    row_indices = pick_trace_indices(len(trace), TRAJECTORY_STEP_COUNT)
    batch = trace[0].shape[0]
    rows = len(row_indices)
    cols = batch
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.15), squeeze=False)

    for row, trace_idx in enumerate(row_indices):
        images = trace[trace_idx].detach().cpu().clamp(0.0, 1.0)
        for col in range(cols):
            ax = axes[row][col]
            ax.axis("off")
            image = images[col]
            if image.shape[0] == 1:
                ax.imshow(image.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(image.permute(1, 2, 0))
            if row == 0:
                text = column_titles[col] if column_titles else str(int(labels[col].item()))
                ax.set_title(text, fontsize=8)
        axes[row][0].set_ylabel(f"{trace_idx}", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    maybe_savefig(name)
    plt.show()


# EXPERIMENT 1: SQUARE RESOLUTION TRAJECTORIES =========================================================================

def run_square_resolution_trajectories(model, conditioner, model_config, device):
    labels = digit_labels(1, device)
    del model_config

    for size in SQUARE_SIZES:
        _, image_trace, _, _ = sample_pixel_trace(
            model=model,
            conditioner=conditioner,
            labels=labels,
            height=size,
            width=size,
            device=device,
        )
        show_trace_grid(
            image_trace,
            title=f"Pixel RIFT trajectory | {size}x{size}",
            name=f"square_{size}_pixel_trace",
            labels=labels.cpu(),
        )


# EXPERIMENT 2: ASPECT RATIO GENERALIZATION ============================================================================

def run_aspect_ratio_grid(model, conditioner, model_config, device):
    labels = digit_labels(FINAL_GRID_COUNT_PER_DIGIT, device)
    image_channels = int(model_config["image_channels"])
    positive_tokens = conditioner(labels)
    negative_tokens = conditioner(torch.full_like(labels, conditioner.num_classes))

    for height, width in ASPECT_SIZES:
        initial_noise = sample_noise((labels.shape[0], image_channels, height, width), device=device)
        samples, _ = run_rift_sampling(
            model=model,
            initial_noise=initial_noise,
            positive_text_conditioning=positive_tokens,
            negative_text_conditioning=negative_tokens,
            num_steps=SAMPLE_NUM_STEPS,
            step_size=SAMPLE_STEP_SIZE,
            cfg_scale=SAMPLE_CFG_SCALE,
            device=device,
        )
        show_tensor_grid(
            samples,
            title=f"Pixel RIFT final samples | {height}x{width}",
            name=f"aspect_{height}x{width}",
        )


# MAIN =================================================================================================================

def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE if DEVICE != "cuda" or torch.cuda.is_available() else "cpu")
    model, conditioner, model_config = load_checkpoint(device)

    if RUN_SQUARE_TRAJECTORIES:
        run_square_resolution_trajectories(model, conditioner, model_config, device)
    if RUN_ASPECT_RATIO_GRID:
        run_aspect_ratio_grid(model, conditioner, model_config, device)


if __name__ == "__main__":
    main()
