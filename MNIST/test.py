import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.conditioning import ClassLabelConditioner
from modules.inference import newest_checkpoint_path
from modules.rift import RIFT
from modules.rift_diffusion import (
    rift_training_pair,
    rift_velocity_step,
    sample_noise,
    weighted_velocity_mse_loss,
)

__test__ = False  # This is a plotting script, not a pytest test module.

# CONFIG ===============================================================================================================

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = None  # If None, use newest MNIST_E*_rift.safetensors from MODEL_DIR.
CONDITIONER_PATH = None  # If None, inferred from MODEL_PATH.
CONFIG_PATH = None  # If None, inferred from MODEL_PATH.

DATA_ROOT = "data"
DEVICE = "cuda"
SEED = 0
SAVE_FIGURES = False
FIGURE_DIR = Path("MNIST/media/tests")

SAMPLE_NUM_STEPS = 10
TRAJECTORY_STEP_COUNT = SAMPLE_NUM_STEPS + 1  # Show every image step for default traces.
SAMPLE_STEP_SIZE = 0.2
SAMPLE_EVIDENCE_SCALE = 1.0

CLEAN_START_NUM_STEPS = 20
CLEAN_START_STEP_SIZE = 0.3
CLEAN_START_INVERT_STEPS = 10
CLEAN_START_EVIDENCE_SCALE = 1.0
CLEAN_START_ALPHA = 0.01

LABEL_EVIDENCE_STRENGTH = 1.0

SQUARE_SIZES = (20, 28, 48)
# ASPECT_EDGE_LENGTHS = (28, 32, 36, 40)
ASPECT_EDGE_LENGTHS = (28, 32)

ALPHA_SCRAPE_POINTS = 10
ALPHA_SCRAPE_BATCH_SIZE = 64
ALPHA_SCRAPE_MAX_BATCHES = 8
ALPHA_SCRAPE_IMAGE_SIZE = 28

RUN_SQUARE_TRAJECTORIES = True
RUN_ASPECT_RATIO_GRID = True
RUN_ALPHA_SCRAPE_LOSS = True
RUN_CLEAN_START_TRAJECTORY = True
CLEAN_START_SHOW_MODEL_OUTPUTS = True


# DATA =================================================================================================================

class MNISTImages(torch.utils.data.Dataset):
    def __init__(self, train: bool, image_size: int):
        self.dataset = datasets.MNIST(
            root=DATA_ROOT,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]),
        )

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset)


# CHECKPOINT LOADING ===================================================================================================

def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def newest_model_path() -> Path:
    return newest_checkpoint_path(MODEL_DIR, "MNIST")


def sidecar_path(model_path: Path, suffix: str) -> Path:
    stem = model_path.name.removesuffix("_rift.safetensors")
    return model_path.with_name(f"{stem}_{suffix}")


def load_checkpoint(device: torch.device):
    model_path = Path(MODEL_PATH) if MODEL_PATH else newest_model_path()
    conditioner_path = Path(CONDITIONER_PATH) if CONDITIONER_PATH else sidecar_path(model_path,
                                                                                    "conditioner.safetensors")
    config_path = Path(CONFIG_PATH) if CONFIG_PATH else sidecar_path(model_path, "config.json")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_config = config["model"]

    model = RIFT(
        c_channels=model_config["image_channels"],
        d_channels=model_config["d_channels"],
        num_heads=model_config["num_heads"],
        block_count=model_config["block_count"],
        pos_freq=model_config["pos_freq"],
        time_freq=model_config["time_freq"],
    ).to(device)
    conditioner = ClassLabelConditioner(
        num_classes=model_config["num_classes"],
        token_count=model_config["token_count"],
        d_channels=model_config["d_channels"],
    ).to(device)

    model.load_state_dict(load_file(str(model_path)))
    conditioner.load_state_dict(load_file(str(conditioner_path)))
    model.eval()
    conditioner.eval()

    print(f"Loaded model: {model_path}")
    print(f"Loaded conditioner: {conditioner_path}")
    print(f"Loaded config: {config_path}")
    return model, conditioner, model_config


# LABELS / SAMPLING ====================================================================================================

def digit_labels(count_per_digit: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(10, dtype=torch.long, device=device)
    return labels.repeat_interleave(count_per_digit)


def source_to_target_titles(source_labels: torch.Tensor, target_labels: torch.Tensor) -> list[str]:
    return [f"{int(src)}->{int(dst)}" for src, dst in zip(source_labels.cpu(), target_labels.cpu())]


def clean_digit_batch(image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MNISTImages(train=False, image_size=image_size)
    images_by_digit: dict[int, torch.Tensor] = {}
    for image, label in dataset:
        digit = int(label.item())
        if digit not in images_by_digit:
            images_by_digit[digit] = image
        if len(images_by_digit) == 10:
            break

    missing = sorted(set(range(10)) - set(images_by_digit))
    if missing:
        raise RuntimeError(f"Could not find MNIST examples for digits: {missing}")

    images = torch.stack([images_by_digit[digit] for digit in range(10)], dim=0).to(device).clamp(0.0, 1.0)
    labels = torch.arange(10, dtype=torch.long, device=device)
    return images, labels


@torch.no_grad()
def predict_rift_fields(model, conditioner, x, labels, model_time, label_strength: float, evidence_scale: float):
    text_conditions = [(conditioner(labels), label_strength)]
    return model(x, model_time, text_conditions, evidence_scale=evidence_scale)


@torch.no_grad()
def sample_with_trace(
        model: RIFT,
        conditioner: ClassLabelConditioner,
        labels: torch.Tensor,
        height: int,
        width: int,
        num_steps: int,
        device: torch.device,
        label_strength: float = 1.0,
        evidence_scale: float = 1.0,
        initial_x: torch.Tensor | None = None,
        step_size: float = 0.1,
        invert_steps: int = 0,
):
    if num_steps <= 0:
        raise ValueError(f"steps must be positive, got {num_steps}")
    if step_size < 0.0:
        raise ValueError(f"step_size must be non-negative, got {step_size}")
    if invert_steps < 0 or invert_steps > num_steps:
        raise ValueError(f"invert_steps must be in [0, num_steps], got {invert_steps}")

    if initial_x is None:
        x = sample_noise((labels.shape[0], 1, height, width), device=device)
    else:
        x = initial_x.to(device=device).clamp(0.0, 1.0)
        if x.shape != (labels.shape[0], 1, height, width):
            raise ValueError(
                f"initial_x shape {tuple(x.shape)} does not match labels/size {(labels.shape[0], 1, height, width)}")
    image_trace = [x.detach().cpu()]
    velocity_trace = []
    model_time_values = []

    for step in range(num_steps):
        model_time = torch.zeros((labels.shape[0],), device=device)

        predicted_velocity = predict_rift_fields(
            model,
            conditioner,
            x.clamp(0.0, 1.0),
            labels,
            model_time,
            label_strength,
            evidence_scale,
        )[0]
        velocity_trace.append(predicted_velocity.detach().cpu())
        model_time_values.append(float(model_time[0].item()))
        signed_step_size = -float(step_size) if step < invert_steps else float(step_size)
        x = rift_velocity_step(x, predicted_velocity, step_size=signed_step_size)
        image_trace.append(x.detach().cpu())

    return image_trace, velocity_trace, model_time_values


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


def show_tensor_grid(tensor: torch.Tensor, title: str, name: str | None = None, cmap: str = "gray") -> None:
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
            ax.imshow(image.squeeze(0), cmap=cmap, vmin=0.0, vmax=1.0)
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
        value_range: tuple[float, float] = (0.0, 1.0),
        map_signed: bool = False,
        column_titles: list[str] | None = None,
) -> None:
    row_indices = pick_trace_indices(len(trace), TRAJECTORY_STEP_COUNT)
    batch = trace[0].shape[0]
    rows = len(row_indices)
    cols = batch
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.15), squeeze=False)
    vmin, vmax = value_range

    for row, trace_idx in enumerate(row_indices):
        images = trace[trace_idx].detach().cpu()
        if map_signed:
            signed_images = images.clamp(-1.0, 1.0)
            red = signed_images.clamp(min=0.0)
            blue = (-signed_images).clamp(min=0.0)
            green = torch.zeros_like(red)
            images = torch.cat([red, green, blue], dim=1).clamp(0.0, 1.0)
        else:
            images = images.clamp(vmin, vmax)
        for col in range(cols):
            ax = axes[row][col]
            ax.axis("off")
            image = images[col]
            if image.shape[0] == 1:
                ax.imshow(image.squeeze(0), cmap="gray", vmin=vmin, vmax=vmax)
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


def show_alpha_loss_curve(alpha_values: torch.Tensor, losses: list[float]) -> None:
    plt.figure(figsize=(10, 4))
    plt.title("MNIST RIFT residual velocity loss across fixed alpha values")
    plt.plot(alpha_values.cpu().tolist(), losses)
    plt.xlabel("global alpha used for x_t = (1 - alpha) * x1 + alpha * x0_noise")
    plt.ylabel("MSE(predicted velocity, x1 - x_t)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    maybe_savefig("alpha_scrape_loss_curve")
    plt.show()


# EXPERIMENT 1: SQUARE RESOLUTION GENERALIZATION =======================================================================

def run_square_resolution_trajectories(model, conditioner, device):
    labels = digit_labels(1, device)
    for size in SQUARE_SIZES:
        image_trace, velocity_trace, _ = sample_with_trace(
            model=model,
            conditioner=conditioner,
            labels=labels,
            height=size,
            width=size,
            num_steps=SAMPLE_NUM_STEPS,
            device=device,
            label_strength=LABEL_EVIDENCE_STRENGTH,
            evidence_scale=SAMPLE_EVIDENCE_SCALE,
            step_size=SAMPLE_STEP_SIZE,
        )
        show_trace_grid(
            image_trace,
            title=f"1:1 RIFT trajectory | {size}x{size}",
            name=f"square_{size}_image_trace",
            labels=labels.cpu(),
        )
        show_trace_grid(
            velocity_trace,
            title=f"1:1 predicted residual velocity | {size}x{size}",
            name=f"square_{size}_predicted_velocity_trace",
            labels=labels.cpu(),
            value_range=(-1.0, 1.0),
            map_signed=True,
        )


# EXPERIMENT 2: ASPECT RATIO GENERALIZATION ============================================================================

def run_aspect_ratio_grid(model, conditioner, device):
    labels = digit_labels(10, device)
    for height in ASPECT_EDGE_LENGTHS:
        for width in ASPECT_EDGE_LENGTHS:
            image_trace, _, _ = sample_with_trace(
                model=model,
                conditioner=conditioner,
                labels=labels,
                height=height,
                width=width,
                num_steps=SAMPLE_NUM_STEPS,
                device=device,
                label_strength=LABEL_EVIDENCE_STRENGTH,
                evidence_scale=SAMPLE_EVIDENCE_SCALE,
                step_size=SAMPLE_STEP_SIZE,
            )
            show_tensor_grid(
                image_trace[-1],
                title=f"Aspect-ratio final samples | {height}x{width}",
                name=f"aspect_{height}x{width}",
            )


# EXPERIMENT 3: ALPHA SCRAPE LOSS CURVE ================================================================================

@torch.no_grad()
def run_alpha_scrape_loss(model, conditioner, device):
    dataset = MNISTImages(train=False, image_size=ALPHA_SCRAPE_IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=ALPHA_SCRAPE_BATCH_SIZE, shuffle=True, num_workers=0)
    alpha_values = torch.linspace(0.0, 1.0, steps=ALPHA_SCRAPE_POINTS, device=device)
    losses = []

    model.eval()
    conditioner.eval()
    for alpha_value in tqdm(
            alpha_values,
            total=len(alpha_values),
            desc="alpha scrape",
    ):
        total_loss = 0.0
        batches = 0
        for images, labels in dataloader:
            if batches >= ALPHA_SCRAPE_MAX_BATCHES:
                break
            images = images.to(device).clamp(0.0, 1.0)
            labels = labels.to(device, dtype=torch.long)
            alpha_batch = torch.full((images.shape[0],), float(alpha_value.item()), device=device)
            model_input, target_velocity, _, alpha_map = rift_training_pair(
                images,
                alpha=alpha_batch,
            )
            model_time = torch.zeros(images.shape[0], device=device)
            predicted_velocity = model(
                model_input,
                model_time,
                [(conditioner(labels), LABEL_EVIDENCE_STRENGTH)],
                evidence_scale=SAMPLE_EVIDENCE_SCALE,
            )[0]
            total_loss += weighted_velocity_mse_loss(predicted_velocity, target_velocity, alpha_map).item()
            batches += 1
        losses.append(total_loss / max(1, batches))

    show_alpha_loss_curve(alpha_values, losses)
    best_idx = min(range(len(losses)), key=lambda idx: losses[idx])
    worst_idx = max(range(len(losses)), key=lambda idx: losses[idx])
    print(f"Best alpha={float(alpha_values[best_idx].item()):.4f} loss={losses[best_idx]:.6f}")
    print(f"Worst alpha={float(alpha_values[worst_idx].item()):.4f} loss={losses[worst_idx]:.6f}")


# EXPERIMENT 4: CLEAN-IMAGE START ======================================================================================

def run_clean_start_trajectory(model, conditioner, device):
    clean_images, source_labels = clean_digit_batch(image_size=28, device=device)
    clean_start_noise = sample_noise(clean_images.shape, device=device)
    clean_start_images = (
            (1.0 - CLEAN_START_ALPHA) * clean_images + CLEAN_START_ALPHA * clean_start_noise
    ).clamp(0.0, 1.0)

    for shift in range(10):
        target_labels = (source_labels + shift) % 10
        image_trace, velocity_trace, _ = sample_with_trace(
            model=model,
            conditioner=conditioner,
            labels=target_labels,
            height=28,
            width=28,
            num_steps=CLEAN_START_NUM_STEPS,
            device=device,
            label_strength=LABEL_EVIDENCE_STRENGTH,
            evidence_scale=CLEAN_START_EVIDENCE_SCALE,
            initial_x=clean_start_images,
            step_size=CLEAN_START_STEP_SIZE,
            invert_steps=CLEAN_START_INVERT_STEPS,
        )
        column_titles = source_to_target_titles(source_labels, target_labels)
        show_trace_grid(
            image_trace,
            title=(
                f"28x28 clean-start overwrite trajectory | target = source + {shift} mod 10 | "
                f"num_steps={CLEAN_START_NUM_STEPS}, step_size={CLEAN_START_STEP_SIZE:g}, "
                f"invert_steps={CLEAN_START_INVERT_STEPS}, evidence_scale={CLEAN_START_EVIDENCE_SCALE:g}, "
                f"start_alpha={CLEAN_START_ALPHA:g}"
            ),
            name=f"clean_start_shift_{shift}_image_trace",
            labels=target_labels.cpu(),
            column_titles=column_titles,
        )
        if CLEAN_START_SHOW_MODEL_OUTPUTS:
            show_trace_grid(
                velocity_trace,
                title=f"28x28 clean-start predicted residual velocity | target = source + {shift} mod 10",
                name=f"clean_start_shift_{shift}_predicted_velocity_trace",
                labels=target_labels.cpu(),
                value_range=(-1.0, 1.0),
                map_signed=True,
                column_titles=column_titles,
            )


# MAIN =================================================================================================================

def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE if DEVICE != "cuda" or torch.cuda.is_available() else "cpu")
    model, conditioner, _ = load_checkpoint(device)

    if RUN_SQUARE_TRAJECTORIES:
        run_square_resolution_trajectories(model, conditioner, device)
    if RUN_ASPECT_RATIO_GRID:
        run_aspect_ratio_grid(model, conditioner, device)
    if RUN_ALPHA_SCRAPE_LOSS:
        run_alpha_scrape_loss(model, conditioner, device)
    if RUN_CLEAN_START_TRAJECTORY:
        run_clean_start_trajectory(model, conditioner, device)


if __name__ == "__main__":
    main()
