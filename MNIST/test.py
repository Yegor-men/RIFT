import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.conditioning import ClassLabelConditioner
from modules.inference import newest_checkpoint_path
from modules.r2id import R2ID
from modules.velocity import sample_noise, velocity_step, velocity_training_pair

__test__ = False  # This is a plotting script, not a pytest test module.


# CONFIG ===============================================================================================================

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = None  # If None, use newest MNIST_E*_r2id.safetensors from MODEL_DIR.
CONDITIONER_PATH = None  # If None, inferred from MODEL_PATH.
CONFIG_PATH = None  # If None, inferred from MODEL_PATH.

DATA_ROOT = "data"
DEVICE = "cuda"
SEED = 0
SAVE_FIGURES = False
FIGURE_DIR = Path("MNIST/media/tests")

CFG_SCALE = 1.0
SAMPLE_STEPS = 16
TRAJECTORY_STEP_COUNT = 10  # Number of timestep rows to show; snapshots are sub-sampled from SAMPLE_STEPS.

SQUARE_SIZES = (20, 28, 48)
# ASPECT_EDGE_LENGTHS = (28, 32, 36, 40)
ASPECT_EDGE_LENGTHS = (28, 32)
WRONG_T_VALUES = (None, 0.1, 0.5, 0.9)  # None means normal true-t sampling.

T_SCRAPE_POINTS = 200
T_SCRAPE_BATCH_SIZE = 64
T_SCRAPE_MAX_BATCHES = 8
T_SCRAPE_IMAGE_SIZE = 28

RUN_SQUARE_TRAJECTORIES = True
RUN_ASPECT_RATIO_GRID = True
RUN_WRONG_T_TRAJECTORIES = True
RUN_T_SCRAPE_LOSS = True
RUN_CLEAN_START_TRAJECTORY = True
CLEAN_START_SHOW_VELOCITY = False


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
    stem = model_path.name.removesuffix("_r2id.safetensors")
    return model_path.with_name(f"{stem}_{suffix}")


def load_checkpoint(device: torch.device):
    model_path = Path(MODEL_PATH) if MODEL_PATH else newest_model_path()
    conditioner_path = Path(CONDITIONER_PATH) if CONDITIONER_PATH else sidecar_path(model_path, "conditioner.safetensors")
    config_path = Path(CONFIG_PATH) if CONFIG_PATH else sidecar_path(model_path, "config.json")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_config = config["model"]

    model = R2ID(
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
def predict_velocity(model, conditioner, x, labels, t_for_model, cfg_scale: float):
    pos_tokens = conditioner(labels)
    if cfg_scale == 1.0:
        return model(x, t_for_model, [pos_tokens])[0]

    null_labels = torch.full_like(labels, conditioner.null_label)
    null_tokens = conditioner(null_labels)
    v_null, v_pos = model(x, t_for_model, [null_tokens, pos_tokens])
    return v_null + cfg_scale * (v_pos - v_null)


@torch.no_grad()
def sample_with_trace(
        model: R2ID,
        conditioner: ClassLabelConditioner,
        labels: torch.Tensor,
        height: int,
        width: int,
        steps: int,
        device: torch.device,
        forced_t: float | None = None,
        cfg_scale: float = 1.0,
        initial_x: torch.Tensor | None = None,
):
    if initial_x is None:
        x = sample_noise((labels.shape[0], 1, height, width), device=device)
    else:
        x = initial_x.to(device=device).clamp(0.0, 1.0)
        if x.shape != (labels.shape[0], 1, height, width):
            raise ValueError(f"initial_x shape {tuple(x.shape)} does not match labels/size {(labels.shape[0], 1, height, width)}")
    times = torch.linspace(0.0, 1.0, steps=steps + 1, device=device)
    image_trace = [x.detach().cpu()]
    velocity_trace = []
    used_t_values = []

    for step in range(steps):
        t_current = torch.full((labels.shape[0],), float(times[step].item()), device=device)
        t_next = torch.full((labels.shape[0],), float(times[step + 1].item()), device=device)
        if forced_t is None:
            t_for_model = t_current
        else:
            t_for_model = torch.full_like(t_current, float(forced_t))

        v = predict_velocity(model, conditioner, x.clamp(0.0, 1.0), labels, t_for_model, cfg_scale)
        velocity_trace.append(v.detach().cpu())
        used_t_values.append(float(t_for_model[0].item()))
        x, _ = velocity_step(x, t_current, t_next, v)
        image_trace.append(x.detach().cpu())

    return image_trace, velocity_trace, used_t_values


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
        map_velocity: bool = False,
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
        if map_velocity:
            images = (images * 0.5 + 0.5).clamp(0.0, 1.0)
            vmin, vmax = 0.0, 1.0
        else:
            images = images.clamp(vmin, vmax)
        for col in range(cols):
            ax = axes[row][col]
            ax.axis("off")
            ax.imshow(images[col].squeeze(0), cmap="gray", vmin=vmin, vmax=vmax)
            if row == 0:
                text = column_titles[col] if column_titles else str(int(labels[col].item()))
                ax.set_title(text, fontsize=8)
        axes[row][0].set_ylabel(f"{trace_idx}", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    maybe_savefig(name)
    plt.show()


def show_t_loss_curve(t_values: torch.Tensor, losses: list[float]) -> None:
    plt.figure(figsize=(10, 4))
    plt.title("MNIST R2ID velocity MSE across fixed t values")
    plt.plot(t_values.cpu().tolist(), losses)
    plt.xlabel("t given to model / corruption mix")
    plt.ylabel("MSE(predicted velocity, clean - noise)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    maybe_savefig("t_scrape_loss_curve")
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
            steps=SAMPLE_STEPS,
            device=device,
            forced_t=None,
            cfg_scale=CFG_SCALE,
        )
        show_trace_grid(
            image_trace,
            title=f"1:1 diffusion trajectory | {size}x{size}",
            name=f"square_{size}_image_trace",
            labels=labels.cpu(),
        )
        show_trace_grid(
            velocity_trace,
            title=f"1:1 model velocity outputs | {size}x{size}",
            name=f"square_{size}_velocity_trace",
            labels=labels.cpu(),
            map_velocity=True,
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
                steps=SAMPLE_STEPS,
                device=device,
                forced_t=None,
                cfg_scale=CFG_SCALE,
            )
            show_tensor_grid(
                image_trace[-1],
                title=f"Aspect-ratio final samples | {height}x{width}",
                name=f"aspect_{height}x{width}",
            )


# EXPERIMENT 3: WRONG-T SAMPLING =======================================================================================

def run_wrong_t_trajectories(model, conditioner, device):
    labels = digit_labels(1, device)
    for forced_t in WRONG_T_VALUES:
        image_trace, velocity_trace, used_t_values = sample_with_trace(
            model=model,
            conditioner=conditioner,
            labels=labels,
            height=28,
            width=28,
            steps=SAMPLE_STEPS,
            device=device,
            forced_t=forced_t,
            cfg_scale=CFG_SCALE,
        )
        name = "true_t" if forced_t is None else f"forced_t_{forced_t:.2f}"
        title_t = "true t" if forced_t is None else f"forced t={forced_t:.2f}"
        show_trace_grid(
            image_trace,
            title=f"28x28 diffusion trajectory | {title_t}",
            name=f"wrong_t_{name}_image_trace",
            labels=labels.cpu(),
        )
        show_trace_grid(
            velocity_trace,
            title=f"28x28 model velocity outputs | {title_t}",
            name=f"wrong_t_{name}_velocity_trace",
            labels=labels.cpu(),
            map_velocity=True,
        )
        print(f"Wrong-t run {name}: used model t values {used_t_values[:5]} ... {used_t_values[-5:]}")


# EXPERIMENT 4: T-SCRAPE LOSS CURVE ====================================================================================

@torch.no_grad()
def run_t_scrape_loss(model, conditioner, device):
    dataset = MNISTImages(train=False, image_size=T_SCRAPE_IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=T_SCRAPE_BATCH_SIZE, shuffle=True, num_workers=0)
    t_values = torch.linspace(0.0, 1.0, steps=T_SCRAPE_POINTS, device=device)
    losses = []

    model.eval()
    conditioner.eval()
    for t_value in tqdm(t_values, total=len(t_values), desc="t scrape"):
        total_loss = 0.0
        batches = 0
        for images, labels in dataloader:
            if batches >= T_SCRAPE_MAX_BATCHES:
                break
            images = images.to(device).clamp(0.0, 1.0)
            labels = labels.to(device, dtype=torch.long)
            t_batch = torch.full((images.shape[0],), float(t_value.item()), device=device)
            model_input, target_velocity, _ = velocity_training_pair(images, t_batch)
            predicted_velocity = model(model_input, t_batch, [conditioner(labels)])[0]
            total_loss += nn.functional.mse_loss(predicted_velocity, target_velocity).item()
            batches += 1
        losses.append(total_loss / max(1, batches))

    show_t_loss_curve(t_values, losses)
    best_idx = min(range(len(losses)), key=lambda idx: losses[idx])
    worst_idx = max(range(len(losses)), key=lambda idx: losses[idx])
    print(f"Best t={float(t_values[best_idx].item()):.4f} loss={losses[best_idx]:.6f}")
    print(f"Worst t={float(t_values[worst_idx].item()):.4f} loss={losses[worst_idx]:.6f}")


# EXPERIMENT 5: CLEAN-IMAGE START ======================================================================================

def run_clean_start_trajectory(model, conditioner, device):
    clean_images, source_labels = clean_digit_batch(image_size=28, device=device)
    for shift in range(10):
        target_labels = (source_labels + shift) % 10
        image_trace, velocity_trace, _ = sample_with_trace(
            model=model,
            conditioner=conditioner,
            labels=target_labels,
            height=28,
            width=28,
            steps=SAMPLE_STEPS,
            device=device,
            forced_t=None,
            cfg_scale=CFG_SCALE,
            initial_x=clean_images,
        )
        column_titles = source_to_target_titles(source_labels, target_labels)
        show_trace_grid(
            image_trace,
            title=f"28x28 clean-start overwrite trajectory | target = source + {shift} mod 10",
            name=f"clean_start_shift_{shift}_image_trace",
            labels=target_labels.cpu(),
            column_titles=column_titles,
        )
        if CLEAN_START_SHOW_VELOCITY:
            show_trace_grid(
                velocity_trace,
                title=f"28x28 clean-start overwrite velocity | target = source + {shift} mod 10",
                name=f"clean_start_shift_{shift}_velocity_trace",
                labels=target_labels.cpu(),
                map_velocity=True,
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
    if RUN_WRONG_T_TRAJECTORIES:
        run_wrong_t_trajectories(model, conditioner, device)
    if RUN_T_SCRAPE_LOSS:
        run_t_scrape_loss(model, conditioner, device)
    if RUN_CLEAN_START_TRAJECTORY:
        run_clean_start_trajectory(model, conditioner, device)


if __name__ == "__main__":
    main()
