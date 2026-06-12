import copy
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.conditioning import ClassLabelConditioner
from modules.rift import RIFT
from modules.render_image import render_image
from modules.sampling import run_rift_sampling
from modules.rift_diffusion import rift_training_pair, sample_noise, weighted_velocity_mse_loss

# CONFIG ===============================================================================================================

DATASET_NAME = "MNIST"
DATA_ROOT = "data"
OUTPUT_DIR = "models"
DEVICE = "cuda"
SEED = 0

IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 10

D_CHANNELS = 256
NUM_HEADS = 8
BLOCK_COUNT = 8
POS_FREQ = 4
TIME_FREQ = 5
TOKEN_COUNT = 2

SELF_ATTN_DROPOUT = 0.0
CROSS_ATTN_DROPOUT = 0.0
FFN_DROPOUT = 0.0

EPOCHS = 40
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 2
LR = 1e-3
LR_END = 1e-6
EMA_DECAY = 0.999
VELOCITY_LOSS_WEIGHT = 1.0
ALPHA_LOSS_WEIGHT_MAX = 100.0

TEST_SIZES = (14, 28, 64)
ALPHA_LOSS_VALUES = (0.00, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98, 1.00)
SAMPLE_SIZES = (14, 28, 64)
SAMPLE_STEP_SIZE = 0.05
SAMPLE_STEPS = 20
SAMPLE_COUNT = 10
CONDITION_STRENGTH = 1.0
EVIDENCE_SCALE = 1.0

SAMPLE_EVERY = 1
SAVE_EVERY = 1
PLOT_EVERY = 1
MAX_TRAIN_BATCHES = None
MAX_TEST_BATCHES = None


# DATA =================================================================================================================

class MNISTImages(torch.utils.data.Dataset):
    def __init__(self, train: bool):
        self.dataset = datasets.MNIST(
            root=DATA_ROOT,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
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


# SMALL HELPERS ========================================================================================================

def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def format_parameters(num_params: int) -> str:
    if num_params >= 1_000_000:
        return f"{num_params:,} ({num_params / 1_000_000:.2f}M)"
    if num_params >= 1_000:
        return f"{num_params:,} ({num_params / 1_000:.1f}K)"
    return f"{num_params:,}"


def resize_image(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.shape[-2:] == (size, size):
        return image
    return torch.nn.functional.interpolate(
        image,
        size=(size, size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def make_cosine_with_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    peak_lr = float(optimizer.defaults["lr"])
    min_mult = LR_END / peak_lr

    def lr_lambda(step: int):
        step = float(step)
        if step <= 0:
            return max(min_mult, 0.0)
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_mult + (1.0 - min_mult) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module) -> None:
    for param, ema_param in zip(source.parameters(), target.parameters()):
        ema_param.data.mul_(EMA_DECAY).add_(param.data, alpha=1.0 - EMA_DECAY)


def label_grid(count: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(NUM_CLASSES, dtype=torch.long, device=device)
    repeats = (count + NUM_CLASSES - 1) // NUM_CLASSES
    return labels.repeat(repeats)[:count]


# MODEL / LOSS =========================================================================================================

def build_model(device: torch.device) -> RIFT:
    return RIFT(
        c_channels=IMAGE_CHANNELS,
        d_channels=D_CHANNELS,
        num_heads=NUM_HEADS,
        block_count=BLOCK_COUNT,
        pos_freq=POS_FREQ,
        time_freq=TIME_FREQ,
        self_attn_dropout=SELF_ATTN_DROPOUT,
        cross_attn_dropout=CROSS_ATTN_DROPOUT,
        ffn_dropout=FFN_DROPOUT,
    ).to(device)


def build_conditioner(device: torch.device) -> ClassLabelConditioner:
    return ClassLabelConditioner(NUM_CLASSES, TOKEN_COUNT, D_CHANNELS).to(device)


def rift_prediction_loss(model: RIFT, conditioner: ClassLabelConditioner, image: torch.Tensor,
                         labels: torch.Tensor):
    image = image.clamp(0.0, 1.0)
    model_input, target_velocity, _, alpha_map = rift_training_pair(
        image,
    )
    model_time = torch.zeros(image.shape[0], device=image.device)
    condition_tokens = conditioner(labels)
    predicted_velocity = model(
        model_input,
        model_time,
        [(condition_tokens, CONDITION_STRENGTH)],
        evidence_scale=EVIDENCE_SCALE,
    )[0]
    return VELOCITY_LOSS_WEIGHT * weighted_velocity_mse_loss(
        predicted_velocity,
        target_velocity,
        alpha_map,
        max_weight=ALPHA_LOSS_WEIGHT_MAX,
    )


# EVAL / SAMPLING ======================================================================================================

@torch.no_grad()
def evaluate(model: RIFT, conditioner: ClassLabelConditioner, dataloader: DataLoader, device: torch.device):
    model.eval()
    conditioner.eval()
    losses_by_size = {}

    for size in TEST_SIZES:
        total_loss = 0.0
        batches = 0
        for image, labels in tqdm(dataloader, total=len(dataloader), desc=f"test {size}px", leave=False):
            if MAX_TEST_BATCHES is not None and batches >= MAX_TEST_BATCHES:
                break
            image = image[:EVAL_BATCH_SIZE]
            labels = labels[:EVAL_BATCH_SIZE]
            image = resize_image(image, size).to(device).clamp(0.0, 1.0)
            labels = labels.to(device, dtype=torch.long)

            model_input, target_velocity, _, alpha_map = rift_training_pair(
                image,
            )
            model_time = torch.zeros(image.shape[0], device=device)
            predicted_velocity = model(
                model_input,
                model_time,
                [(conditioner(labels), CONDITION_STRENGTH)],
                evidence_scale=EVIDENCE_SCALE,
            )[0]
            total_loss += (
                    VELOCITY_LOSS_WEIGHT * weighted_velocity_mse_loss(
                predicted_velocity,
                target_velocity,
                alpha_map,
                max_weight=ALPHA_LOSS_WEIGHT_MAX,
            )
            ).item()
            batches += 1

        losses_by_size[size] = total_loss / max(1, batches)

    return losses_by_size


@torch.no_grad()
def evaluate_by_alpha(
        model: RIFT,
        conditioner: ClassLabelConditioner,
        dataloader: DataLoader,
        device: torch.device,
):
    model.eval()
    conditioner.eval()
    image, labels = next(iter(dataloader))
    image = image[:EVAL_BATCH_SIZE].to(device).clamp(0.0, 1.0)
    labels = labels[:EVAL_BATCH_SIZE].to(device, dtype=torch.long)

    losses = []
    for value in ALPHA_LOSS_VALUES:
        alpha = torch.full((image.shape[0],), float(value), device=device)
        model_input, target_velocity, _, alpha_map = rift_training_pair(
            image,
            alpha=alpha,
        )
        model_time = torch.zeros(image.shape[0], device=device)
        predicted_velocity = model(
            model_input,
            model_time,
            [(conditioner(labels), CONDITION_STRENGTH)],
            evidence_scale=EVIDENCE_SCALE,
        )[0]
        loss = VELOCITY_LOSS_WEIGHT * weighted_velocity_mse_loss(
            predicted_velocity,
            target_velocity,
            alpha_map,
            max_weight=ALPHA_LOSS_WEIGHT_MAX,
        )
        losses.append(loss.item())
    return losses


@torch.no_grad()
def render_samples(model: RIFT, conditioner: ClassLabelConditioner, epoch: int, device: torch.device) -> None:
    model.eval()
    conditioner.eval()
    labels = label_grid(SAMPLE_COUNT, device)
    text_conditions = [(conditioner(labels), CONDITION_STRENGTH)]

    for size in SAMPLE_SIZES:
        initial_noise = sample_noise((labels.shape[0], IMAGE_CHANNELS, size, size), device=device)
        samples, _ = run_rift_sampling(
            model=model,
            initial_noise=initial_noise,
            text_conditions=text_conditions,
            num_steps=SAMPLE_STEPS,
            step_size=SAMPLE_STEP_SIZE,
            evidence_scale=EVIDENCE_SCALE,
            device=device,
        )
        render_image(samples, title=f"MNIST RIFT | E{epoch + 1} | {size}px")


# SAVE / PLOT ==========================================================================================================

def checkpoint_config() -> dict:
    return {
        "dataset": DATASET_NAME,
        "image_size": IMAGE_SIZE,
        "model": {
            "image_channels": IMAGE_CHANNELS,
            "num_classes": NUM_CLASSES,
            "d_channels": D_CHANNELS,
            "num_heads": NUM_HEADS,
            "block_count": BLOCK_COUNT,
            "pos_freq": POS_FREQ,
            "time_freq": TIME_FREQ,
            "token_count": TOKEN_COUNT,
            "time_conditioning": False,
            "prediction_target": "velocity = clean_image - x_t",
            "model_input": "x_t = (1 - alpha) * clean_image + alpha * x0_noise",
            "x0_sampling": "x0_noise = torch.rand_like(clean_image)",
            "alpha_sampling": "per-image alpha U(0, 1)",
            "loss": "per-image MSE scaled by min(alpha^-2, 100), normalized by sum of batch weights",
            "sampler": "x_next = clamp(x + step_size * predicted_velocity)",
            "conditioning": "evidence accumulation with per-condition strengths and per-pixel AdaLN",
            "evidence_scale": EVIDENCE_SCALE,
            "sample_steps": SAMPLE_STEPS,
            "sample_step_size": SAMPLE_STEP_SIZE,
        },
    }


def save_checkpoint(model: RIFT, conditioner: ClassLabelConditioner, epoch: int, test_loss: float) -> None:
    folder = Path(OUTPUT_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"MNIST_E{epoch + 1:03d}_{test_loss:.5f}_{timestamp}"

    save_file(model.state_dict(), str(folder / f"{stem}_rift.safetensors"))
    save_file(conditioner.state_dict(), str(folder / f"{stem}_conditioner.safetensors"))
    with open(folder / f"{stem}_config.json", "w", encoding="utf-8") as handle:
        json.dump(checkpoint_config(), handle, indent=2)
    print(f"Saved checkpoint stem: {folder / stem}")


def plot_history(history: dict, epoch: int) -> None:
    if PLOT_EVERY <= 0 or (epoch + 1) % PLOT_EVERY != 0:
        return

    plt.figure()
    plt.title("MNIST RIFT Alpha-Weighted Residual Velocity Loss")
    plt.plot(history["train"], label="train")
    plt.plot(history["test"], label="test avg")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("MNIST RIFT Test Loss by Resolution")
    for size, values in history["test_by_size"].items():
        plt.plot(values, label=f"{size}px")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("MNIST RIFT Loss by Fixed Alpha")
    alpha_history = torch.tensor(history["alpha_losses"])
    for idx, value in enumerate(ALPHA_LOSS_VALUES):
        plt.plot(alpha_history[:, idx].tolist(), label=f"alpha={value:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# TRAIN ================================================================================================================

def main() -> None:
    if D_CHANNELS % NUM_HEADS != 0:
        raise ValueError("D_CHANNELS must be divisible by NUM_HEADS")

    set_seed(SEED)
    device = torch.device(DEVICE if DEVICE != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"MNIST RIFT: image={IMAGE_SIZE}px, d={D_CHANNELS}, heads={NUM_HEADS}, blocks={BLOCK_COUNT}, "
        f"pos_freq={POS_FREQ}, time_freq={TIME_FREQ}"
    )

    train_loader = DataLoader(
        MNISTImages(train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        MNISTImages(train=False),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(device)
    ema_model = copy.deepcopy(model).eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    conditioner = build_conditioner(device)
    print(f"RIFT parameters: {format_parameters(count_parameters(model))}")
    print(f"Conditioner parameters: {format_parameters(count_parameters(conditioner))}")
    model.print_model_summary()

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.parameters()), lr=LR)
    scheduler = make_cosine_with_warmup(optimizer, len(train_loader), EPOCHS * len(train_loader))

    history = {
        "train": [],
        "test": [],
        "test_by_size": {size: [] for size in TEST_SIZES},
        "alpha_losses": [],
    }
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        conditioner.train()
        total_train_loss = 0.0
        train_batches = 0

        for image, labels in tqdm(train_loader, total=len(train_loader), desc=f"train E{epoch + 1}"):
            if MAX_TRAIN_BATCHES is not None and train_batches >= MAX_TRAIN_BATCHES:
                break

            image = image.to(device).clamp(0.0, 1.0)
            labels = labels.to(device, dtype=torch.long)
            loss = rift_prediction_loss(model, conditioner, image, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(model, ema_model)

            total_train_loss += loss.item()
            train_batches += 1

        train_loss = total_train_loss / max(1, train_batches)
        test_by_size = evaluate(ema_model, conditioner, test_loader, device)
        test_loss = sum(test_by_size.values()) / max(1, len(test_by_size))
        alpha_losses = evaluate_by_alpha(ema_model, conditioner, test_loader, device)

        history["train"].append(train_loss)
        history["test"].append(test_loss)
        history["alpha_losses"].append(alpha_losses)
        for size, loss_value in test_by_size.items():
            history["test_by_size"][size].append(loss_value)

        print(f"Epoch {epoch + 1} | TRAIN: {train_loss:.5f} | TEST: {test_loss:.5f}")
        print("Test by size:", " | ".join(f"{size}px: {loss:.5f}" for size, loss in test_by_size.items()))
        print("Alpha losses:", " | ".join(
            f"alpha={alpha:.2f}: {loss:.5f}"
            for alpha, loss in zip(ALPHA_LOSS_VALUES, alpha_losses)
        ))

        plot_history(history, epoch)

        if SAMPLE_EVERY > 0 and (epoch + 1) % SAMPLE_EVERY == 0:
            render_samples(ema_model, conditioner, epoch, device)

        if SAVE_EVERY > 0 and (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(ema_model, conditioner, epoch, test_loss)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished training in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
