import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from modules.conditioning import ClassLabelConditioner
from modules.render_image import render_image
from modules.rift import RIFT
from modules.rift_diffusion import sample_noise
from modules.sampling import run_rift_sampling


def infer_sidecar_path(model_path: Path, suffix: str) -> Path:
    stem = model_path.name.removesuffix("_rift.safetensors")
    return model_path.with_name(f"{stem}_{suffix}")


def newest_checkpoint_path(model_dir: str | Path, dataset_name: str | None = None) -> Path:
    model_dir = Path(model_dir)
    pattern = f"{dataset_name}_E*_rift.safetensors" if dataset_name else "*_E*_rift.safetensors"
    candidates = sorted(model_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern!r} found in {model_dir}")
    return candidates[-1]


def load_checkpoint_config(model_path: str | Path, config_path: str | Path | None = None) -> dict:
    model_path = Path(model_path)
    path = Path(config_path) if config_path else infer_sidecar_path(model_path, "config.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model_from_config(config: dict, device: torch.device):
    model_config = config["model"]
    model = RIFT(
        c_channels=model_config["image_channels"],
        d_channels=model_config["d_channels"],
        num_heads=model_config["num_heads"],
        block_count=model_config["block_count"],
        pos_freq=model_config["pos_freq"],
        time_freq=model_config["time_freq"],
        linear_attention=model_config.get("linear_attention", True),
    ).to(device)
    conditioner = ClassLabelConditioner(
        num_classes=model_config["num_classes"],
        token_count=model_config["token_count"],
        d_channels=model_config["d_channels"],
    ).to(device)
    return model, conditioner, model_config


def parse_label_list(labels: str, count: int, num_classes: int, device: torch.device) -> torch.Tensor:
    if labels.strip().lower() == "grid":
        base = torch.arange(min(num_classes, count), dtype=torch.long, device=device)
    else:
        values = [int(item.strip()) for item in labels.split(",") if item.strip()]
        if not values:
            raise ValueError("labels must be 'grid' or a comma-separated label list")
        base = torch.tensor(values, dtype=torch.long, device=device)
    if base.numel() >= count:
        return base[:count]
    repeats = (count + base.numel() - 1) // base.numel()
    return base.repeat(repeats)[:count]


def normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"size tuples must be (height, width), got {size}")
    return int(size[0]), int(size[1])


@torch.no_grad()
def render_checkpoint_samples(
        model_path: str | Path | None,
        title_prefix: str,
        model_dir: str | Path | None = None,
        dataset_name: str | None = None,
        conditioner_path: str | Path | None = None,
        config_path: str | Path | None = None,
        sizes: tuple[int | tuple[int, int], ...] = ((128, 128), (160, 160), (192, 192)),
        labels: str = "grid",
        batch_size: int = 100,
        sample_steps: int = 20,
        step_size: float = 0.05,
        cfg_scale: float = 4.0,
        condition_strength: float | None = None,
        device: str = "cuda",
        save: bool = False,
) -> None:
    if condition_strength is not None:
        cfg_scale = float(condition_strength)

    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if model_path is None:
        if model_dir is None:
            raise ValueError("model_dir must be provided when model_path is None")
        model_path = newest_checkpoint_path(model_dir, dataset_name or title_prefix)
    else:
        model_path = Path(model_path)
    conditioner_path = Path(conditioner_path) if conditioner_path else infer_sidecar_path(model_path, "conditioner.safetensors")
    config = load_checkpoint_config(model_path, config_path)

    model, conditioner, model_config = build_model_from_config(config, device_obj)
    model.load_state_dict(load_file(str(model_path)))
    conditioner.load_state_dict(load_file(str(conditioner_path)))
    model.eval()
    conditioner.eval()
    print(f"Loaded model: {model_path}")
    print(f"Loaded conditioner: {conditioner_path}")

    sample_labels = parse_label_list(labels, batch_size, model_config["num_classes"], device_obj)
    positive_tokens = conditioner(sample_labels)
    negative_tokens = conditioner(torch.full_like(sample_labels, model_config["num_classes"]))

    for size in sizes:
        height, width = normalize_size(size)
        initial_noise = sample_noise(
            (sample_labels.shape[0], model_config["image_channels"], height, width),
            device=device_obj,
        )
        samples, _ = run_rift_sampling(
            model=model,
            initial_noise=initial_noise,
            positive_text_conditioning=positive_tokens,
            negative_text_conditioning=negative_tokens,
            num_steps=sample_steps,
            step_size=step_size,
            cfg_scale=cfg_scale,
            device=device_obj,
        )
        render_image(
            samples,
            title=f"{title_prefix} pixel RIFT | {height}x{width}",
            name=f"{title_prefix.lower()}_pixel_rift_{height}x{width}",
            save=save,
        )
