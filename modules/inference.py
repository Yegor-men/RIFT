import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from modules.conditioning import ClassLabelConditioner
from modules.r2id import R2ID
from modules.render_image import render_image
from modules.sampling import run_velocity_sampling
from modules.velocity import sample_noise


def infer_sidecar_path(model_path: Path, suffix: str) -> Path:
    stem = model_path.name.removesuffix("_r2id.safetensors")
    return model_path.with_name(f"{stem}_{suffix}")


def newest_checkpoint_path(model_dir: str | Path, dataset_name: str | None = None) -> Path:
    model_dir = Path(model_dir)
    pattern = f"{dataset_name}_E*_r2id.safetensors" if dataset_name else "*_E*_r2id.safetensors"
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
    return model, conditioner, model_config


def parse_label_list(labels: str, count: int, num_classes: int, device: torch.device) -> torch.Tensor:
    if labels.strip().lower() == "grid":
        base = torch.arange(min(num_classes, count), dtype=torch.long, device=device)
    else:
        base = torch.tensor([int(item.strip()) for item in labels.split(",") if item.strip()], dtype=torch.long, device=device)
    if base.numel() >= count:
        return base[:count]
    repeats = (count + base.numel() - 1) // base.numel()
    return base.repeat(repeats)[:count]


@torch.no_grad()
def render_checkpoint_samples(
        model_path: str | Path | None,
        title_prefix: str,
        model_dir: str | Path | None = None,
        dataset_name: str | None = None,
        conditioner_path: str | Path | None = None,
        config_path: str | Path | None = None,
        sizes: tuple[int, ...] = (28, 64, 128),
        labels: str = "grid",
        batch_size: int = 100,
        sample_steps: int = 100,
        cfg_scale: float = 1.0,
        device: str = "cuda",
        save: bool = False,
) -> None:
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
    pos_tokens = conditioner(sample_labels)
    null_tokens = None
    if float(cfg_scale) != 1.0:
        null_tokens = conditioner(torch.full_like(sample_labels, conditioner.null_label))

    for size in sizes:
        initial_noise = sample_noise(
            (sample_labels.shape[0], model_config["image_channels"], size, size),
            device=device_obj,
        )
        samples, _ = run_velocity_sampling(
            model=model,
            initial_noise=initial_noise,
            pos_text_cond=pos_tokens,
            null_text_cond=null_tokens,
            num_steps=sample_steps,
            cfg_scale=cfg_scale,
            device=device_obj,
        )
        render_image(
            samples.clamp(0.0, 1.0),
            title=f"{title_prefix} R2ID | {size}px",
            name=f"{title_prefix.lower()}_r2id_{size}px",
            save=save,
        )
