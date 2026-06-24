import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.conditioning import ClassLabelConditioner
from modules.inference import build_model_from_config, infer_sidecar_path, load_checkpoint_config, newest_checkpoint_path
from modules.rift import ImageFFN, RIFT
from modules.rift_diffusion import rift_training_pair, velocity_mse_loss

# CONFIG ===============================================================================================================

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = None
CONDITIONER_PATH = None
CONFIG_PATH = None

DATA_ROOT = PROJECT_ROOT / "data"
DEVICE = "cuda"
SEED = 0

BATCH_SIZE = 8
IMAGE_CHANNELS = 3
ANALYSIS_SIZES = (14, 28, 64, 128)
ALPHA = 0.5
CFG_SCALE_FOR_DELTA = 4.0
FINAL_BLOCK_MAP_SIZE = 64
FINAL_BLOCK_MAP_COUNT = 8
RENDER_FINAL_BLOCK_MAPS = True
RENDER_RMS_FINAL_BLOCK_MAPS = True
RENDER_SIGNED_FINAL_BLOCK_MAPS = True
RENDER_COHERENCE_FINAL_BLOCK_MAPS = True
SIGNED_MAP_QUANTILE = 0.99
SIGNED_MAP_CMAP = "bwr"
COHERENCE_MAP_CMAP = "magma"
SAVE_FIGURES = False
FIGURE_DIR = Path("MNIST/media/adaptive_ffn")


# DATA =================================================================================================================

class MNISTImages(torch.utils.data.Dataset):
    def __init__(self, image_size: int):
        self.dataset = datasets.MNIST(
            root=str(DATA_ROOT),
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.Grayscale(num_output_channels=IMAGE_CHANNELS),
                transforms.ToTensor(),
            ]),
        )

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset)


# HELPERS ==============================================================================================================

def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.square().mean().sqrt()


def spatial_fraction(tensor: torch.Tensor) -> float:
    total = rms(tensor).item()
    if total <= 1e-12:
        return 0.0
    centered = tensor - tensor.mean(dim=(-2, -1), keepdim=True)
    return rms(centered).item() / total


def decompose_tensor(tensor: torch.Tensor) -> dict[str, float]:
    total = rms(tensor).item()
    if total <= 1e-12:
        return {
            "rms": 0.0,
            "global_frac": 0.0,
            "coord_frac": 0.0,
            "content_frac": 0.0,
            "spatial_frac": 0.0,
        }

    global_component = tensor.mean(dim=(0, 2, 3), keepdim=True)
    coord_component = tensor.mean(dim=0, keepdim=True) - global_component
    content_component = tensor - tensor.mean(dim=0, keepdim=True)
    spatial_component = tensor - tensor.mean(dim=(-2, -1), keepdim=True)
    return {
        "rms": total,
        "global_frac": rms(global_component).item() / total,
        "coord_frac": rms(coord_component).item() / total,
        "content_frac": rms(content_component).item() / total,
        "spatial_frac": rms(spatial_component).item() / total,
    }


def relative_rms_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = 0.5 * (rms(a).item() + rms(b).item())
    if denominator <= 1e-12:
        return 0.0
    return rms(a - b).item() / denominator


def channel_sign_coherence(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    channel_mean = tensor.mean(dim=1, keepdim=True)
    channel_rms = tensor.square().mean(dim=1, keepdim=True).sqrt()
    return (channel_mean.abs() / channel_rms.clamp_min(eps)).clamp(0.0, 1.0)


def mean_channel_sign_coherence(tensor: torch.Tensor) -> float:
    return channel_sign_coherence(tensor).mean().item()


def mean_or_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def set_adaptive_enabled(model: RIFT, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, ImageFFN):
            module.adaptive_enabled = enabled


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


def load_batch(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dataloader = DataLoader(MNISTImages(size), batch_size=BATCH_SIZE, shuffle=False)
    image, labels = next(iter(dataloader))
    return image.to(device).clamp(0.0, 1.0), labels.to(device, dtype=torch.long)


def make_conditioning(conditioner: ClassLabelConditioner, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positive_tokens = conditioner(labels)
    negative_tokens = conditioner(torch.full_like(labels, conditioner.num_classes))
    return negative_tokens, positive_tokens


@torch.no_grad()
def prediction_metrics(
        model: RIFT,
        conditioner: ClassLabelConditioner,
        image: torch.Tensor,
        labels: torch.Tensor,
        adaptive_enabled: bool,
) -> dict[str, torch.Tensor | float]:
    set_adaptive_enabled(model, adaptive_enabled)
    alpha = torch.full((image.shape[0],), ALPHA, device=image.device, dtype=image.dtype)
    model_input, target_velocity, _, alpha = rift_training_pair(image, alpha=alpha)
    model_time = 1.0 - alpha
    negative_tokens, positive_tokens = make_conditioning(conditioner, labels)
    predicted_null, predicted_pos = model(model_input, model_time, [negative_tokens, positive_tokens])
    predicted_cfg = predicted_null + CFG_SCALE_FOR_DELTA * (predicted_pos - predicted_null)

    null_loss = velocity_mse_loss(predicted_null, target_velocity).item()
    pos_loss = velocity_mse_loss(predicted_pos, target_velocity).item()
    cfg_loss = velocity_mse_loss(predicted_cfg, target_velocity).item()
    return {
        "model_input": model_input,
        "model_time": model_time,
        "negative_tokens": negative_tokens,
        "positive_tokens": positive_tokens,
        "target_velocity": target_velocity,
        "predicted_null": predicted_null,
        "predicted_pos": predicted_pos,
        "predicted_cfg": predicted_cfg,
        "loss": 0.5 * (null_loss + pos_loss),
        "null_loss": null_loss,
        "pos_loss": pos_loss,
        "cfg_loss": cfg_loss,
    }


def stat_row(name: str, branch: str, expanded: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> dict[str, float | str]:
    delta = expanded * gamma + beta
    expanded_rms = rms(expanded).item()
    delta_rms = rms(delta).item()
    return {
        "name": name,
        "branch": branch,
        "expanded_rms": expanded_rms,
        "gamma_abs": gamma.abs().mean().item(),
        "gamma_rms": rms(gamma).item(),
        "gamma_spatial_frac": spatial_fraction(gamma),
        "gamma_sign_coherence": mean_channel_sign_coherence(gamma),
        "beta_abs": beta.abs().mean().item(),
        "beta_rms": rms(beta).item(),
        "beta_spatial_frac": spatial_fraction(beta),
        "beta_sign_coherence": mean_channel_sign_coherence(beta),
        "delta_rms": delta_rms,
        "delta_to_expanded": delta_rms / max(expanded_rms, 1e-12),
        "delta_spatial_frac": spatial_fraction(delta),
        "delta_sign_coherence": mean_channel_sign_coherence(delta),
    }


@torch.no_grad()
def collect_adaptive_stats(
        model: RIFT,
        model_input: torch.Tensor,
        model_time: torch.Tensor,
        negative_tokens: torch.Tensor,
        positive_tokens: torch.Tensor,
) -> list[dict[str, float | str]]:
    rows = []
    handles = []
    call_counts: dict[str, int] = {}

    def make_hook(name: str):
        def hook(module: ImageFFN, inputs: tuple[torch.Tensor], _output: torch.Tensor) -> None:
            x = inputs[0].detach()
            expanded = module.expand(x)
            gamma = module.to_gamma(x)
            beta = module.to_beta(x)
            call_index = call_counts.get(name, 0)
            call_counts[name] = call_index + 1
            branch = "null" if call_index == 0 else "pos" if call_index == 1 else f"call{call_index}"
            rows.append(stat_row(name, branch, expanded, gamma, beta))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, ImageFFN):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        set_adaptive_enabled(model, True)
        model(model_input, model_time, [negative_tokens, positive_tokens])
    finally:
        for handle in handles:
            handle.remove()

    return rows


def aggregate_stats(rows: list[dict[str, float | str]]) -> dict[str, float]:
    keys = [
        "expanded_rms",
        "gamma_abs",
        "gamma_rms",
        "gamma_spatial_frac",
        "gamma_sign_coherence",
        "beta_abs",
        "beta_rms",
        "beta_spatial_frac",
        "beta_sign_coherence",
        "delta_rms",
        "delta_to_expanded",
        "delta_spatial_frac",
        "delta_sign_coherence",
    ]
    return {key: mean_or_nan([float(row[key]) for row in rows]) for key in keys}


@torch.no_grad()
def capture_final_block_adaptation(
        model: RIFT,
        model_input: torch.Tensor,
        model_time: torch.Tensor,
        text_conditions: list[torch.Tensor],
        branch_names: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    final_ffn = model.dec_blocks[-1].ffn
    records: list[dict[str, torch.Tensor]] = []

    def hook(module: ImageFFN, inputs: tuple[torch.Tensor], _output: torch.Tensor) -> None:
        x = inputs[0].detach()
        expanded = module.expand(x)
        gamma = module.to_gamma(x)
        beta = module.to_beta(x)
        delta = expanded * gamma + beta
        records.append({
            "input": x.detach(),
            "expanded": expanded.detach(),
            "gamma": gamma.detach(),
            "beta": beta.detach(),
            "delta": delta.detach(),
        })

    set_adaptive_enabled(model, True)
    handle = final_ffn.register_forward_hook(hook)
    try:
        model(model_input, model_time, text_conditions)
    finally:
        handle.remove()

    if len(records) != len(branch_names):
        raise RuntimeError(f"Expected {len(branch_names)} final FFN calls, got {len(records)}")
    return {branch: record for branch, record in zip(branch_names, records)}


def print_decomposition(prefix: str, tensor: torch.Tensor) -> None:
    parts = decompose_tensor(tensor)
    print(
        f"    {prefix}: rms={parts['rms']:.6f} "
        f"global={parts['global_frac']:.3f} "
        f"coord={parts['coord_frac']:.3f} "
        f"content={parts['content_frac']:.3f} "
        f"spatial={parts['spatial_frac']:.3f}"
    )


def print_sign_coherence(prefix: str, record: dict[str, torch.Tensor]) -> None:
    print(
        f"    {prefix}: "
        f"gamma={mean_channel_sign_coherence(record['gamma']):.3f}, "
        f"beta={mean_channel_sign_coherence(record['beta']):.3f}, "
        f"delta={mean_channel_sign_coherence(record['delta']):.3f}"
    )


def print_final_block_report(
        size: int,
        true_records: dict[str, dict[str, torch.Tensor]],
        fixed_prompt_records: dict[str, dict[str, torch.Tensor]],
        prompt_records: dict[str, dict[str, torch.Tensor]],
) -> None:
    print("  final block direct decomposition")
    print("  fractions are relative RMS: global constant / coordinate-shared / per-image-content / spatial")
    for branch in ("null", "pos"):
        print(f"  branch={branch}")
        print_decomposition("gamma", true_records[branch]["gamma"])
        print_decomposition("beta ", true_records[branch]["beta"])
        print_decomposition("delta", true_records[branch]["delta"])
        print_sign_coherence("sign coherence abs(mean)/rms", true_records[branch])

    print("  fixed-prompt content test")
    print_decomposition("gamma", fixed_prompt_records["fixed_label_0"]["gamma"])
    print_decomposition("beta ", fixed_prompt_records["fixed_label_0"]["beta"])
    print_decomposition("delta", fixed_prompt_records["fixed_label_0"]["delta"])
    print_sign_coherence("sign coherence abs(mean)/rms", fixed_prompt_records["fixed_label_0"])

    gamma_prompt_delta = relative_rms_delta(
        prompt_records["label_0"]["gamma"],
        prompt_records["label_1"]["gamma"],
    )
    beta_prompt_delta = relative_rms_delta(
        prompt_records["label_0"]["beta"],
        prompt_records["label_1"]["beta"],
    )
    delta_prompt_delta = relative_rms_delta(
        prompt_records["label_0"]["delta"],
        prompt_records["label_1"]["delta"],
    )
    print(
        "  prompt swap relative delta, same x_t: "
        f"gamma={gamma_prompt_delta:.3f}, beta={beta_prompt_delta:.3f}, delta={delta_prompt_delta:.3f}"
    )


def maybe_savefig(name: str) -> None:
    if not SAVE_FIGURES:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=160)


def normalize_map(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    low = tensor.amin()
    high = torch.quantile(tensor.flatten(), 0.99)
    if (high - low).abs().item() <= 1e-12:
        return torch.zeros_like(tensor)
    return ((tensor - low) / (high - low)).clamp(0.0, 1.0)


def normalize_signed_map(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    scale = torch.quantile(tensor.abs().flatten(), SIGNED_MAP_QUANTILE)
    if scale.item() <= 1e-12:
        return torch.zeros_like(tensor)
    return (tensor / scale).clamp(-1.0, 1.0)


def channel_rms_map(tensor: torch.Tensor, count: int) -> torch.Tensor:
    return normalize_map(tensor[:count].square().mean(dim=1, keepdim=True).sqrt())


def channel_signed_mean_map(tensor: torch.Tensor, count: int) -> torch.Tensor:
    return normalize_signed_map(tensor[:count].mean(dim=1, keepdim=True))


def channel_sign_coherence_map(tensor: torch.Tensor, count: int) -> torch.Tensor:
    return channel_sign_coherence(tensor[:count]).detach().cpu()


def show_tensor_grid(
        tensor: torch.Tensor,
        title: str,
        name: str,
        cmap: str = "gray",
        vmin: float = 0.0,
        vmax: float = 1.0,
        clamp: bool = True,
) -> None:
    tensor = tensor.detach().cpu()
    if clamp:
        tensor = tensor.clamp(vmin, vmax)
    batch, channels, _, _ = tensor.shape
    cols = min(batch, 4)
    rows = math.ceil(batch / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0), squeeze=False)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= batch:
            continue
        image = tensor[idx]
        if channels == 1:
            ax.imshow(image.squeeze(0), cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(image.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        ax.set_title(str(idx), fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    maybe_savefig(name)
    plt.show()


def render_final_block_maps(size: int, image: torch.Tensor, model_input: torch.Tensor,
                            records: dict[str, dict[str, torch.Tensor]]) -> None:
    if not RENDER_FINAL_BLOCK_MAPS or size != FINAL_BLOCK_MAP_SIZE:
        return

    count = min(FINAL_BLOCK_MAP_COUNT, image.shape[0])
    show_tensor_grid(image[:count], f"clean images | {size}px", f"clean_{size}")
    show_tensor_grid(model_input[:count], f"flow inputs x_t | alpha={ALPHA:.2f} | {size}px", f"xt_{size}")

    for branch in ("null", "pos"):
        for tensor_name, title_name in (
                ("gamma", "gamma"),
                ("beta", "beta"),
                ("delta", "adaptive delta"),
        ):
            tensor = records[branch][tensor_name]
            if RENDER_RMS_FINAL_BLOCK_MAPS:
                show_tensor_grid(
                    channel_rms_map(tensor, count),
                    f"final FFN {title_name} RMS map | branch={branch} | {size}px",
                    f"final_{tensor_name}_rms_{branch}_{size}",
                    cmap="magma",
                )
            if RENDER_SIGNED_FINAL_BLOCK_MAPS:
                show_tensor_grid(
                    channel_signed_mean_map(tensor, count),
                    f"final FFN {title_name} signed mean map | branch={branch} | {size}px",
                    f"final_{tensor_name}_signed_mean_{branch}_{size}",
                    cmap=SIGNED_MAP_CMAP,
                    vmin=-1.0,
                    vmax=1.0,
                    clamp=False,
                )
            if RENDER_COHERENCE_FINAL_BLOCK_MAPS:
                show_tensor_grid(
                    channel_sign_coherence_map(tensor, count),
                    f"final FFN {title_name} sign coherence map | branch={branch} | {size}px",
                    f"final_{tensor_name}_sign_coherence_{branch}_{size}",
                    cmap=COHERENCE_MAP_CMAP,
                )


def print_size_report(size: int, full: dict, disabled: dict, stats: list[dict[str, float | str]]) -> None:
    aggregate = aggregate_stats(stats)
    target_rms = rms(full["target_velocity"]).item()
    pos_delta_rms = rms(full["predicted_pos"] - disabled["predicted_pos"]).item()
    null_delta_rms = rms(full["predicted_null"] - disabled["predicted_null"]).item()
    cfg_delta_rms = rms(full["predicted_cfg"] - disabled["predicted_cfg"]).item()
    loss_delta = float(disabled["loss"]) - float(full["loss"])
    cfg_loss_delta = float(disabled["cfg_loss"]) - float(full["cfg_loss"])

    print(f"\nsize={size}px alpha={ALPHA:.2f}")
    print(
        "  loss full/off/delta: "
        f"{float(full['loss']):.6f} / {float(disabled['loss']):.6f} / {loss_delta:+.6f}"
    )
    print(
        f"  cfg{CFG_SCALE_FOR_DELTA:g} loss full/off/delta: "
        f"{float(full['cfg_loss']):.6f} / {float(disabled['cfg_loss']):.6f} / {cfg_loss_delta:+.6f}"
    )
    print(
        "  output delta rms vs target rms: "
        f"null={null_delta_rms:.6f}, pos={pos_delta_rms:.6f}, "
        f"cfg{CFG_SCALE_FOR_DELTA:g}={cfg_delta_rms:.6f}, target={target_rms:.6f}"
    )
    print(
        "  adaptive magnitude: "
        f"delta/expanded={aggregate['delta_to_expanded']:.6f}, "
        f"gamma_rms={aggregate['gamma_rms']:.6f}, beta_rms={aggregate['beta_rms']:.6f}"
    )
    print(
        "  spatial dependence fraction: "
        f"gamma={aggregate['gamma_spatial_frac']:.3f}, "
        f"beta={aggregate['beta_spatial_frac']:.3f}, "
        f"delta={aggregate['delta_spatial_frac']:.3f}"
    )
    print(
        "  channel sign coherence abs(mean)/rms: "
        f"gamma={aggregate['gamma_sign_coherence']:.3f}, "
        f"beta={aggregate['beta_sign_coherence']:.3f}, "
        f"delta={aggregate['delta_sign_coherence']:.3f}"
    )


# MAIN =================================================================================================================

def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE if DEVICE != "cuda" or torch.cuda.is_available() else "cpu")
    model, conditioner, _ = load_checkpoint(device)

    print("\nInterpretation:")
    print("  useful ablation signal: disabling adaptive FFN increases loss and changes velocity outputs")
    print("  useful input-dependence signal: spatial dependence fractions are clearly above 0")
    print("  final-block decomposition: high content fraction means same coordinate differs across images")
    print("  prompt swap delta: high value means same x_t gets different adaptation under different prompts")
    print("  sign coherence: low abs(channel mean)/channel RMS means strong hidden work cancels in signed averages")

    for size in ANALYSIS_SIZES:
        image, labels = load_batch(size, device)
        full = prediction_metrics(model, conditioner, image, labels, adaptive_enabled=True)
        disabled = prediction_metrics(model, conditioner, image, labels, adaptive_enabled=False)
        stats = collect_adaptive_stats(
            model,
            full["model_input"],
            full["model_time"],
            full["negative_tokens"],
            full["positive_tokens"],
        )
        print_size_report(size, full, disabled, stats)

        true_records = capture_final_block_adaptation(
            model,
            full["model_input"],
            full["model_time"],
            [full["negative_tokens"], full["positive_tokens"]],
            ["null", "pos"],
        )
        fixed_label_0 = conditioner(torch.zeros_like(labels))
        fixed_prompt_records = capture_final_block_adaptation(
            model,
            full["model_input"],
            full["model_time"],
            [fixed_label_0],
            ["fixed_label_0"],
        )
        label_1 = conditioner(torch.ones_like(labels))
        prompt_records = capture_final_block_adaptation(
            model,
            full["model_input"],
            full["model_time"],
            [fixed_label_0, label_1],
            ["label_0", "label_1"],
        )
        print_final_block_report(size, true_records, fixed_prompt_records, prompt_records)
        render_final_block_maps(size, image, full["model_input"], true_records)

    set_adaptive_enabled(model, True)


if __name__ == "__main__":
    main()
