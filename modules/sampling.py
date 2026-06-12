from typing import Optional

import torch
from tqdm import tqdm

from modules.rift_diffusion import rift_velocity_step


@torch.no_grad()
def run_rift_sampling(
        model: torch.nn.Module,
        initial_noise: torch.Tensor,
        text_conditions: Optional[list[tuple[torch.Tensor, float | torch.Tensor]]] = None,
        num_steps: int = 20,
        step_size: float = 0.05,
        invert_steps: int = 0,
        evidence_scale: float = 1.0,
        device: Optional[torch.device] = None,
):
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if step_size < 0.0:
        raise ValueError(f"step_size must be non-negative, got {step_size}")
    if invert_steps < 0 or invert_steps > num_steps:
        raise ValueError(f"invert_steps must be in [0, num_steps], got {invert_steps}")

    device = device or initial_noise.device
    model = model.to(device)
    model.eval()

    x = initial_noise.to(device).clamp(0.0, 1.0)
    text_conditions = [] if text_conditions is None else [
        (tokens.to(device), strength.to(device) if torch.is_tensor(strength) else strength)
        for tokens, strength in text_conditions
    ]

    batch_size = x.shape[0]
    clean_hat = x

    for step in tqdm(range(num_steps), total=num_steps, desc="RIFT sampling"):
        t_batch = torch.zeros((batch_size,), device=device)
        model_input = x.clamp(0.0, 1.0)
        predicted_velocity = model(
            model_input,
            t_batch,
            text_conditions,
            evidence_scale=evidence_scale,
        )[0]
        signed_step_size = -float(step_size) if step < invert_steps else float(step_size)
        x = rift_velocity_step(x, predicted_velocity, step_size=signed_step_size)
        clean_hat = x

    return clean_hat, x
