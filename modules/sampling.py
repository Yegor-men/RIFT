from typing import Optional

import torch
from tqdm import tqdm

from modules.rift_diffusion import rift_velocity_step


@torch.no_grad()
def predict_cfg_velocity(
        model: torch.nn.Module,
        image: torch.Tensor,
        time: torch.Tensor,
        positive_text_conditioning: torch.Tensor | None,
        negative_text_conditioning: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
) -> torch.Tensor:
    if positive_text_conditioning is None:
        return model(image, time, None)[0]

    if negative_text_conditioning is None:
        return model(image, time, positive_text_conditioning)[0]

    negative_text_conditioning = negative_text_conditioning.to(device=image.device, dtype=image.dtype)
    positive_text_conditioning = positive_text_conditioning.to(device=image.device, dtype=image.dtype)
    velocity_null, velocity_pos = model(
        image,
        time,
        [negative_text_conditioning, positive_text_conditioning],
    )
    return velocity_null + float(cfg_scale) * (velocity_pos - velocity_null)


@torch.no_grad()
def run_rift_sampling(
        model: torch.nn.Module,
        initial_noise: torch.Tensor,
        positive_text_conditioning: torch.Tensor | None = None,
        negative_text_conditioning: torch.Tensor | None = None,
        num_steps: int = 20,
        step_size: float | None = None,
        cfg_scale: float = 1.0,
        device: Optional[torch.device] = None,
        return_trace: bool = False,
        text_conditions: Optional[list[tuple[torch.Tensor, float | torch.Tensor]] | list[torch.Tensor]] = None,
        invert_steps: int = 0,
):
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if step_size is None:
        step_size = 1.0 / float(num_steps)
    if step_size < 0.0:
        raise ValueError(f"step_size must be non-negative, got {step_size}")
    if invert_steps < 0 or invert_steps > num_steps:
        raise ValueError(f"invert_steps must be in [0, num_steps], got {invert_steps}")

    if text_conditions is not None and positive_text_conditioning is None:
        first = text_conditions[0]
        positive_text_conditioning = first[0] if isinstance(first, tuple) else first

    device = device or initial_noise.device
    model = model.to(device)
    model.eval()

    x = initial_noise.to(device).clamp(0.0, 1.0)
    if positive_text_conditioning is not None:
        positive_text_conditioning = positive_text_conditioning.to(device)
    if negative_text_conditioning is not None:
        negative_text_conditioning = negative_text_conditioning.to(device)

    batch_size = x.shape[0]
    image_trace = [x.detach().cpu()] if return_trace else None
    velocity_trace = [] if return_trace else None
    time_trace = [] if return_trace else None

    current_time = 0.0
    for step in tqdm(range(num_steps), total=num_steps, desc="RIFT sampling"):
        signed_step_size = -float(step_size) if step < invert_steps else float(step_size)
        time_value = min(1.0, max(0.0, current_time))
        time_batch = torch.full((batch_size,), time_value, device=device, dtype=x.dtype)
        predicted_velocity = predict_cfg_velocity(
            model=model,
            image=x.clamp(0.0, 1.0),
            time=time_batch,
            positive_text_conditioning=positive_text_conditioning,
            negative_text_conditioning=negative_text_conditioning,
            cfg_scale=cfg_scale,
        )
        x = rift_velocity_step(x, predicted_velocity, step_size=signed_step_size)
        current_time += signed_step_size

        if return_trace:
            image_trace.append(x.detach().cpu())
            velocity_trace.append(predicted_velocity.detach().cpu())
            time_trace.append(time_value)

    if return_trace:
        return x, x, image_trace, velocity_trace, time_trace
    return x, x
