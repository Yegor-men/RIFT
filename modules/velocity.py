from typing import Optional

import torch


def expand_time(t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    t = t.to(device=target.device, dtype=target.dtype)
    return t.view(t.shape[0], *([1] * (target.ndim - 1)))


def sample_noise_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(tensor)


def sample_noise(
        shape: tuple[int, ...],
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=dtype)


def velocity_training_pair(
        clean: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        detach_input: bool = True,
):
    clean = clean.clamp(0.0, 1.0)
    if noise is None:
        noise = sample_noise_like(clean)
    else:
        noise = noise.to(device=clean.device, dtype=clean.dtype).clamp(0.0, 1.0)

    t_expanded = expand_time(t, clean)
    model_input = ((1.0 - t_expanded) * noise + t_expanded * clean).clamp(0.0, 1.0)
    velocity = clean - noise
    if detach_input:
        model_input = model_input.detach()
    return model_input, velocity, noise


def guided_velocity(v_null: torch.Tensor, v_pos: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    return v_null + float(cfg_scale) * (v_pos - v_null)


def velocity_step(
        current: torch.Tensor,
        t_current: torch.Tensor,
        t_next: torch.Tensor,
        predicted_velocity: torch.Tensor,
):
    t_current = expand_time(t_current, current)
    t_next = expand_time(t_next, current)
    next_x = (current + (t_next - t_current) * predicted_velocity).clamp(0.0, 1.0)
    x1_hat = (current + (1.0 - t_current) * predicted_velocity).clamp(0.0, 1.0)
    return next_x, x1_hat
