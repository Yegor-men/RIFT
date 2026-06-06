from typing import Optional

import torch


def expand_time(t: torch.Tensor | float, target: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=target.device, dtype=target.dtype)
    t = t.to(device=target.device, dtype=target.dtype)
    if t.ndim == 0:
        return t.view(*([1] * target.ndim)).expand_as(target)
    if t.ndim == target.ndim:
        return t
    if t.ndim == 1:
        return t.view(t.shape[0], *([1] * (target.ndim - 1))).expand_as(target)
    raise ValueError(f"Cannot expand t with shape {tuple(t.shape)} to target shape {tuple(target.shape)}")


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
        t: Optional[torch.Tensor | float] = None,
        noise: Optional[torch.Tensor] = None,
        detach_input: bool = True,
):
    """Build x_t and the residual target v = clean - x_t.

    t is diffusion progress / cleanliness: t=0 is pure uniform noise and t=1 is
    clean. alpha = 1 - t is the noise weight.

    If t is omitted, each image gets a global alpha and each pixel/channel gets
    a local alpha. Multiplying them makes the image clean overall while retaining
    nonuniform local corruption.
    """
    clean = clean.clamp(0.0, 1.0)
    if noise is None:
        noise = sample_noise_like(clean)
    else:
        noise = noise.to(device=clean.device, dtype=clean.dtype).clamp(0.0, 1.0)

    if t is None:
        per_image_alpha = torch.rand(
            (clean.shape[0], *([1] * (clean.ndim - 1))),
            device=clean.device,
            dtype=clean.dtype,
        )
        per_pixel_alpha = torch.rand_like(clean)
        alpha = per_image_alpha * per_pixel_alpha
    else:
        t_expanded = expand_time(t, clean).clamp(0.0, 1.0)
        alpha = 1.0 - t_expanded

    model_input = ((1.0 - alpha) * clean + alpha * noise).clamp(0.0, 1.0)
    velocity = clean - model_input
    if detach_input:
        model_input = model_input.detach()
    return model_input, velocity, noise


def guided_velocity(v_null: torch.Tensor, v_pos: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    return v_null + float(cfg_scale) * (v_pos - v_null)


def scheduled_residual_step(
        current: torch.Tensor,
        predicted_velocity: torch.Tensor,
        t_current: torch.Tensor | float,
        t_next: torch.Tensor | float,
        schedule_scale: float = 1.0,
) -> torch.Tensor:
    """Advance linear cleanliness t when the model predicts clean - current."""
    t_current = expand_time(t_current, current)
    t_next = expand_time(t_next, current)
    remaining = (1.0 - t_current).clamp_min(1e-6)
    residual_scale = (t_next - t_current) / remaining
    return (current + float(schedule_scale) * residual_scale * predicted_velocity).clamp(0.0, 1.0)
