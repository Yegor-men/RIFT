from typing import Optional

import torch


def expand_batch_value(value: torch.Tensor | float, target: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.tensor(float(value), device=target.device, dtype=target.dtype)
    value = value.to(device=target.device, dtype=target.dtype)
    if value.ndim == 0:
        return value.view(*([1] * target.ndim)).expand_as(target)
    if value.ndim == target.ndim:
        return value
    if value.ndim == 1:
        return value.view(value.shape[0], *([1] * (target.ndim - 1))).expand_as(target)
    raise ValueError(f"Cannot expand value with shape {tuple(value.shape)} to target shape {tuple(target.shape)}")


def sample_noise_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(tensor)


def sample_noise(
        shape: tuple[int, ...],
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=dtype)


def sample_alpha(
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.rand((batch_size,), device=device, dtype=dtype)


def rift_training_pair(
        clean: torch.Tensor,
        alpha: Optional[torch.Tensor | float] = None,
        noise: Optional[torch.Tensor] = None,
        detach_input: bool = True,
):
    """Build x_t and target velocity for flow matching.

    clean is x1, noise is x0, alpha is the corruption level, and t = 1 - alpha.
    The model input is x_t = (1 - alpha) * x1 + alpha * x0.
    The training target is the constant path velocity v = x1 - x0.
    """
    clean = clean.clamp(0.0, 1.0)
    batch_size = clean.shape[0]

    if noise is None:
        noise = sample_noise_like(clean)
    else:
        noise = noise.to(device=clean.device, dtype=clean.dtype).clamp(0.0, 1.0)

    if alpha is None:
        alpha = sample_alpha(batch_size, clean.device, clean.dtype)
    elif not torch.is_tensor(alpha):
        alpha = torch.full((batch_size,), float(alpha), device=clean.device, dtype=clean.dtype)
    else:
        alpha = alpha.to(device=clean.device, dtype=clean.dtype).flatten()
        if alpha.numel() == 1:
            alpha = alpha.expand(batch_size)
        if alpha.numel() != batch_size:
            raise ValueError(f"alpha must have {batch_size} values, got {alpha.numel()}")

    alpha = alpha.clamp(0.0, 1.0)
    alpha_map = alpha.view(batch_size, 1, 1, 1)
    model_input = ((1.0 - alpha_map) * clean + alpha_map * noise).clamp(0.0, 1.0)
    target_velocity = noise.neg().add(clean)

    if detach_input:
        model_input = model_input.detach()

    return model_input, target_velocity, noise, alpha


def velocity_mse_loss(
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        alpha: torch.Tensor | None = None,
        max_weight: float | None = None,
) -> torch.Tensor:
    del alpha, max_weight
    return torch.nn.functional.mse_loss(predicted_velocity, target_velocity)


def weighted_velocity_mse_loss(
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        alpha: torch.Tensor | None = None,
        max_weight: float | None = None,
) -> torch.Tensor:
    return velocity_mse_loss(predicted_velocity, target_velocity, alpha, max_weight)


def rift_velocity_step(
        current: torch.Tensor,
        predicted_velocity: torch.Tensor,
        step_size: float = 0.05,
) -> torch.Tensor:
    """Euler step along the predicted x1-x0 flow velocity in [0, 1] space."""
    current = current.clamp(0.0, 1.0)
    predicted_velocity = predicted_velocity.to(device=current.device, dtype=current.dtype).clamp(-1.0, 1.0)
    return (current + float(step_size) * predicted_velocity).clamp(0.0, 1.0)
