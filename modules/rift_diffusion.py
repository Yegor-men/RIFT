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


def rift_training_pair(
        clean: torch.Tensor,
        alpha: Optional[torch.Tensor | float] = None,
        noise: Optional[torch.Tensor] = None,
        detach_input: bool = True,
):
    """Build a linear-corruption residual-velocity training pair.

    x0 is uniform random noise, x1 is the clean image, and xt is a per-image
    linear interpolation. The target velocity is x1-xt, not x1-x0.
    """
    clean = clean.clamp(0.0, 1.0)
    if noise is None:
        noise = sample_noise_like(clean)
    else:
        noise = noise.to(device=clean.device, dtype=clean.dtype).clamp(0.0, 1.0)

    if alpha is None:
        alpha_map = torch.rand(
            (clean.shape[0], *([1] * (clean.ndim - 1))),
            device=clean.device,
            dtype=clean.dtype,
        ).clamp_min(1e-6)
    else:
        alpha_map = expand_batch_value(alpha, clean).clamp(0.0, 1.0)

    model_input = ((1.0 - alpha_map) * clean + alpha_map * noise).clamp(0.0, 1.0)
    target_velocity = clean - model_input
    if detach_input:
        model_input = model_input.detach()
    return model_input, target_velocity, noise, alpha_map


def alpha_loss_weights(alpha_map: torch.Tensor, max_weight: float = 100.0, eps: float = 1e-8) -> torch.Tensor:
    alpha = alpha_map.reshape(alpha_map.shape[0], -1).mean(dim=1)
    weights = alpha.clamp_min(float(eps)).pow(-2.0)
    return weights.clamp(max=float(max_weight))


def weighted_velocity_mse_loss(
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        alpha_map: torch.Tensor,
        max_weight: float = 100.0,
) -> torch.Tensor:
    """Per-image alpha^-2 weighted MSE, normalized by total batch weight."""
    per_image_mse = (predicted_velocity - target_velocity).square().flatten(1).mean(dim=1)
    weights = alpha_loss_weights(alpha_map, max_weight=max_weight).to(
        device=per_image_mse.device,
        dtype=per_image_mse.dtype,
    )
    return (per_image_mse * weights).sum() / weights.sum().clamp_min(1e-8)


def rift_velocity_step(
        current: torch.Tensor,
        predicted_velocity: torch.Tensor,
        step_size: float = 0.05,
) -> torch.Tensor:
    """Apply the model-predicted residual velocity in pixel space."""
    current = current.clamp(0.0, 1.0)
    predicted_velocity = predicted_velocity.to(device=current.device, dtype=current.dtype).clamp(-1.0, 1.0)
    return (current + float(step_size) * predicted_velocity).clamp(0.0, 1.0)
