from typing import Optional

import torch
from tqdm import tqdm

from modules.velocity import guided_velocity, scheduled_residual_step


@torch.no_grad()
def run_velocity_sampling(
        model: torch.nn.Module,
        initial_noise: torch.Tensor,
        pos_text_cond: torch.Tensor,
        null_text_cond: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        schedule_scale: float = 1.0,
        device: Optional[torch.device] = None,
):
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    device = device or initial_noise.device
    model = model.to(device)
    model.eval()

    x = initial_noise.to(device).clamp(0.0, 1.0)
    pos_text_cond = pos_text_cond.to(device)
    if null_text_cond is not None:
        null_text_cond = null_text_cond.to(device)

    batch_size = x.shape[0]
    x1_hat = x
    use_cfg = null_text_cond is not None and float(cfg_scale) != 1.0
    ts = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)

    for i in tqdm(range(num_steps), total=num_steps, desc="velocity sampling"):
        t_batch = torch.zeros((batch_size,), device=device)
        model_input = x.clamp(0.0, 1.0)

        if use_cfg:
            v_null, v_pos = model(model_input, t_batch, [null_text_cond, pos_text_cond])
            v_hat = guided_velocity(v_null, v_pos, cfg_scale)
        else:
            v_hat = model(model_input, t_batch, [pos_text_cond])[0]

        x = scheduled_residual_step(
            x,
            v_hat,
            t_current=float(ts[i].item()),
            t_next=float(ts[i + 1].item()),
            schedule_scale=schedule_scale,
        )
        x1_hat = x

    return x1_hat, x
