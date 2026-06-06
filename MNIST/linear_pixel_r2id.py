import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.r2id import R2IDLinear
from modules.render_image import render_image


# HELPERS ==============================================================================================================


def parse_size_list(size_list: str) -> list[int]:
    return [int(size.strip()) for size in size_list.split(",") if size.strip()]


def resize_image_batch(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.shape[-2:] == (size, size):
        return image
    return torch.nn.functional.interpolate(
        image,
        size=(size, size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(param.numel() for param in module.parameters() if param.requires_grad)
    return sum(param.numel() for param in module.parameters())


def format_parameters(num_params: int) -> str:
    if num_params >= 1_000_000:
        return f"{num_params:,} ({num_params / 1_000_000:.2f}M)"
    if num_params >= 1_000:
        return f"{num_params:,} ({num_params / 1_000:.1f}K)"
    return f"{num_params:,}"


def expand_time(time_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return time_tensor.to(device=target.device, dtype=target.dtype).view(
        time_tensor.shape[0], *([1] * (target.ndim - 1))
    )


def uniform_flow_training_pair(clean: torch.Tensor, time_tensor: torch.Tensor):
    clean = clean.clamp(0.0, 1.0)
    noise = torch.rand_like(clean)
    t = expand_time(time_tensor, clean)
    corrupted = ((1.0 - t) * noise + t * clean).clamp(0.0, 1.0)
    velocity = clean - noise
    return corrupted.detach(), velocity


def guided_velocity(v_null: torch.Tensor, v_pos: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    return v_null + float(cfg_scale) * (v_pos - v_null)


def uniform_flow_step(
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


# CONDITIONER ==========================================================================================================


class LabelTextCond(nn.Module):
    def __init__(self, num_classes: int, token_sequence_length: int, d_channels: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.null_label = int(num_classes)
        self.token_sequence_length = int(token_sequence_length)
        self.d_channels = int(d_channels)
        self.embedding = nn.Embedding(num_classes + 1, token_sequence_length * d_channels)

    def forward(self, labels: torch.Tensor):
        labels = labels.to(dtype=torch.long).clamp(min=0, max=self.null_label)
        tokens = self.embedding(labels)
        return tokens.view(labels.shape[0], self.token_sequence_length, self.d_channels)


# DATA ==================================================================================================================


class MNISTLabels(torch.utils.data.Dataset):
    def __init__(self, root: str, image_size: int, train: bool):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]),
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


# SAMPLING ==============================================================================================================


@torch.no_grad()
def sample_linear_pixel_r2id(
        model: R2IDLinear,
        text_encoder: LabelTextCond,
        labels: torch.Tensor,
        height: int,
        width: int,
        num_steps: int,
        cfg_scale: float,
        device: torch.device,
):
    model.eval()
    text_encoder.eval()
    labels = labels.to(device=device, dtype=torch.long)
    null_labels = torch.full_like(labels, fill_value=text_encoder.null_label)
    pos_text_cond = text_encoder(labels)
    null_text_cond = text_encoder(null_labels)
    x = torch.rand((labels.shape[0], 1, height, width), device=device)

    ts = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)
    x1_hat = x
    for i in tqdm(range(num_steps), total=num_steps, desc=f"sample {height}x{width}", leave=False):
        t_batch = torch.full((labels.shape[0],), float(ts[i].item()), device=device)
        s_batch = torch.full((labels.shape[0],), float(ts[i + 1].item()), device=device)

        model_input = x.clamp(0.0, 1.0)
        v_null, v_pos = model(model_input, t_batch, [null_text_cond, pos_text_cond])
        v_hat = guided_velocity(v_null, v_pos, cfg_scale)
        x, x1_hat = uniform_flow_step(x, t_batch, s_batch, v_hat)

    return x1_hat, x


# TRAIN/EVAL ===========================================================================================================


def make_cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, lr_end: float):
    peak_lr = float(optimizer.defaults["lr"])
    min_mult = float(lr_end) / peak_lr

    def lr_lambda(step):
        step = float(step)
        if step <= 0:
            return max(min_mult, 0.0)
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_mult + (1.0 - min_mult) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


@torch.no_grad()
def update_ema_model(model: nn.Module, ema_model: nn.Module, decay: float):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def velocity_prediction_loss(model, text_encoder, image, label, cfg_dropout: float):
    image = image.to(label.device).clamp(0.0, 1.0)
    if cfg_dropout > 0:
        drop = torch.rand_like(label.float()) < cfg_dropout
        train_label = torch.where(drop, torch.full_like(label, text_encoder.null_label), label)
    else:
        train_label = label

    t = torch.rand(image.shape[0], device=image.device)
    model_input, velocity = uniform_flow_training_pair(image, t)
    pred = model(model_input, t, [text_encoder(train_label)])[0]
    return nn.functional.mse_loss(pred, velocity)


@torch.no_grad()
def evaluate(model, text_encoder, dloader, sizes: list[int], args, device):
    model.eval()
    text_encoder.eval()
    losses_by_size = {}
    for size in sizes:
        total_loss = 0.0
        batches = 0
        for image, label in tqdm(dloader, total=len(dloader), desc=f"test {size}px", leave=False):
            if args.max_test_batches is not None and batches >= args.max_test_batches:
                break
            if image.shape[0] > args.eval_batch_size:
                image = image[:args.eval_batch_size]
                label = label[:args.eval_batch_size]
            image = resize_image_batch(image, size)
            image = image.to(device).clamp(0.0, 1.0)
            label = label.to(device)

            t = torch.rand(image.shape[0], device=device)
            model_input, velocity = uniform_flow_training_pair(image, t)
            null_label = torch.full_like(label, text_encoder.null_label)
            v_null, v_pos = model(model_input, t, [text_encoder(null_label), text_encoder(label)])
            loss = (nn.functional.mse_loss(v_null, velocity) + nn.functional.mse_loss(v_pos, velocity)) / 2
            total_loss += loss.item()
            batches += 1
        losses_by_size[size] = total_loss / max(1, batches)
    return losses_by_size


@torch.no_grad()
def scrape_t_losses(model, text_encoder, dloader, args, device, t_values: torch.Tensor):
    model.eval()
    text_encoder.eval()
    image, label = next(iter(dloader))
    if image.shape[0] > args.eval_batch_size:
        image = image[:args.eval_batch_size]
        label = label[:args.eval_batch_size]
    image = resize_image_batch(image, args.scrape_size)
    image = image.to(device).clamp(0.0, 1.0)
    label = label.to(device)
    null_label = torch.full_like(label, text_encoder.null_label)

    losses = []
    for t in t_values:
        t_batch = torch.full((image.shape[0],), float(t.item()), device=device)
        model_input, velocity = uniform_flow_training_pair(image, t_batch)
        v_null, v_pos = model(model_input, t_batch, [text_encoder(null_label), text_encoder(label)])
        loss = (nn.functional.mse_loss(v_null, velocity) + nn.functional.mse_loss(v_pos, velocity)) / 2
        losses.append(loss.item())
    return losses


def render_epoch_samples(model, text_encoder, epoch: int, args, device):
    labels = torch.arange(10, device=device, dtype=torch.long)
    for size in parse_size_list(args.sample_sizes):
        x1_hat, _ = sample_linear_pixel_r2id(
            model=model,
            text_encoder=text_encoder,
            labels=labels,
            height=size,
            width=size,
            num_steps=args.sample_steps,
            cfg_scale=args.cfg_scale,
            device=device,
        )
        render_image(x1_hat.clamp(0, 1), f"Linear Pixel R2ID | E{epoch + 1} | {size}px")


def save_model(model: nn.Module, text_encoder: nn.Module, epoch: int, test_loss: float):
    folder = Path("models")
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file(model.state_dict(),
              str(folder / f"E{epoch + 1}_{test_loss:.5f}_LINEAR_PIXEL_R2ID_{timestamp}.safetensors"))
    save_file(text_encoder.state_dict(),
              str(folder / f"E{epoch + 1}_{test_loss:.5f}_LINEAR_PIXEL_TEXT_{timestamp}.safetensors"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train raw-pixel R2ID with linear attention and uniform velocity flow.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-image-size", type=int, default=28)
    parser.add_argument("--test-sizes", default="28,64,128")
    parser.add_argument("--sample-sizes", default="28,64,128")
    parser.add_argument("--d-channels", type=int, default=192)
    parser.add_argument("--enc-blocks", type=int, default=3)
    parser.add_argument("--dec-blocks", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--pos-freq", type=int, default=16)
    parser.add_argument("--time-high-freq", type=int, default=7)
    parser.add_argument("--time-low-freq", type=int, default=3)
    parser.add_argument("--film-dim", type=int, default=128)
    parser.add_argument("--token-sequence-length", type=int, default=2)
    parser.add_argument("--self-attn-dropout", type=float, default=0.0)
    parser.add_argument("--cross-attn-dropout", type=float, default=0.0)
    parser.add_argument("--ffn-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-end", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--disable-skip-fusion", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--scrape-size", type=int, default=28)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.d_channels % args.num_heads == 0, "d_channels must be divisible by num_heads"
    if not args.disable_skip_fusion:
        assert args.dec_blocks == args.enc_blocks, "skip fusion expects --dec-blocks to equal --enc-blocks"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cuda is available: {torch.cuda.is_available()}")

    test_sizes = parse_size_list(args.test_sizes)
    print(
        f"Linear pixel R2ID config: train_size={args.dataset_image_size}, noise=uniform[0,1], "
        f"test_sizes={test_sizes}, samples={args.sample_sizes}"
    )

    model = R2IDLinear(
        c_channels=1,
        d_channels=args.d_channels,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
        num_heads=args.num_heads,
        pos_freq=args.pos_freq,
        time_high_freq=args.time_high_freq,
        time_low_freq=args.time_low_freq,
        film_dim=args.film_dim,
        self_attn_dropout=args.self_attn_dropout,
        cross_attn_dropout=args.cross_attn_dropout,
        ffn_dropout=args.ffn_dropout,
        skip_fusion=not args.disable_skip_fusion,
        velocity_output_scale=1.0,
    ).to(device)
    text_encoder = LabelTextCond(10, args.token_sequence_length, args.d_channels).to(device)

    ema_model = R2IDLinear(
        c_channels=1,
        d_channels=args.d_channels,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
        num_heads=args.num_heads,
        pos_freq=args.pos_freq,
        time_high_freq=args.time_high_freq,
        time_low_freq=args.time_low_freq,
        film_dim=args.film_dim,
        self_attn_dropout=args.self_attn_dropout,
        cross_attn_dropout=args.cross_attn_dropout,
        ffn_dropout=args.ffn_dropout,
        skip_fusion=not args.disable_skip_fusion,
        velocity_output_scale=1.0,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    print(f"R2IDLinear trainable parameters: {format_parameters(count_parameters(model, True))}")
    print(f"Text trainable parameters: {format_parameters(count_parameters(text_encoder, True))}")
    model.print_model_summary()

    train_dataset = MNISTLabels(args.data_root, args.dataset_image_size, train=True)
    test_dataset = MNISTLabels(args.data_root, args.dataset_image_size, train=False)
    train_dloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_dloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(text_encoder.parameters()), lr=args.lr)
    total_steps = args.epochs * len(train_dloader)
    scheduler = make_cosine_with_warmup(optimizer, len(train_dloader), total_steps, args.lr_end)

    train_losses = []
    test_avg_losses = []
    test_loss_history_by_size = {size: [] for size in test_sizes}
    t_values = torch.tensor([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    t_loss_history = []
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        text_encoder.train()
        train_loss = 0.0
        train_batches = 0

        for image, label in tqdm(train_dloader, total=len(train_dloader), desc=f"train E{epoch + 1}"):
            if args.max_train_batches is not None and train_batches >= args.max_train_batches:
                break
            label = label.to(device)
            loss = velocity_prediction_loss(model, text_encoder, image, label, args.cfg_dropout)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema_model(model, ema_model, args.ema_decay)

            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(1, train_batches)
        test_losses_by_size = evaluate(ema_model, text_encoder, test_dloader, test_sizes, args, device)
        test_loss = sum(test_losses_by_size.values()) / max(1, len(test_losses_by_size))
        t_losses = scrape_t_losses(ema_model, text_encoder, test_dloader, args, device, t_values)

        train_losses.append(train_loss)
        test_avg_losses.append(test_loss)
        t_loss_history.append(t_losses)
        for size in test_sizes:
            test_loss_history_by_size[size].append(test_losses_by_size[size])

        print(f"Epoch {epoch + 1} | TRAIN: {train_loss:.5f} | TEST: {test_loss:.5f}")
        print("Test by size:", " | ".join(f"{size}px: {test_losses_by_size[size]:.5f}" for size in test_sizes))
        print("T-losses:", " | ".join(f"t={t.item():.2f}: {loss:.5f}" for t, loss in zip(t_values, t_losses)))

        plt.title("Linear Pixel R2ID Velocity Loss")
        plt.plot(train_losses, label="train")
        plt.plot(test_avg_losses, label="test avg")
        plt.legend()
        plt.show()

        plt.title("Linear Pixel R2ID Test Loss by Resolution")
        for size in test_sizes:
            plt.plot(test_loss_history_by_size[size], label=f"{size}px")
        plt.legend()
        plt.show()

        t_history_tensor = torch.tensor(t_loss_history)
        plt.title("Linear Pixel R2ID Loss by t")
        for idx, t in enumerate(t_values):
            plt.plot(t_history_tensor[:, idx].tolist(), label=f"t={t.item():.2f}")
        plt.legend()
        plt.show()

        should_sample = args.sample_every > 0 and (epoch + 1) % args.sample_every == 0
        if not args.skip_sampling and should_sample:
            render_epoch_samples(ema_model, text_encoder, epoch, args, device)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_model(ema_model, text_encoder, epoch, test_loss)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished training in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
