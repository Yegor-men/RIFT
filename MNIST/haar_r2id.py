import time

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ======================================================================================================================
def one_hot_encode(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


image_size = 32


class OneHotMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root='data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                # transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
            ])
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        one_hot_label = one_hot_encode(label)
        return image, one_hot_label

    def __len__(self):
        return len(self.dataset)


train_dataset = OneHotMNIST(train=True)
test_dataset = OneHotMNIST(train=False)

# ======================================================================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

from modules.r2ir_r2id import R2IR, R2ID, HaarWavelet
from dummy_textencoder import DummyTextCond

r2ir = HaarWavelet(
    levels=2,
    channels=1,
)

r2id = R2ID(
    c_channels=r2ir.channels * 4 ** r2ir.levels,
    d_channels=256,
    enc_blocks=8,
    dec_blocks=8,
    num_heads=8,
    pos_freq=16,
    # pos_high_freq=10,
    # pos_low_freq=6,
    # time_freq=10,
    time_high_freq=7,
    time_low_freq=3,
    film_dim=128,
    self_attn_dropout=0.1,
    cross_attn_dropout=0.1,
    ffn_dropout=0.2,
)
# r2id.print_model_summary()

text_encoder = DummyTextCond(
    token_sequence_length=2,
    d_channels=r2id.d_channels
)

from save_load_model import save_model, load_model

# text_encoder = load_model(text_encoder, "MNIST_TEXT.safetensors")
# r2id = load_model(r2id, "MNIST_R2ID.safetensors")

text_encoder = text_encoder.to(device)
r2id = r2id.to(device)

import copy

ema_r2id = copy.deepcopy(r2id)
ema_r2id.eval()
for param in ema_r2id.parameters():
    param.requires_grad = False


@torch.no_grad()
def update_ema_model(model, ema_model, decay):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


time.sleep(0.2)

# ======================================================================================================================
from torch.optim.lr_scheduler import LambdaLR


def make_cosine_with_warmup(optimizer, warmup_steps, total_steps, lr_end):
    peak_lr = float(optimizer.defaults['lr'])

    lr_end = float(lr_end)
    min_mult = lr_end / peak_lr

    def lr_lambda(step):
        step = float(step)
        if step <= 0:
            return max(min_mult, 0.0)
        if step < warmup_steps:
            return (step / float(max(1.0, warmup_steps)))
        # after warmup: cosine decay from 1.0 -> min_mult
        progress = (step - warmup_steps) / float(max(1.0, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # map cosine in [0,1] to multiplier in [min_mult, 1.0]
        return min_mult + (1.0 - min_mult) * cosine

    return LambdaLR(optimizer, lr_lambda, -1)


num_epochs = 40
batch_size = 100
ema_decay = 0.999

train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

peak_lr = 1e-3
final_lr = 1e-5
total_steps = num_epochs * len(train_dloader)
warmup_steps = len(train_dloader)

optimizer = torch.optim.AdamW(params=list(r2id.parameters()) + list(text_encoder.parameters()), lr=peak_lr)
scheduler = make_cosine_with_warmup(optimizer, warmup_steps, total_steps, final_lr)

# ======================================================================================================================
import math

from tqdm import tqdm
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


train_losses = []
test_losses = []
percentile_losses = []

start = time.time()
for E in range(num_epochs):

    # TRAINING
    train_loss = 0
    r2id.train()
    for i, (image, label) in tqdm(enumerate(train_dloader), total=len(train_dloader), leave=True, desc=f"E:{E}"):
        b, c, h, w = image.shape
        if b != batch_size:
            continue

        with torch.no_grad():
            image, label = invert_image(image).to(device), label.to(device)
            image = r2ir.encode(image)

            t = torch.rand(b)
            t, _ = torch.sort(t)
            alpha_bar = alpha_bar_cosine(t).to(device)
            noisy_image, eps = corrupt_image(image, alpha_bar)
            noisy_image, eps = noisy_image.to(device), eps.to(device)

        pos_cond = text_encoder(label).to(device)
        null_cond = text_encoder(torch.zeros_like(label)).to(device)
        cond_list = [pos_cond, null_cond]

        predicted_eps_list = r2id(noisy_image, alpha_bar, cond_list)
        eps_pos, eps_null = predicted_eps_list[0], predicted_eps_list[1]
        loss = (nn.functional.mse_loss(eps_null, eps) + nn.functional.mse_loss(eps_pos, eps)) / 2
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        update_ema_model(r2id, ema_r2id, ema_decay)

    train_loss /= len(train_dloader)
    train_losses.append(train_loss)

    # TESTING
    test_loss = 0
    r2id.eval()
    with torch.no_grad():
        for i, (image, label) in tqdm(enumerate(test_dloader), total=len(test_dloader), leave=True, desc=f"E:{E}"):
            b, c, h, w = image.shape
            if b != batch_size:
                continue

            image, label = invert_image(image).to(device), label.to(device)
            image = r2ir.encode(image)

            t = torch.rand(b)
            t, _ = torch.sort(t)
            alpha_bar = alpha_bar_cosine(t).to(device)
            noisy_image, eps = corrupt_image(image, alpha_bar)
            noisy_image, eps = noisy_image.to(device), eps.to(device)
            pos_cond = text_encoder(label).to(device)
            null_cond = text_encoder(torch.zeros_like(label)).to(device)
            cond_list = [pos_cond, null_cond]

            predicted_eps_list = r2id(noisy_image, alpha_bar, cond_list)
            eps_pos, eps_null = predicted_eps_list[0], predicted_eps_list[1]
            loss = (nn.functional.mse_loss(eps_null, eps) + nn.functional.mse_loss(eps_pos, eps)) / 2
            test_loss += loss.item()

    test_loss /= len(test_dloader)
    test_losses.append(test_loss)
    print(f"Epoch {E} - TRAIN: {train_loss:.5f}, TEST: {test_loss:.5f}")
    time.sleep(0.2)

    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.legend()
    plt.show()

    # T SCRAPE LOSSES
    with torch.no_grad():
        t_range = torch.linspace(0, 1, steps=500)
        t_scrape_null_losses = []
        t_scrape_pos_losses = []

        for t in t_range:
            image, label = next(iter(train_dloader))
            b, c, h, w = image.shape
            image, label = invert_image(image).to(device), label.to(device)
            image = r2ir.encode(image)

            alpha_bar = alpha_bar_cosine(torch.ones(b) * t).to(device)
            noisy_image, eps = corrupt_image(image, alpha_bar)
            noisy_image, eps = noisy_image.to(device), eps.to(device)
            pos_cond = text_encoder(label).to(device)
            null_cond = text_encoder(torch.zeros_like(label)).to(device)
            cond_list = [pos_cond, null_cond]

            predicted_eps_list = r2id(noisy_image, alpha_bar, cond_list)
            eps_pos, eps_null = predicted_eps_list[0], predicted_eps_list[1]

            null_loss = nn.functional.mse_loss(eps_null, eps)
            pos_loss = nn.functional.mse_loss(eps_pos, eps)

            t_scrape_null_losses.append(null_loss.item())
            t_scrape_pos_losses.append(pos_loss.item())

        x = np.linspace(0, 1, len(t_scrape_null_losses))
        plt.plot(x, t_scrape_null_losses, label="Null")
        plt.plot(x, t_scrape_pos_losses, label="Pos")
        percentiles = [1, 25, 50, 75, 99]
        indices = [int(p / 100 * (len(t_scrape_null_losses) - 1)) for p in percentiles]
        percentile_x = [x[i] for i in indices]
        percentile_y = [t_scrape_null_losses[i] for i in indices]
        for px, py, p in zip(percentile_x, percentile_y, percentiles):
            plt.scatter(px, py, color='red')
            plt.text(px, py, f'{py}', fontsize=9, ha='center', va='bottom')
        plt.title('T scrape Losses')
        plt.legend()
        plt.show()

        percentile_losses.append(percentile_y)
        transposed = list(zip(*percentile_losses))
        for i, series in enumerate(transposed):
            plt.plot(series, label=f"t = {(percentiles[i] / 100):.2f}")
        plt.title("T scrape percentile losses over time")
        plt.legend()
        plt.show()

    # RENDERING
    with torch.no_grad():
        positive_label = torch.zeros(100, 10).to(device)
        for i in range(10):
            positive_label[i * 10:(i + 1) * 10, i] = 1.0

        # positive_label = torch.eye(10).to(device)

        pos_text_cond = text_encoder(positive_label)
        null_text_cond = text_encoder(torch.zeros_like(positive_label))

        for lat_size in (4, 6, 8, 10):
            lat_h, lat_w = lat_size, lat_size
            image_h, image_w = lat_h * (2 ** r2ir.levels), lat_w * (2 ** r2ir.levels)
            grid_noise = torch.randn(100, r2ir.channels * 4 ** r2ir.levels, lat_h, lat_w).to(device)
            final_x0_hat, final_x = run_ddim_visualization(
                model=ema_r2id,
                initial_noise=grid_noise,
                pos_text_cond=pos_text_cond,
                null_text_cond=null_text_cond,
                alpha_bar_fn=alpha_bar_cosine,
                num_steps=100,
                cfg_scale=4.0,
                eta=2.0,
                device=torch.device("cuda"),
            )

            diffused_image = uninvert_image(r2ir.decode(final_x))
            render_image(diffused_image, f"{"foo"} - Latent H:{lat_h}, W:{lat_w} - Image H:{image_h}, W:{image_w}",
                         "foo",
                         False)

    del final_x0_hat, final_x, grid_noise, pos_text_cond, null_text_cond
    torch.cuda.empty_cache()

    import gc

    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # MODEL SAVING
    if (E + 1) % 1 == 0 or E == num_epochs:
        model_path = save_model(ema_r2id, name=f"E{E + 1}_{test_loss:.5f}_MNIST_R2ID")
        text_encoder_path = save_model(text_encoder, name=f"E{E + 1}_{test_loss:.5f}_MNIST_TEXT")
        time.sleep(0.2)

# ======================================================================================================================
end = time.time()
total_time = end - start
import datetime

print(f"Finished training, total time: {datetime.timedelta(seconds=total_time)}")
