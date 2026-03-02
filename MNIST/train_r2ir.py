import matplotlib.pyplot as plt
import torch
from MNIST.save_load_model import save_checkpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
num_epochs = 40
batch_size = 100
ema_decay = 0.999
train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

from modules.r2ir_r2id import R2IR
from modules.render_image import render_image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

model = R2IR(
    col_channels=1,
    lat_channels=64,
    embed_dim=128 + 64,
    pos_high_freq=10,
    pos_low_freq=6,
    enc_blocks=4,
    dec_blocks=4,
    num_heads=6,
    mha_dropout=0.1,
    ffn_dropout=0.2,
).to(device)
r2ir_scale = 8
lat_size = image_size // r2ir_scale
lat_values = model.lat_channels * lat_size ** 2
lat_ratio = lat_values / (model.col_channels * image_size ** 2)
print(f"Latent size: {lat_size:,} | Total values: {lat_values:,} | Image to Latent ratio 1:{lat_ratio:.3}")

model.print_model_summary()

# from save_load_model import load_checkpoint_into
#
# model = load_checkpoint_into(model, "models/_E40_0.07420_autoencoder_20260226_140527.pt", "cuda")

import copy

ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad = False


@torch.no_grad()
def update_ema_model(model, ema_model, decay):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


import math
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


peak_lr = 1e-3
final_lr = 1e-5
total_steps = num_epochs * len(train_dloader)
warmup_steps = len(train_dloader)

optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)
scheduler = make_cosine_with_warmup(optimizer, warmup_steps, total_steps, final_lr)


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


from tqdm import tqdm

train_loss_sums = []
test_loss_sums = []
train_losses = []

for E in range(num_epochs):
    model.train()
    model.zero_grad()
    train_loss_sum = 0.0
    for i, (image, label) in tqdm(enumerate(train_dloader), total=len(train_dloader), desc=f"TRAIN - E{E}"):
        image = invert_image(image).to(device)

        lat_img = model.encode(image, scale=r2ir_scale)
        recon_img = model.decode(lat_img, scale=r2ir_scale)

        loss = torch.nn.functional.mse_loss(recon_img, image)
        loss.backward()

        train_losses.append(loss.item())
        train_loss_sum += loss.item()

        optimizer.step()
        scheduler.step()
        update_ema_model(model, ema_model, ema_decay)
        model.zero_grad()

    train_loss_sum /= len(train_dloader)
    train_loss_sums.append(train_loss_sum)

    plt.title("Loss")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.show()

    model.eval()
    ema_model.eval()
    test_loss_sum = 0.0
    for i, (image, label) in tqdm(enumerate(test_dloader), total=len(test_dloader), desc=f"TEST - E{E}"):
        with torch.no_grad():
            image = invert_image(image).to(device)
            lat_img = ema_model.encode(image, scale=r2ir_scale)
            recon_img = ema_model.decode(lat_img, scale=r2ir_scale)
            loss = torch.nn.functional.mse_loss(recon_img, image)
            test_loss_sum += loss.item()
            if i == 0:
                render_image(uninvert_image(image))
                render_image(uninvert_image(recon_img), f"LOSS: {loss}")
    test_loss_sum /= len(test_dloader)
    test_loss_sums.append(test_loss_sum)

    print(f"TRAIN: {train_loss_sum:.5f} | TEST: {test_loss_sum:.5f}")
    plt.title("Loss")
    plt.plot(train_loss_sums, label="train")
    plt.plot(test_loss_sums, label="test")
    plt.legend()
    plt.show()

    model_path = save_checkpoint(ema_model, prefix=f"E{E + 1}_{test_loss_sum:.5f}_autoencoder")
