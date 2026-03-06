import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL CONFIGURATION ==================================================================================================
from modules.r2ir_r2id import R2IR
from save_load_model import save_model, load_model

model = R2IR(
    col_channels=1,
    lat_channels=64,
    embed_dim=256,
    pos_high_freq=10,
    pos_low_freq=6,
    enc_blocks=4,
    dec_blocks=4,
    num_heads=8,
    mha_dropout=0.1,
    ffn_dropout=0.2,
)
r2ir_scale = 1

# model = load_model(model, "foobar.safetensors")

import copy

ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad = False


@torch.no_grad()
def update_ema_model(model, ema_model, decay):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


ema_decay = 0.999

model = model.to(device)
ema_model = ema_model.to(device)

# IMAGE MANIPULATION AND DATALOADER CONFIGURATION ======================================================================
from modules.render_image import render_image


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


# Add these imports after the existing imports
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add this after the batch_size definition (around line 57)
image_size = 32
batch_size = 40


class OmniglotDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        # Omniglot has background set (train) and evaluation set (test)
        self.dataset = datasets.Omniglot(
            root='data',
            background=train,  # background=True for train, False for test
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
            ])
        )
        # Create a mapping from character class to integer label
        self.class_to_idx = {}
        current_idx = 0
        for _, class_idx in self.dataset:
            if class_idx not in self.class_to_idx:
                self.class_to_idx[class_idx] = current_idx
                current_idx += 1

    def __getitem__(self, index):
        image, class_idx = self.dataset[index]
        # Convert the original class index to our sequential integer label
        label = self.class_to_idx[class_idx]
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


# Create the datasets and dataloaders
train_dataset = OmniglotDataset(train=True)
test_dataset = OmniglotDataset(train=False)
train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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


peak_lr = 1e-4
final_lr = 1e-6
num_epochs = 40
total_steps = num_epochs * len(train_dloader)
warmup_steps = len(train_dloader)
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)
scheduler = make_cosine_with_warmup(optimizer, warmup_steps, total_steps, final_lr)

# TRAIN ================================================================================================================
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    model_path = save_model(ema_model, name=f"E{E + 1}_{test_loss_sum:.5f}_r2ir")
