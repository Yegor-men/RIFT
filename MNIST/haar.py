import torch
import matplotlib.pyplot as plt
from save_load_model import save_model, load_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def one_hot_encode(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


image_size = 64


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
                transforms.Grayscale(num_output_channels=3),
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
from modules.r2ir_r2id import HaarWavelet

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
print(f"Cuda is available: {torch.cuda.is_available()}")

model = HaarWavelet(3, 3)


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


from tqdm import tqdm

train_loss_sums = []
test_loss_sums = []
train_losses = []
percentile_losses = []

from modules.corrupt_image import corrupt_image
from modules.alpha_bar import alpha_bar_cosine
import numpy as np
from torch import nn

for E in range(num_epochs):
    train_loss_sum = 0.0
    for i, (image, label) in tqdm(enumerate(train_dloader), total=len(train_dloader), desc=f"TRAIN - E{E}"):
        image = invert_image(torch.rand_like(image)).to(device)
        b, c, h, w = image.size()

        latent, _ = model.encode(image)
        recon = model.decode(latent, {'original_shape': (b, c, h, w)})

        loss = nn.functional.mse_loss(recon, image)

    train_loss_sum /= len(train_dloader)
    train_loss_sums.append(train_loss_sum)

    plt.title("Loss")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.show()

    test_loss_sum = 0.0
    for i, (image, label) in tqdm(enumerate(test_dloader), total=len(test_dloader), desc=f"TEST - E{E}"):
        with torch.no_grad():
            image = invert_image(image).to(device)
            b, c, h, w = image.size()
            latent, _ = model.encode(image)
            recon = model.decode(latent, {'original_shape': (b, c, h, w)})
            loss = torch.nn.functional.mse_loss(recon, image)
            test_loss_sum += loss.item()
            if i == 0:
                render_image(uninvert_image(image))
                render_image(uninvert_image(recon), f"LOSS: {loss}")
    test_loss_sum /= len(test_dloader)
    test_loss_sums.append(test_loss_sum)

    print(f"TRAIN: {train_loss_sum:.5f} | TEST: {test_loss_sum:.5f}")
    plt.title("Loss")
    plt.plot(train_loss_sums, label="train")
    plt.plot(test_loss_sums, label="test")
    plt.legend()
    plt.show()
