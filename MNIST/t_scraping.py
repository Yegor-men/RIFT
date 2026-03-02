import matplotlib.pyplot as plt
import torch
from torch import nn
# ======================================================================================================================
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 100
NUM_CLASSES = 10


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

train_dloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

from modules.r2ir_r2id import R2ID, R2IR
from MNIST.dummy_textencoder import DummyTextCond

r2ir = R2IR(
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

r2id = R2ID(
    c_channels=r2ir.lat_channels,
    d_channels=128 + r2ir.lat_channels,
    enc_blocks=8,
    dec_blocks=8,
    num_heads=6,
    pos_high_freq=10,
    pos_low_freq=6,
    time_high_freq=7,
    time_low_freq=3,
    film_dim=128,
    self_attn_dropout=0.1,
    cross_attn_dropout=0.1,
    ffn_dropout=0.2,
)

text_encoder = DummyTextCond(
    token_sequence_length=2,
    d_channels=r2id.d_channels
)

from MNIST.save_load_model import load_checkpoint_into

r2ir = load_checkpoint_into(r2ir, "models/_E40_0.01037_autoencoder_20260301_194643.pt", "cuda")
text_encoder = load_checkpoint_into(text_encoder, "models/_E40_0.01263_text_embedding_20260302_021117.pt")
r2id = load_checkpoint_into(r2id, "models/_E40_0.01263_diffusion_20260302_021117.pt", "cuda")

r2ir.eval()
r2id.to(device)
r2id.eval()

text_encoder.to(device)
text_encoder.eval()

# ======================================================================================================================
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from tqdm import tqdm
import numpy as np


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


max_num = 500

losses = []

with torch.no_grad():
    t_range = torch.linspace(0, 1, steps=500)
    t_scrape_null_losses = []
    t_scrape_pos_losses = []

    for t in tqdm(t_range, total=max_num, desc="Scraping"):
        image, label = next(iter(train_dloader))
        b, c, h, w = image.shape
        image, label = invert_image(image).to(device), label.to(device)
        image = r2ir.encode(image, width=8, height=8)

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
