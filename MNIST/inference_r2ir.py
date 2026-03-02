import torch

torch.manual_seed(0)
torch.cuda.manual_seed(0)

import torch
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


test_dataset = OneHotMNIST(train=False)
batch_size = 20
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

model.print_model_summary()

from MNIST.save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/_E40_0.01037_autoencoder_20260301_194643.pt", "cuda")


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


from tqdm import tqdm

model.eval()
test_loss_sum = 0.0
for i, (image, label) in tqdm(enumerate(test_dloader), total=len(test_dloader), desc="TEST"):
    with torch.no_grad():
        image = invert_image(image).to(device)
        lat_img = model.encode(image, height=32, width=32)
        recon_img = model.decode(lat_img, height=32, width=32)
        loss = torch.nn.functional.mse_loss(recon_img, image)
        test_loss_sum += loss.item()
        if i == 0:
            render_image(uninvert_image(image))
            render_image(uninvert_image(recon_img), f"LOSS: {loss}")

            # Render latent channels in a grid: rows=channels, columns=batch
            # lat_img_uninverted = uninvert_image(lat_img)
            # B, C, H, W = lat_img_uninverted.shape
            #
            # fig, axes = plt.subplots(C, B, figsize=(B * 2, C * 2))
            #
            # for batch_idx in range(B):
            #     for channel_idx in range(C):
            #         ax = axes[channel_idx, batch_idx] if C > 1 else axes[batch_idx]
            #         channel_img = lat_img_uninverted[batch_idx, channel_idx].cpu()
            #         ax.imshow(channel_img, cmap='gray')
            #
            #         # Add labels only on edges to avoid clutter
            #         if channel_idx == 0:
            #             ax.set_title(f"Batch {batch_idx}")
            #         if batch_idx == 0:
            #             ax.set_ylabel(f"Ch {channel_idx}")
            #
            #         ax.axis('off')
            #
            # plt.suptitle("Latent Representation (Channels × Batch)")
            # plt.tight_layout()
            # plt.show()

test_loss_sum /= len(test_dloader)
print(f"LOSS: {test_loss_sum}")
