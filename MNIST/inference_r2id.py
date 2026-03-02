import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.r2ir_r2id import R2ID, R2IR
from modules.dummy_textencoder import DummyTextCond

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
).to(device)

text_encoder = DummyTextCond(
    token_sequence_length=2,
    d_channels=r2id.d_channels
).to(device)

from modules.save_load_model import load_checkpoint_into

r2ir = load_checkpoint_into(r2ir, "models/_E40_0.01037_autoencoder_20260301_194643.pt", "cuda")
text_encoder = load_checkpoint_into(text_encoder, "models/_E40_0.01263_text_embedding_20260302_021117.pt")
r2id = load_checkpoint_into(r2id, "models/_E40_0.01263_diffusion_20260302_021117.pt", "cuda")

r2ir.eval()
r2id.eval()
text_encoder.eval()


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


with torch.no_grad():
    batch_size = 1

    if batch_size == 1:
        positive_label = torch.zeros(1, 10)
        positive_label[0][9] = 1
    if batch_size == 10:
        positive_label = torch.eye(10).to(device)
    if batch_size == 100:
        positive_label = torch.zeros(100, 10).to(device)
        for i in range(10):
            positive_label[i * 10:(i + 1) * 10, i] = 1.0

    positive_label = positive_label.to(device)

    pos_text_cond = text_encoder(positive_label)
    null_text_cond = text_encoder(torch.zeros_like(positive_label))

    sizes = [
        # (10, 10, "10px"),
        # (16, 16, "16px"),
        # (28, 28, "28px"),
        # (32, 32, "32px"),
        # (24, 30, "4_5"),
        # (30, 24, "5_4"),
        # (32, 24, "4_3"),
        # (24, 32, "3_4"),
        # (18, 27, "2_3"),
        # (27, 18, "3_2"),
        # (24, 40, "2_5"),
        # (40, 24, "5_2"),
        # (18, 32, "9_16"),
        # (32, 18, "16_9"),
        # (64, 64, "64px"),
        (1024, 1024, "1024px"),  # comment out because vram
    ]

    for lat_w in [256]:
        for lat_h in [256]:
            grid_noise = torch.randn(batch_size, r2ir.lat_channels, lat_h, lat_w).to(device)

            final_x0_hat, final_x = run_ddim_visualization(
                model=r2id,
                initial_noise=grid_noise,
                pos_text_cond=pos_text_cond,
                null_text_cond=null_text_cond,
                alpha_bar_fn=alpha_bar_cosine,
                num_steps=100,
                cfg_scale=4.0,
                eta=2.0,
                device=torch.device("cuda"),
            )

            for (height, width, name) in sizes:
                diffused_image = uninvert_image(r2ir.decode(final_x, height=height, width=width))
                filename = f"gens/LH{lat_h}LW{lat_w}_IH{height}IW{width}"
                render_image(
                    tensor=diffused_image,
                    title=f"{name} - Latent H:{lat_h}, W:{lat_w} - Image H:{height}, W:{width}",
                    name=filename,
                    save=False,
                )
