import torch
import math
from torch import nn


class MultiHeadLinearAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_embed: torch.Tensor, key_embed: torch.Tensor, value: torch.Tensor):
        B, Nq, _ = query_embed.shape
        _, Nk, _ = key_embed.shape

        # Project & head-split
        q = self.q_proj(query_embed).view(B, Nq, self.num_heads, self.head_dim)
        k = self.k_proj(key_embed).view(B, Nk, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim)

        q = nn.functional.softmax(q, -1)
        k = nn.functional.softmax(k, -1)

        # Transpose to (B, heads, seq, head_dim) for clean einsums
        q = q.transpose(1, 2)  # (B, H, Nq, D)
        k = k.transpose(1, 2)  # (B, H, Nk, D)
        v = v.transpose(1, 2)  # (B, H, Nk, D)

        # Linear attention core
        kv_sum = torch.einsum('b h n d, b h n e -> b h d e', k, v)  # (B, H, D, D)
        k_sum = k.sum(dim=2)  # (B, H, D)

        # Query the summary
        num = torch.einsum('b h q d, b h d e -> b h q e', q, kv_sum)
        den = torch.einsum('b h q d, b h d -> b h q', q, k_sum).unsqueeze(-1) + 1e-8

        out = (num / den).transpose(1, 2).reshape(B, Nq, -1)  # back to (B, Nq, embed_dim)

        return self.dropout(self.out_proj(out))


class ImageNorm(nn.Module):
    def __init__(self, num_channels: int, affine: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, elementwise_affine=affine)

    def forward(self, x):
        x = torch.movedim(x, -3, -1)
        x = self.norm(x)
        x = torch.movedim(x, -1, -3)
        return x


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2.
       Global, resolution-invariant, inter-channel competition, no pixel mixing."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # 1. Global L2 response per channel (energy of the whole image per channel)
        gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]

        # 2. Normalize responses across channels (competition)
        nx = gx / (gx.mean(dim=-3, keepdim=True) + self.eps)  # relative strength

        # 3. Apply + learnable calibration + residual
        return (self.gamma.unsqueeze(-1).unsqueeze(-1)) * (x * nx) + self.beta.unsqueeze(-1).unsqueeze(-1) + x


class PosEmbed2d(nn.Module):
    def __init__(self, num_frequencies: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.num_frequencies = int(num_frequencies)

        powers = torch.arange(self.num_frequencies, dtype=torch.float32)  # [0, 1, ...]
        frequencies = torch.pi * (2.0 ** powers)  # [..., pi/4, pi/2, pi, 2pi, 4pi, ...]
        self.register_buffer("frequencies", frequencies, persistent=True)

        self.norm = nn.Sequential(
            GRN(4 * self.num_frequencies),
            ImageNorm(4 * self.num_frequencies),
        )

    def _make_grid(self, h: int, w: int, relative: bool):
        if relative:
            if w >= h:
                x_min, x_max = -0.5, 0.5
                y_extent = h / w
                y_min, y_max = -0.5 * y_extent, 0.5 * y_extent
            else:
                y_min, y_max = -0.5, 0.5
                x_extent = w / h
                x_min, x_max = -0.5 * x_extent, 0.5 * x_extent
        else:
            x_min, x_max, y_min, y_max = -0.5, 0.5, -0.5, 0.5

        x_coordinates = torch.linspace(x_min + self.eps, x_max - self.eps, steps=w)
        y_coordinates = torch.linspace(y_min + self.eps, y_max - self.eps, steps=h)

        yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        grid = torch.stack([xx, yy], dim=0)
        return grid

    def make_grid(self, batch_size: int, h: int, w: int, relative: bool):
        base_grid = self._make_grid(h, w, relative)
        base_grid = base_grid.to(self.frequencies.device)  # [2, h, w]

        grid = base_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [b, 2, h, w]

        if self.training:
            if relative:
                max_dim = max(h, w)
                sigma = 1.0 / (2 * max_dim)
                jitter_x = torch.normal(mean=0.0, std=sigma, size=(batch_size, 1, h, w), device=grid.device)
                jitter_y = torch.normal(mean=0.0, std=sigma, size=(batch_size, 1, h, w), device=grid.device)
            else:
                sigma_x = 1.0 / (2 * w)
                sigma_y = 1.0 / (2 * h)
                jitter_x = torch.normal(mean=0.0, std=sigma_x, size=(batch_size, 1, h, w), device=grid.device)
                jitter_y = torch.normal(mean=0.0, std=sigma_y, size=(batch_size, 1, h, w), device=grid.device)
            jitter = torch.cat([jitter_x, jitter_y], dim=1)  # [b, 2, h, w]
            grid = grid + jitter

        grid_unsqueezed = grid.unsqueeze(-1)  # [b, 2, h, w, 1]
        frequencies = self.frequencies.view(1, 1, 1, 1, -1)  # [1, 1, 1, 1, F]
        tproj = grid_unsqueezed * frequencies  # [b, 2, h, w, F]

        sin_feat = torch.sin(tproj)  # [b, 2, h, w, F]
        cos_feat = torch.cos(tproj)  # [b, 2, h, w, F]

        sin_ch = sin_feat.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, 2 * self.num_frequencies, h, w)
        cos_ch = cos_feat.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, 2 * self.num_frequencies, h, w)
        fourier_ch = torch.cat([sin_ch, cos_ch], dim=1)  # [b, 4F, h, w]

        positional_embedding = self.norm(fourier_ch)  # [b, 4F, h, w]

        return positional_embedding

    def forward(self, batch_size: int, h: int, w: int):
        rel_pos_map = self.make_grid(batch_size, h, w, True)
        abs_pos_map = self.make_grid(batch_size, h, w, False)
        pos_map = torch.cat([rel_pos_map, abs_pos_map], dim=-3)
        return pos_map


class ContTimeEmbed(nn.Module):
    def __init__(self, num_high_freq: int, num_low_freq: int = 0, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.num_frequencies = num_high_freq + num_low_freq

        powers = torch.arange(self.num_frequencies, dtype=torch.float32) - num_low_freq
        frequencies = torch.pi * (2.0 ** powers)  # [pi, 2pi, 4pi, ...]
        self.register_buffer("frequencies", frequencies, persistent=True)

        self.norm = nn.LayerNorm(2 * self.num_frequencies, elementwise_affine=False)
        # self.norm = nn.RMSNorm(2 * self.num_frequencies, elementwise_affine=False)

    def forward(self, alpha_bar: torch.Tensor) -> torch.Tensor:
        alpha_mapped = alpha_bar * (1 - 2 * self.eps) - (0.5 - self.eps)
        # Now it's between [-0.5 + eps, 0.5 - eps]

        tproj = alpha_mapped.unsqueeze(1) * self.frequencies.view(1, -1)
        sin_feat = torch.sin(tproj)
        cos_feat = torch.cos(tproj)
        feat = torch.cat([sin_feat, cos_feat], dim=-1)

        time_vector = self.norm(feat)

        return time_vector


class ImageAdaLN(nn.Module):
    def __init__(self, film_dim: int, out_dim: int):
        super().__init__()

        self.gb = nn.Sequential(
            nn.Linear(film_dim, 2 * out_dim),
        )

        nn.init.normal_(self.gb[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gb[-1].bias)

        self.norm = nn.Sequential(
            GRN(out_dim),
            ImageNorm(out_dim),
        )

    def forward(self, x, time_cond):
        gb = self.gb(time_cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + gamma

        x = self.norm(x) * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)

        return x


class ImageFFN(nn.Module):
    def __init__(self, d_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_channels, 4 * d_channels, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(4 * d_channels, d_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, d_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_channels % num_heads == 0, f"d_channels ({d_channels}) must be divisible by num_heads ({num_heads})"

        # self.mha = nn.MultiheadAttention(
        #     embed_dim=d_channels,
        #     num_heads=num_heads,
        #     batch_first=True,
        #     dropout=dropout,
        # )

        self.mha = MultiHeadLinearAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, image, text_tokens):
        b, d, h, w = image.shape

        s = h * w
        Q = image.permute(0, 2, 3, 1).contiguous().view(b, s, d)  # [B, S, D]

        # MHA wants shapes: (B, seq_q, D), (B, seq_k, D), (B, seq_k, D)
        # attn_out, _ = self.mha(Q, text_tokens, text_tokens, need_weights=False)  # [B, S, D]
        attn_out = self.mha(Q, text_tokens, text_tokens)

        # reshape back to image grid [B, D, H, W]
        attn_out = attn_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        return attn_out


class SelfAttention(nn.Module):
    def __init__(self, d_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_channels % num_heads == 0, f"d_channels ({d_channels}) must be divisible by num_heads ({num_heads})"
        # self.mha = nn.MultiheadAttention(
        #     embed_dim=d_channels,
        #     num_heads=num_heads,
        #     batch_first=True,
        #     dropout=dropout,
        # )

        self.mha = MultiHeadLinearAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, image):
        b, d, h, w = image.shape

        # Flatten to sequence for attention [B, H*W, D]
        flat_img = image.flatten(2).transpose(1, 2)  # [B, H*W, D]

        # Full global self-attention
        attn_out = self.mha(flat_img, flat_img, flat_img)

        # Reshape back to image format
        attn_out = attn_out.transpose(1, 2).view(b, d, h, w)

        return attn_out


class EncBlock(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            film_dim: int,
            self_attn_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_channels = d_channels

        self.self_ada = ImageAdaLN(film_dim, d_channels)
        self.self_attn = SelfAttention(
            d_channels=d_channels,
            num_heads=num_heads,
            dropout=self_attn_dropout,
        )
        self.self_scalar = nn.Parameter(torch.ones(d_channels))

        self.ffn_ada = ImageAdaLN(film_dim, d_channels)
        self.ffn = ImageFFN(d_channels, ffn_dropout)
        self.ffn_scalar = nn.Parameter(torch.ones(d_channels))

        self.final_scalar = nn.Parameter(torch.ones(d_channels) * 0.1)

    def forward(self, image, film_vector):
        working_image = image

        self_adad = self.self_ada(working_image, film_vector)
        self_out = self.self_attn(self_adad)

        working_image = working_image + self_out * self.self_scalar.view(1, self.d_channels, 1, 1)

        ffn_adad = self.ffn_ada(working_image, film_vector)
        ffn_out = self.ffn(ffn_adad)

        working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

        final_image = image + working_image * self.final_scalar.view(1, self.d_channels, 1, 1)

        return final_image


class DecBlock(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            film_dim: int,
            self_attn_dropout: float = 0.0,
            cross_attn_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_channels = d_channels

        self.self_ada = ImageAdaLN(film_dim, d_channels)
        self.self_attn = SelfAttention(
            d_channels=d_channels,
            num_heads=num_heads,
            dropout=self_attn_dropout,
        )
        self.self_scalar = nn.Parameter(torch.ones(d_channels))

        self.cross_ada = ImageAdaLN(film_dim, d_channels)
        self.cross_attn = CrossAttention(
            d_channels=d_channels,
            num_heads=num_heads,
            dropout=cross_attn_dropout,
        )
        self.cross_scalar = nn.Parameter(torch.ones(d_channels))

        self.ffn_ada = ImageAdaLN(film_dim, d_channels)
        self.ffn = ImageFFN(d_channels, ffn_dropout)
        self.ffn_scalar = nn.Parameter(torch.ones(d_channels))

        self.final_scalar = nn.Parameter(torch.ones(d_channels) * 0.1)

    def forward(self, image, film_vector, text_tokens):
        working_image = image

        self_adad = self.self_ada(working_image, film_vector)
        self_out = self.self_attn(self_adad)

        working_image = working_image + self_out * self.self_scalar.view(1, self.d_channels, 1, 1)

        cross_adad = self.cross_ada(working_image, film_vector)
        cross_out = self.cross_attn(cross_adad, text_tokens)

        working_image = working_image + cross_out * self.cross_scalar.view(1, self.d_channels, 1, 1)

        ffn_adad = self.ffn_ada(working_image, film_vector)
        ffn_out = self.ffn(ffn_adad)

        working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

        final_image = image + working_image * self.final_scalar.view(1, self.d_channels, 1, 1)

        return final_image


# ======================================================================================================================


class R2IRCrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mha_dropout: float = 0.0, ffn_dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim

        self.attn_norm = nn.Sequential(
            GRN(embed_dim),
            ImageNorm(embed_dim),
        )
        self.attn = MultiHeadLinearAttention(embed_dim, num_heads, dropout=mha_dropout)
        self.attn_scalar = nn.Parameter(torch.ones(embed_dim))

        self.ffn_norm = nn.Sequential(
            GRN(embed_dim),
            ImageNorm(embed_dim),
        )
        self.ffn = ImageFFN(embed_dim, ffn_dropout)
        self.ffn_scalar = nn.Parameter(torch.ones(embed_dim))

        self.final_scalar = nn.Parameter(torch.ones(embed_dim) * 0.1)

    def forward(self, query_img: torch.Tensor, key_img: torch.Tensor, value_img: torch.Tensor):
        # All inputs are [B, C, Hq/Wq, Wq/Hq] for query, [B, C, Hk, Wk] for key/value (Hq/Wq may != Hk/Wk)
        B, C, Hq, Wq = query_img.shape
        _, _, Hk, Wk = key_img.shape

        working_img = query_img

        # Pre-attn normalization
        attn_normed = self.attn_norm(working_img)

        # Flatten for attn (queries, keys, values)
        Q_flat = attn_normed.flatten(2).transpose(1, 2)  # [B, Hq*Wq, C]
        K_flat = key_img.flatten(2).transpose(1, 2)  # [B, Hk*Wk, C]
        V_flat = value_img.flatten(2).transpose(1, 2)  # [B, Hk*Wk, C]

        # Linear attn
        attn_out_flat = self.attn(Q_flat, K_flat, V_flat)  # [B, Hq*Wq, C]

        # Reshape back to image
        attn_out = attn_out_flat.transpose(1, 2).view(B, C, Hq, Wq)

        # Residual + scalar
        working_img = working_img + attn_out * self.attn_scalar.view(1, C, 1, 1)

        # Pre-FFN normalization
        ffn_normed = self.ffn_norm(working_img)

        # FFN (stays in image shape)
        ffn_out = self.ffn(ffn_normed)

        # Residual + scalar
        working_img = working_img + ffn_out * self.ffn_scalar.view(1, C, 1, 1)

        # Final residual + block scalar
        final_img = query_img + working_img * self.final_scalar.view(1, C, 1, 1)

        return final_img


class R2IR(nn.Module):
    def __init__(
            self,
            col_channels: int = 1,
            lat_channels: int = 768,
            embed_dim: int = 1024,
            pos_freq: int = 10,
            enc_blocks: int = 2,
            dec_blocks: int = 2,
            num_heads: int = 16,
            mha_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.col_channels = col_channels
        self.lat_channels = lat_channels
        self.embed_dim = embed_dim
        self.pos_dim = 4 * pos_freq
        self.num_enc_blocks = int(enc_blocks)
        self.num_dec_blocks = int(dec_blocks)
        self.num_heads = int(num_heads)

        self.pos_embed = PosEmbed2d(pos_freq)

        # col_c + pos -> embed dim
        self.color_to_embed_proj = nn.Conv2d(col_channels + self.pos_dim * 2, embed_dim, 1)

        # pos -> embed_dim
        self.pos_to_embed_proj = nn.Conv2d(self.pos_dim * 2, embed_dim, 1)

        # lat_c + pos -> embed dim
        self.latent_to_embed_proj = nn.Conv2d(lat_channels + self.pos_dim * 2, embed_dim, 1)

        # output head for encoding (to lat_channels colors)
        self.enc_out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, lat_channels, 1),
            nn.Tanh()
        )
        nn.init.zeros_(self.enc_out_proj[-2].weight)
        nn.init.zeros_(self.enc_out_proj[-2].bias)

        # Output head for decoding (to col_channels colors)
        self.dec_out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, col_channels, 1),
            nn.Tanh()
        )
        nn.init.zeros_(self.dec_out_proj[-2].weight)
        nn.init.zeros_(self.dec_out_proj[-2].bias)

        self.enc_blocks = nn.ModuleList([
            R2IRCrossBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mha_dropout=mha_dropout,
                ffn_dropout=ffn_dropout,
            ) for _ in range(enc_blocks)
        ])

        self.dec_blocks = nn.ModuleList([
            R2IRCrossBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mha_dropout=mha_dropout,
                ffn_dropout=ffn_dropout,
            ) for _ in range(dec_blocks)
        ])

    def print_model_summary(self, detailed: bool = False):
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        print("=== R2IR Model Summary ===")
        print(f"\tembed_dim: {self.embed_dim} | pos_dim: {self.pos_dim * 2}")
        print(f"\tcol/lat channels: {self.col_channels}/{self.lat_channels}")
        print(f"Total Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        if detailed:
            print(f"\tPosition Embedding: {count_params(self.pos_embed):,}")
            print(f"\tColor→Embed Proj: {count_params(self.color_to_embed_proj):,}")
            print(f"\tPos→Embed Proj: {count_params(self.pos_to_embed_proj):,}")
            print(f"\tLatent→Embed Proj: {count_params(self.latent_to_embed_proj):,}")
            print(f"\tEncoder Output Proj: {count_params(self.enc_out_proj):,}")
            print(f"\tDecoder Output Proj: {count_params(self.dec_out_proj):,}")
            print(f"\tEncoder Blocks: {count_params(self.enc_blocks):,}")
            print(f"\tDecoder Blocks: {count_params(self.dec_blocks):,}")

    def encode(self, image, scale: int = 2, height: int = None, width: int = None):
        b, _, ih, iw = image.shape
        if height is None or width is None:
            height = ih // scale
            width = iw // scale
        lh, lw = height, width

        # Input tokens (color + pos) — keep 4D
        pos = self.pos_embed(b, ih, iw)
        stacked = torch.cat([image, pos], dim=1)
        input_tokens = self.color_to_embed_proj(stacked)  # [B, embed_dim, ih, iw]

        latent_pos_map = self.pos_embed(b, lh, lw)
        latent_queries = self.pos_to_embed_proj(latent_pos_map)  # [B, embed_dim, lh, lw]

        for enc_block in self.enc_blocks:
            latent_queries = enc_block(latent_queries, input_tokens, input_tokens)

        latent = self.enc_out_proj(latent_queries)  # already 4D [B, lat_channels, lh, lw]
        return latent

    def decode(self, latent, scale: int = 2, height: int = None, width: int = None):
        b, _, lh, lw = latent.shape
        if height is None or width is None:
            height = lh * scale
            width = lw * scale
        ih, iw = height, width

        # Latent tokens (latent_color + pos) — keep 4D
        latent_pos_map = self.pos_embed(b, lh, lw)
        stacked = torch.cat([latent, latent_pos_map], dim=1)
        latent_tokens = self.latent_to_embed_proj(stacked)  # [B, embed_dim, lh, lw]

        out_pos_map = self.pos_embed(b, ih, iw)
        out_queries = self.pos_to_embed_proj(out_pos_map)  # [B, embed_dim, ih, iw]

        for dec_block in self.dec_blocks:
            out_queries = dec_block(out_queries, latent_tokens, latent_tokens)

        out = self.dec_out_proj(out_queries)  # already 4D [B, col_channels, ih, iw]
        return out


class HaarWavelet:
    def __init__(self, levels=3, channels=3):
        self.levels = levels
        self.channels = channels
        self.scale = 1.0 / math.sqrt(2.0)

    def single_level_forward(self, img):
        b, c, h, w = img.shape
        assert h % 2 == 0 and w % 2 == 0

        # Horizontal
        even = img[..., ::2]
        odd = img[..., 1::2]
        avg_h = (even + odd) * self.scale
        det_h = (even - odd) * self.scale
        tmp = torch.cat((avg_h, det_h), dim=-1)  # [b, c, h, w]

        # Vertical
        even = tmp[..., ::2, :]
        odd = tmp[..., 1::2, :]
        avg_v = (even + odd) * self.scale
        det_v = (even - odd) * self.scale
        coeffs = torch.cat((avg_v, det_v), dim=-2)  # [b, c, h, w]

        return coeffs

    def single_level_inverse(self, ll, lh, hl, diag):
        b, c, hh, ww = ll.shape
        # Vertical inverse
        stacked_left = torch.stack((ll, hl), dim=-1)  # [b, c, hh, ww, 2]
        upper_left = (stacked_left[..., 0] + stacked_left[..., 1]) * self.scale
        lower_left = (stacked_left[..., 0] - stacked_left[..., 1]) * self.scale

        left_recon = torch.empty((b, c, 2 * hh, ww), dtype=ll.dtype, device=ll.device)
        left_recon[..., ::2, :] = upper_left
        left_recon[..., 1::2, :] = lower_left

        stacked_right = torch.stack((lh, diag), dim=-1)
        upper_right = (stacked_right[..., 0] + stacked_right[..., 1]) * self.scale
        lower_right = (stacked_right[..., 0] - stacked_right[..., 1]) * self.scale

        right_recon = torch.empty((b, c, 2 * hh, ww), dtype=ll.dtype, device=ll.device)
        right_recon[..., ::2, :] = upper_right
        right_recon[..., 1::2, :] = lower_right

        tmp = torch.cat((left_recon, right_recon), dim=-1)  # [b, c, 2*hh, 2*ww]

        # Horizontal inverse
        h_full, w_full = tmp.shape[-2:]
        stacked = torch.stack((tmp[..., :w_full // 2], tmp[..., w_full // 2:]), dim=-1)  # [b, c, h_full, w_full//2, 2]
        even = (stacked[..., 0] + stacked[..., 1]) * self.scale
        odd = (stacked[..., 0] - stacked[..., 1]) * self.scale
        recon = torch.empty_like(tmp)
        recon[..., ::2] = even
        recon[..., 1::2] = odd

        return recon

    def fold(self, sub, factor, H, W):
        b, c, sub_h, sub_w = sub.shape
        assert sub_h == factor * H and sub_w == factor * W
        tmp = sub.reshape(b, c, factor, H, factor, W)
        tmp = tmp.permute(0, 1, 2, 4, 3, 5)  # b, c, factor, factor, H, W
        folded = tmp.reshape(b, c * factor ** 2, H, W)
        return folded

    def unfold(self, folded, factor, H, W):
        b, ch, h, w = folded.shape
        assert ch == self.channels * factor ** 2 and h == H and w == W
        tmp = folded.reshape(b, self.channels, factor, factor, H, W)
        tmp = tmp.permute(0, 1, 2, 4, 3, 5)  # b, channels, factor, H, factor, W
        unfolded = tmp.reshape(b, self.channels, factor * H, factor * W)
        return unfolded

    def encode(self, img):
        b, c, h, w = img.shape
        assert c == self.channels
        power = 2 ** self.levels
        assert h % power == 0 and w % power == 0

        subbands = []
        current = img
        for _ in range(self.levels):
            coeffs = self.single_level_forward(current)
            hh = current.shape[-2] // 2
            ww = current.shape[-1] // 2
            subbands.append((
                coeffs[..., :hh, ww:],
                coeffs[..., hh:, :ww],
                coeffs[..., hh:, ww:]
            ))
            current = coeffs[..., :hh, :ww]

        H, W = current.shape[-2:]
        latent_parts = [current]
        lev = 0
        for det in reversed(subbands):
            lh, hl, hh_d = det
            f = 2 ** lev
            lh_f = self.fold(lh, f, H, W)
            hl_f = self.fold(hl, f, H, W)
            hh_f = self.fold(hh_d, f, H, W)
            latent_parts += [lh_f, hl_f, hh_f]
            lev += 1

        latent = torch.cat(latent_parts, dim=1)
        return latent

    def decode(self, latent):
        b_, C, H, W = latent.shape
        power = 2 ** self.levels
        assert C == self.channels * (4 ** self.levels)
        b, c, h, w = b_, self.channels, H * power, W * power

        idx = 0
        current = latent[:, idx: idx + self.channels, :, :]
        idx += self.channels

        details_list = []
        for lev in range(self.levels):
            fold_ch = self.channels * (4 ** lev)
            lh_f = latent[:, idx: idx + fold_ch, :, :]
            idx += fold_ch
            hl_f = latent[:, idx: idx + fold_ch, :, :]
            idx += fold_ch
            hh_f = latent[:, idx: idx + fold_ch, :, :]
            idx += fold_ch
            details_list.append((lh_f, hl_f, hh_f))

        for lev in range(self.levels):
            lh_f, hl_f, hh_f = details_list[lev]
            f = 2 ** lev
            lh = self.unfold(lh_f, f, H, W)
            hl = self.unfold(hl_f, f, H, W)
            diag = self.unfold(hh_f, f, H, W)

            current = self.single_level_inverse(current, lh, hl, diag)

        return current


# ======================================================================================================================

class R2ID(nn.Module):
    def __init__(
            self,
            c_channels: int,  # color channels
            d_channels: int,  # channels in the latent
            enc_blocks: int,  # number of encoder blocks (no cross attention)
            dec_blocks: int,  # number of decoder blocks (yes cross attention)
            num_heads: int,  # num heads in each block, d_channels must be divisible here
            # pos_high_freq: int,
            # pos_low_freq: int,
            time_high_freq: int,
            time_low_freq: int,
            pos_freq: int,
            # time_freq: int,
            film_dim: int,  # dimension that the base film vector sits in, then gets turned to d channels
            self_attn_dropout: float = 0.0,
            cross_attn_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.c_channels = int(c_channels)
        self.d_channels = int(d_channels)
        self.num_enc_blocks = int(enc_blocks)
        self.num_dec_blocks = int(dec_blocks)
        self.num_heads = int(num_heads)
        # self.num_pos_frequencies = int(pos_low_freq + pos_high_freq)
        self.num_time_frequencies = int(time_low_freq + time_high_freq)
        self.num_pos_frequencies = int(pos_freq)
        # self.num_time_frequencies = int(time_freq)
        self.film_dim = int(film_dim)

        self.proj_to_latent = nn.Conv2d(self.num_pos_frequencies * 4 * 2 + c_channels, d_channels, 1)
        self.latent_to_epsilon = nn.Conv2d(d_channels, c_channels, 1)
        nn.init.zeros_(self.latent_to_epsilon.weight)
        nn.init.zeros_(self.latent_to_epsilon.bias)

        # self.pos_embed = PosEmbed2d(pos_high_freq, pos_low_freq)
        # self.time_embed = ContTimeEmbed(time_high_freq, time_low_freq)
        self.pos_embed = PosEmbed2d(pos_freq)
        self.time_embed = ContTimeEmbed(time_high_freq, time_low_freq)
        self.film_proj = nn.Sequential(
            nn.Linear(self.num_time_frequencies * 2, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
            nn.SiLU()
        )

        self.enc_blocks = nn.ModuleList([
            EncBlock(
                d_channels=d_channels,
                num_heads=num_heads,
                film_dim=film_dim,
                self_attn_dropout=self_attn_dropout,
                ffn_dropout=ffn_dropout,
            ) for _ in range(enc_blocks)
        ])

        self.dec_blocks = nn.ModuleList([
            DecBlock(
                d_channels=d_channels,
                num_heads=num_heads,
                film_dim=film_dim,
                self_attn_dropout=self_attn_dropout,
                cross_attn_dropout=cross_attn_dropout,
                ffn_dropout=ffn_dropout,
            ) for _ in range(dec_blocks)
        ])

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total:,}")

        total_pos_channels = self.num_pos_frequencies * 2 * 2 * 2  # x/y, sin/cos, rel/abs
        total_col_channels = self.c_channels
        total_channels = total_pos_channels + total_col_channels

        print(f"Channels for color/positioning: {total_col_channels}/{total_pos_channels}, total: {total_channels}")

    def forward(self, image: torch.Tensor, alpha_bar: torch.Tensor, text_conds: list[torch.Tensor]):
        assert image.ndim == 4, "Image must be batch, tensor shape of [B, C, H, W]"
        b, c, h, w = image.shape

        epsilon_list = []

        time_vector = self.time_embed(alpha_bar)  # [B, time_dim]
        film_vector = self.film_proj(time_vector)

        pos_map = self.pos_embed(b, h, w)

        stacked_latent = torch.cat([image, pos_map], dim=-3)
        latent = self.proj_to_latent(stacked_latent)

        for i, enc_block in enumerate(self.enc_blocks):
            latent = enc_block(latent, film_vector)

        for token_sequence in text_conds:
            lat = latent
            for i, dec_block in enumerate(self.dec_blocks):
                lat = dec_block(lat, film_vector, token_sequence)
            eps = self.latent_to_epsilon(lat)
            epsilon_list.append(eps)

        return epsilon_list
