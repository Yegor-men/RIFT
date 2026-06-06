import torch
from torch import nn


class MultiHeadLinearAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            feature_map: str = "elu",
            gated: bool = True,
            eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.feature_map = str(feature_map)
        self.gated = bool(gated)
        self.eps = float(eps)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        if self.gated:
            self.gate_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _positive_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "softmax":
            return nn.functional.softmax(x, dim=-1)
        if self.feature_map == "relu2":
            return nn.functional.relu(x).square() + self.eps
        if self.feature_map == "exp":
            return torch.exp(x.clamp(min=-15.0, max=15.0)) + self.eps
        if self.feature_map == "elu":
            return nn.functional.elu(x) + 1.0 + self.eps
        raise ValueError(f"Unknown linear-attention feature map: {self.feature_map}")

    def forward(self, query_embed: torch.Tensor, key_embed: torch.Tensor, value: torch.Tensor):
        batch, query_len, _ = query_embed.shape
        _, key_len, _ = key_embed.shape

        q = self.q_proj(query_embed).view(batch, query_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_embed).view(batch, key_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch, key_len, self.num_heads, self.head_dim)

        q = self._positive_features(q).transpose(1, 2)
        k = self._positive_features(k).transpose(1, 2)
        v = v.transpose(1, 2)

        kv_sum = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        k_sum = k.sum(dim=2)

        numerator = torch.einsum("b h q d, b h d e -> b h q e", q, kv_sum)
        denominator = torch.einsum("b h q d, b h d -> b h q", q, k_sum).unsqueeze(-1) + 1e-8

        out = (numerator / denominator).transpose(1, 2).reshape(batch, query_len, -1)
        if self.gated:
            out = out * torch.sigmoid(self.gate_proj(query_embed))
        return self.dropout(self.out_proj(out))


class ImageNorm(nn.Module):
    def __init__(self, num_channels: int, affine: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, elementwise_affine=affine)

    def forward(self, x):
        x = torch.movedim(x, -3, -1)
        x = self.norm(x)
        return torch.movedim(x, -1, -3)


class GRN(nn.Module):
    """Global Response Normalization: global channel competition without spatial mixing."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
        normalized_response = response / (response.mean(dim=-3, keepdim=True) + self.eps)
        gamma = self.gamma.unsqueeze(-1).unsqueeze(-1)
        beta = self.beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * (x * normalized_response) + beta + x


class PosEmbed2d(nn.Module):
    def __init__(self, num_frequencies: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.num_frequencies = int(num_frequencies)

        powers = torch.arange(self.num_frequencies, dtype=torch.float32)
        frequencies = 2.0 ** powers  # Slowest wave changes by 1 radian across the coordinate span.
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
        return torch.stack([xx, yy], dim=0)

    def make_grid(self, batch_size: int, h: int, w: int, relative: bool):
        grid = self._make_grid(h, w, relative).to(self.frequencies.device)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if self.training:
            if relative:
                sigma = 1.0 / (2 * max(h, w))
                jitter_x = torch.normal(mean=0.0, std=sigma, size=(batch_size, 1, h, w), device=grid.device)
                jitter_y = torch.normal(mean=0.0, std=sigma, size=(batch_size, 1, h, w), device=grid.device)
            else:
                jitter_x = torch.normal(mean=0.0, std=1.0 / (2 * w), size=(batch_size, 1, h, w), device=grid.device)
                jitter_y = torch.normal(mean=0.0, std=1.0 / (2 * h), size=(batch_size, 1, h, w), device=grid.device)
            grid = grid + torch.cat([jitter_x, jitter_y], dim=1)

        projected = grid.unsqueeze(-1) * self.frequencies.view(1, 1, 1, 1, -1)
        sin_feat = torch.sin(projected)
        cos_feat = torch.cos(projected)

        sin_channels = sin_feat.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, 2 * self.num_frequencies, h, w)
        cos_channels = cos_feat.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, 2 * self.num_frequencies, h, w)
        return self.norm(torch.cat([sin_channels, cos_channels], dim=1))

    def forward(self, batch_size: int, h: int, w: int):
        rel_pos_map = self.make_grid(batch_size, h, w, True)
        abs_pos_map = self.make_grid(batch_size, h, w, False)
        return torch.cat([rel_pos_map, abs_pos_map], dim=-3)


class ImageFFN(nn.Module):
    def __init__(self, d_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_channels, 4 * d_channels, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(4 * d_channels, d_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, d_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_channels % num_heads == 0, f"d_channels ({d_channels}) must be divisible by num_heads ({num_heads})"
        self.mha = nn.MultiheadAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, image, text_tokens):
        b, d, h, w = image.shape
        query = image.permute(0, 2, 3, 1).contiguous().view(b, h * w, d)
        attn_out, _ = self.mha(query, text_tokens, text_tokens, need_weights=False)
        return attn_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()


class SelfAttention(nn.Module):
    def __init__(self, d_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_channels % num_heads == 0, f"d_channels ({d_channels}) must be divisible by num_heads ({num_heads})"
        self.mha = nn.MultiheadAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, image):
        b, d, h, w = image.shape
        flat_img = image.flatten(2).transpose(1, 2)
        attn_out, _ = self.mha(flat_img, flat_img, flat_img, need_weights=False)
        return attn_out.transpose(1, 2).view(b, d, h, w)


class LinearCrossAttention(nn.Module):
    def __init__(self, d_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_channels % num_heads == 0, f"d_channels ({d_channels}) must be divisible by num_heads ({num_heads})"
        self.mha = MultiHeadLinearAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            dropout=dropout,
            feature_map="elu",
        )

    def forward(self, image, text_tokens):
        b, d, h, w = image.shape
        query = image.permute(0, 2, 3, 1).contiguous().view(b, h * w, d)
        attn_out = self.mha(query, text_tokens, text_tokens)
        return attn_out.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()


class LinearSelfAttention(nn.Module):
    def __init__(self, d_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_channels % num_heads == 0, f"d_channels ({d_channels}) must be divisible by num_heads ({num_heads})"
        self.mha = MultiHeadLinearAttention(
            embed_dim=d_channels,
            num_heads=num_heads,
            dropout=dropout,
            feature_map="elu",
        )

    def forward(self, image):
        b, d, h, w = image.shape
        flat_img = image.flatten(2).transpose(1, 2)
        attn_out = self.mha(flat_img, flat_img, flat_img)
        return attn_out.transpose(1, 2).view(b, d, h, w)


class EncBlock(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            self_attn_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            linear_attention: bool = False,
    ):
        super().__init__()
        self.d_channels = int(d_channels)
        self_attn_cls = LinearSelfAttention if linear_attention else SelfAttention

        self.self_norm = nn.Sequential(GRN(d_channels), ImageNorm(d_channels))
        self.self_attn = self_attn_cls(d_channels=d_channels, num_heads=num_heads, dropout=self_attn_dropout)
        self.self_scalar = nn.Parameter(torch.ones(d_channels))

        self.ffn_norm = nn.Sequential(GRN(d_channels), ImageNorm(d_channels))
        self.ffn = ImageFFN(d_channels, ffn_dropout)
        self.ffn_scalar = nn.Parameter(torch.ones(d_channels))

        self.final_scalar = nn.Parameter(torch.ones(d_channels) * 0.1)

    def forward(self, image):
        working_image = image

        self_out = self.self_attn(self.self_norm(working_image))
        working_image = working_image + self_out * self.self_scalar.view(1, self.d_channels, 1, 1)

        ffn_out = self.ffn(self.ffn_norm(working_image))
        working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

        return image + working_image * self.final_scalar.view(1, self.d_channels, 1, 1)


class DecBlock(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            skip_channels: int = 0,
            self_attn_dropout: float = 0.0,
            cross_attn_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            linear_attention: bool = False,
    ):
        super().__init__()
        self.d_channels = int(d_channels)
        self.skip_channels = int(skip_channels)
        if self.skip_channels not in (0, self.d_channels):
            raise ValueError(f"DecBlock skip_channels must be 0 or d_channels ({self.d_channels}), got {self.skip_channels}")

        self.work_channels = self.d_channels + self.skip_channels
        self.work_heads = int(num_heads) * (2 if self.skip_channels > 0 else 1)
        assert self.work_channels % self.work_heads == 0, (
            f"work_channels ({self.work_channels}) must be divisible by work_heads ({self.work_heads})"
        )

        self_attn_cls = LinearSelfAttention if linear_attention else SelfAttention
        cross_attn_cls = LinearCrossAttention if linear_attention else CrossAttention

        self.text_proj = nn.Identity() if self.work_channels == self.d_channels else nn.Linear(self.d_channels, self.work_channels)
        self.cross_norm = nn.Sequential(GRN(self.work_channels), ImageNorm(self.work_channels))
        self.cross_attn = cross_attn_cls(d_channels=self.work_channels, num_heads=self.work_heads, dropout=cross_attn_dropout)
        self.cross_scalar = nn.Parameter(torch.ones(self.work_channels))

        self.self_norm = nn.Sequential(GRN(self.work_channels), ImageNorm(self.work_channels))
        self.self_attn = self_attn_cls(d_channels=self.work_channels, num_heads=self.work_heads, dropout=self_attn_dropout)
        self.self_scalar = nn.Parameter(torch.ones(self.work_channels))

        self.ffn_norm = nn.Sequential(GRN(self.work_channels), ImageNorm(self.work_channels))
        self.ffn = ImageFFN(self.work_channels, ffn_dropout)
        self.ffn_scalar = nn.Parameter(torch.ones(self.work_channels))

        self.output_proj = nn.Conv2d(self.work_channels, self.d_channels, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        with torch.no_grad():
            eye = torch.eye(self.d_channels)
            self.output_proj.weight[:, :self.d_channels, 0, 0].copy_(eye)
        self.final_scalar = nn.Parameter(torch.ones(d_channels) * 0.1)

    def _make_working_image(self, image, skip_image):
        if self.skip_channels <= 0:
            return image
        assert skip_image is not None, "Decoder block was built with skip channels but no skip image was passed"
        assert skip_image.shape[-3] == self.skip_channels, (
            f"Expected skip with {self.skip_channels} channels, got {skip_image.shape[-3]}"
        )
        return torch.cat([image, skip_image], dim=-3)

    def forward(self, image, text_tokens, skip_image=None):
        working_image = self._make_working_image(image, skip_image)

        cross_out = self.cross_attn(self.cross_norm(working_image), self.text_proj(text_tokens))
        working_image = working_image + cross_out * self.cross_scalar.view(1, self.work_channels, 1, 1)

        self_out = self.self_attn(self.self_norm(working_image))
        working_image = working_image + self_out * self.self_scalar.view(1, self.work_channels, 1, 1)

        ffn_out = self.ffn(self.ffn_norm(working_image))
        working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.work_channels, 1, 1)

        compressed_image = self.output_proj(working_image)
        return image + compressed_image * self.final_scalar.view(1, self.d_channels, 1, 1)


class R2ID(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            block_count: int | None = None,
            c_channels: int = 1,
            pos_freq: int = 16,
            time_freq: int = 10,  # Kept for config/API compatibility; time conditioning is disabled.
            enc_blocks: int | None = None,
            dec_blocks: int | None = None,
            time_high_freq: int | None = None,
            time_low_freq: int = 0,
            film_dim: int | None = None,
            self_attn_dropout: float = 0.0,
            cross_attn_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            linear_attention: bool = True,
            skip_fusion: bool = True,
            velocity_output_scale: float = 1.0,
    ):
        super().__init__()
        if time_high_freq is not None:
            time_freq = int(time_high_freq) + int(time_low_freq)

        if block_count is not None:
            if enc_blocks is not None and int(enc_blocks) != int(block_count):
                raise ValueError("enc_blocks must match block_count when both are provided")
            if dec_blocks is not None and int(dec_blocks) != int(block_count):
                raise ValueError("dec_blocks must match block_count when both are provided")
            enc_blocks = dec_blocks = int(block_count)
        elif enc_blocks is None and dec_blocks is None:
            enc_blocks = dec_blocks = 4
        elif enc_blocks is None:
            enc_blocks = int(dec_blocks)
        elif dec_blocks is None:
            dec_blocks = int(enc_blocks)

        if int(enc_blocks) != int(dec_blocks):
            raise ValueError("R2ID expects one block_count: enc_blocks and dec_blocks must match")

        self.c_channels = int(c_channels)
        self.d_channels = int(d_channels)
        self.block_count = int(enc_blocks)
        self.num_enc_blocks = self.block_count
        self.num_dec_blocks = self.block_count
        self.num_heads = int(num_heads)
        self.num_time_frequencies = int(time_freq)
        self.num_pos_frequencies = int(pos_freq)
        self.film_dim = int(film_dim) if film_dim is not None else int(d_channels)
        self.linear_attention = bool(linear_attention)
        self.skip_fusion = bool(skip_fusion)
        self.velocity_output_scale = float(velocity_output_scale)

        input_channels = self.num_pos_frequencies * 4 * 2 + self.c_channels
        self.input_channels = int(input_channels)

        self.proj_to_latent = nn.Conv2d(input_channels, self.d_channels, 1)
        self.latent_to_velocity = nn.Conv2d(self.d_channels, self.c_channels, 1)
        nn.init.zeros_(self.latent_to_velocity.weight)
        nn.init.zeros_(self.latent_to_velocity.bias)

        self.pos_embed = PosEmbed2d(pos_freq)
        self.enc_blocks = nn.ModuleList([
            EncBlock(
                d_channels=self.d_channels,
                num_heads=self.num_heads,
                self_attn_dropout=self_attn_dropout,
                ffn_dropout=ffn_dropout,
                linear_attention=linear_attention,
            ) for _ in range(self.block_count)
        ])

        decoder_skip_channels = self.d_channels if self.skip_fusion else 0
        self.dec_blocks = nn.ModuleList([
            DecBlock(
                d_channels=self.d_channels,
                num_heads=self.num_heads,
                skip_channels=decoder_skip_channels,
                self_attn_dropout=self_attn_dropout,
                cross_attn_dropout=cross_attn_dropout,
                ffn_dropout=ffn_dropout,
                linear_attention=linear_attention,
            ) for _ in range(self.block_count)
        ])

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_pos_channels = self.num_pos_frequencies * 2 * 2 * 2
        total_channels = total_pos_channels + self.c_channels

        print(f"Trainable parameters: {total:,}")
        print(f"Block count: {self.block_count}")
        print(f"Channels for color/positioning: {self.c_channels}/{total_pos_channels}, total: {total_channels}")
        print(f"Attention: {'linear gated' if self.linear_attention else 'full'}")
        print("Time conditioning: disabled")
        print("Encoder text conditioning: disabled")
        print("Decoder text conditioning: enabled")
        if self.skip_fusion:
            print(
                f"Decoder skip attention: enabled, encoder_input_stack={self.num_enc_blocks}, "
                f"decoder_work_channels={self.d_channels * 2}, decoder_heads={self.num_heads * 2}, "
                f"raw_input_skip=no"
            )
        else:
            print("Decoder skip attention: disabled")

    def forward(self, image: torch.Tensor, time: torch.Tensor, text_conds: list[torch.Tensor]):
        del time  # Time conditioning is intentionally disabled.
        assert image.ndim == 4, "Image must be batch, tensor shape of [B, C, H, W]"
        batch, _, height, width = image.shape

        pos_map = self.pos_embed(batch, height, width)
        latent = self.proj_to_latent(torch.cat([image, pos_map], dim=-3))

        encoder_inputs = []
        for enc_block in self.enc_blocks:
            if self.skip_fusion:
                encoder_inputs.append(latent)
            latent = enc_block(latent)

        if self.skip_fusion:
            skip_sources = list(reversed(encoder_inputs))
        else:
            skip_sources = [None for _ in self.dec_blocks]

        velocity_list = []
        for token_sequence in text_conds:
            lat = latent
            for i, dec_block in enumerate(self.dec_blocks):
                lat = dec_block(lat, token_sequence, skip_sources[i])
            velocity = torch.tanh(self.latent_to_velocity(lat)) * self.velocity_output_scale
            velocity_list.append(velocity)

        return velocity_list


class R2IDLinear(R2ID):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("skip_fusion", True)
        kwargs["linear_attention"] = True
        super().__init__(*args, **kwargs)
