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

        sin_channels = sin_feat.permute(0, 1, 4, 2, 3).contiguous().view(
            batch_size,
            2 * self.num_frequencies,
            h,
            w,
        )
        cos_channels = cos_feat.permute(0, 1, 4, 2, 3).contiguous().view(
            batch_size,
            2 * self.num_frequencies,
            h,
            w,
        )
        return self.norm(torch.cat([sin_channels, cos_channels], dim=1))

    def forward(self, batch_size: int, h: int, w: int):
        rel_pos_map = self.make_grid(batch_size, h, w, True)
        abs_pos_map = self.make_grid(batch_size, h, w, False)
        return torch.cat([rel_pos_map, abs_pos_map], dim=-3)


class ContTimeEmbed(nn.Module):
    def __init__(self, num_frequencies: int):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        powers = torch.arange(self.num_frequencies, dtype=torch.float32)
        frequencies = 2.0 ** powers  # Slowest wave changes by 1 radian across t in [0, 1].
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.norm = nn.LayerNorm(2 * self.num_frequencies, elementwise_affine=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.to(device=self.frequencies.device, dtype=self.frequencies.dtype).flatten()
        projected = time.unsqueeze(1) * self.frequencies.view(1, -1)
        features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        return self.norm(features)


class ImageAdaLN(nn.Module):
    def __init__(self, time_dim: int, num_channels: int):
        super().__init__()
        self.norm = nn.Sequential(
            GRN(num_channels),
            ImageNorm(num_channels),
        )
        self.to_scale_shift = nn.Linear(time_dim, 2 * num_channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, image: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(time_cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return self.norm(image) * (1.0 + scale) + shift


class ImageFFN(nn.Module):
    def __init__(self, d_channels: int, dropout: float = 0.0):
        super().__init__()
        hidden_channels = 4 * d_channels
        self.expand = nn.Conv2d(d_channels, hidden_channels, 1)
        self.to_gamma = nn.Conv2d(d_channels, hidden_channels, 1)
        self.to_beta = nn.Conv2d(d_channels, hidden_channels, 1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.compress = nn.Conv2d(hidden_channels, d_channels, 1)
        self.adaptive_enabled = True

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(x)
        gamma = self.to_gamma(x)
        beta = self.to_beta(x)
        adapted = expanded * (1.0 + gamma) + beta if self.adaptive_enabled else expanded
        return self.compress(self.dropout(self.activation(adapted)))


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

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
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

    def forward(self, image: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, d, h, w = image.shape
        flat_img = image.flatten(2).transpose(1, 2)
        attn_out = self.mha(flat_img, flat_img, flat_img)
        return attn_out.transpose(1, 2).view(b, d, h, w)


class DecBlock(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            time_dim: int,
            self_attn_dropout: float = 0.0,
            cross_attn_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            linear_attention: bool = False,
    ):
        super().__init__()
        self.d_channels = int(d_channels)
        self.num_heads = int(num_heads)

        self_attn_cls = LinearSelfAttention if linear_attention else SelfAttention
        cross_attn_cls = LinearCrossAttention if linear_attention else CrossAttention

        self.self_ada = ImageAdaLN(time_dim, self.d_channels)
        self.self_attn = self_attn_cls(
            d_channels=self.d_channels,
            num_heads=self.num_heads,
            dropout=self_attn_dropout,
        )
        self.self_scalar = nn.Parameter(torch.ones(self.d_channels))

        self.cross_ada = ImageAdaLN(time_dim, self.d_channels)
        self.cross_attn = cross_attn_cls(
            d_channels=self.d_channels,
            num_heads=self.num_heads,
            dropout=cross_attn_dropout,
        )
        self.cross_scalar = nn.Parameter(torch.ones(self.d_channels))

        self.ffn_ada = ImageAdaLN(time_dim, self.d_channels)
        self.ffn = ImageFFN(self.d_channels, ffn_dropout)
        self.ffn_scalar = nn.Parameter(torch.ones(self.d_channels))

        self.final_scalar = nn.Parameter(torch.ones(self.d_channels) * 0.1)

    def forward(
            self,
            image: torch.Tensor,
            time_cond: torch.Tensor,
            text_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        working_image = image

        self_out = self.self_attn(self.self_ada(working_image, time_cond))
        working_image = working_image + self_out * self.self_scalar.view(1, self.d_channels, 1, 1)

        if text_tokens is not None:
            cross_out = self.cross_attn(self.cross_ada(working_image, time_cond), text_tokens)
            working_image = working_image + cross_out * self.cross_scalar.view(1, self.d_channels, 1, 1)

        ffn_out = self.ffn(self.ffn_ada(working_image, time_cond))
        working_image = working_image + ffn_out * self.ffn_scalar.view(1, self.d_channels, 1, 1)

        return image + working_image * self.final_scalar.view(1, self.d_channels, 1, 1)


class RIFT(nn.Module):
    def __init__(
            self,
            d_channels: int,
            num_heads: int,
            block_count: int | None = None,
            c_channels: int = 1,
            pos_freq: int = 16,
            time_freq: int = 10,
            self_attn_dropout: float = 0.0,
            cross_attn_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            linear_attention: bool = True,
    ):
        super().__init__()
        self.c_channels = int(c_channels)
        self.d_channels = int(d_channels)
        self.block_count = 8 if block_count is None else int(block_count)
        self.num_heads = int(num_heads)
        self.num_time_frequencies = int(time_freq)
        self.num_pos_frequencies = int(pos_freq)
        self.linear_attention = bool(linear_attention)

        self.pos_channels = self.num_pos_frequencies * 4 * 2
        self.input_channels = self.c_channels + self.pos_channels

        self.image_proj = nn.Conv2d(self.c_channels, self.d_channels, 1)
        self.pos_proj = nn.Conv2d(self.pos_channels, self.d_channels, 1)
        self.features_to_velocity = nn.Conv2d(self.d_channels, self.c_channels, 1)
        nn.init.zeros_(self.features_to_velocity.weight)
        nn.init.zeros_(self.features_to_velocity.bias)

        self.pos_embed = PosEmbed2d(pos_freq)
        self.time_embed = ContTimeEmbed(time_freq)
        self.time_proj = nn.Sequential(
            nn.Linear(2 * self.num_time_frequencies, self.d_channels),
            nn.SiLU(),
            nn.Linear(self.d_channels, self.d_channels),
            nn.SiLU(),
        )
        self.dec_blocks = nn.ModuleList([
            DecBlock(
                d_channels=self.d_channels,
                num_heads=self.num_heads,
                time_dim=self.d_channels,
                self_attn_dropout=self_attn_dropout,
                cross_attn_dropout=cross_attn_dropout,
                ffn_dropout=ffn_dropout,
                linear_attention=linear_attention,
            ) for _ in range(self.block_count)
        ])

    def print_model_summary(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total:,}")
        print(f"Block count: {self.block_count}")
        print(f"Channels for color/positioning: {self.c_channels}/{self.pos_channels}, total: {self.input_channels}")
        print(f"Attention: {'linear gated' if self.linear_attention else 'full'}")
        print("Time conditioning: enabled")
        print("Encoder blocks: disabled")
        print("Text conditioning: standard cross attention + CFG-compatible outputs")
        print("Output: flow velocity x1-x0 in [-1, 1]")

    @staticmethod
    def _normalize_text_conditions(text_conds):
        if text_conds is None:
            return [None]
        if torch.is_tensor(text_conds):
            return [text_conds]

        normalized = []
        for item in text_conds:
            if item is None:
                normalized.append(None)
            elif torch.is_tensor(item):
                normalized.append(item)
            else:
                if len(item) < 1:
                    raise ValueError("Conditioning items must be tensors or tuples with a tensor first")
                normalized.append(item[0])
        return normalized or [None]

    def forward(
            self,
            image: torch.Tensor,
            time: torch.Tensor,
            text_conds: list[torch.Tensor] | torch.Tensor | None,
    ):
        assert image.ndim == 4, "Image must be batch, tensor shape of [B, C, H, W]"
        batch, _, height, width = image.shape

        pos_map = self.pos_embed(batch, height, width)
        features = self.image_proj(image) + self.pos_proj(pos_map)
        time_cond = self.time_proj(self.time_embed(time.to(device=image.device, dtype=image.dtype)))
        text_conditions = self._normalize_text_conditions(text_conds)

        velocity_predictions = []
        for token_sequence in text_conditions:
            if token_sequence is not None:
                token_sequence = token_sequence.to(device=image.device, dtype=image.dtype)
            conditioned_features = features
            for dec_block in self.dec_blocks:
                conditioned_features = dec_block(conditioned_features, time_cond, token_sequence)
            predicted_velocity = torch.tanh(self.features_to_velocity(conditioned_features))
            velocity_predictions.append(predicted_velocity)
        return velocity_predictions


class RIFTLinear(RIFT):
    def __init__(self, *args, **kwargs):
        kwargs["linear_attention"] = True
        super().__init__(*args, **kwargs)
