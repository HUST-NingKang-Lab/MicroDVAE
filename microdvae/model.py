from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=pad_mask)


class InferenceVectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, use_cosine: bool = True):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.use_cosine = use_cosine

    def encode(self, z_e: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        flat = z_e.reshape(-1, self.code_dim)
        if self.use_cosine:
            flat_norm = F.normalize(flat, dim=1, eps=1e-8)
            codebook_norm = F.normalize(self.codebook.weight, dim=1, eps=1e-8)
            logits = flat_norm @ codebook_norm.T
        else:
            z_norm = (flat ** 2).sum(dim=1, keepdim=True)
            e_norm = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)
            logits = -(z_norm - 2 * flat @ self.codebook.weight.T + e_norm)
        indices = logits.argmax(dim=1).view(z_e.size(0), z_e.size(1))
        if mask is not None:
            indices = indices.masked_fill(mask == 0, 0)
        return indices


class MicroDVAEModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_enc_layers: int = 6,
        codebook_size: int = 8192,
        code_dim: int = 256,
        dropout: float = 0.1,
        ff_mult: int = 4,
        **_: object,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.encoder = TransformerEncoderStack(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_enc_layers,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.pre_vq = nn.Sequential(
            nn.Linear(d_model, code_dim),
            nn.LayerNorm(code_dim),
        )
        self.vq = InferenceVectorQuantizer(codebook_size, code_dim, use_cosine=True)

    @torch.no_grad()
    def get_encoder_embeddings(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, pad_mask=(mask == 0))
        h = self.enc_norm(h)
        return self.pre_vq(h)

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z_e = self.get_encoder_embeddings(x, mask)
        return self.vq.encode(z_e, mask=mask)

    @torch.no_grad()
    def lookup_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.vq.codebook.weight)
