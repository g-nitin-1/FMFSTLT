#!/usr/bin/env python3
"""Patch-based foundation encoder for normalized M-Lab throughput traces."""

from __future__ import annotations

import os
from dataclasses import dataclass

if "TMPDIR" not in os.environ:
    for candidate in ("/dev/shm", "/tmp", "/var/tmp", "/usr/tmp"):
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            os.environ["TMPDIR"] = candidate
            break

import torch
from torch import nn


@dataclass(frozen=True)
class TraceFoundationConfig:
    feature_dim: int = 13
    max_sequence_buckets: int = 100
    patch_size: int = 5
    d_model: int = 256
    patch_hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        if self.max_sequence_buckets % self.patch_size != 0:
            raise ValueError(
                "max_sequence_buckets must be divisible by patch_size: "
                f"{self.max_sequence_buckets} vs {self.patch_size}"
            )
        return self.max_sequence_buckets // self.patch_size

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.feature_dim

    def to_dict(self) -> dict[str, int | float]:
        return {
            "feature_dim": self.feature_dim,
            "max_sequence_buckets": self.max_sequence_buckets,
            "patch_size": self.patch_size,
            "num_patches": self.num_patches,
            "patch_dim": self.patch_dim,
            "d_model": self.d_model,
            "patch_hidden_dim": self.patch_hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        }


def patchify_traces(
    x: torch.Tensor,
    bucket_mask: torch.Tensor,
    *,
    patch_size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert bucket traces to non-overlapping patches and validity masks.

    Args:
        x: Normalized trace tensor with shape ``[batch, buckets, features]``.
        bucket_mask: Boolean observed-bucket mask with shape ``[batch, buckets]``.
        patch_size: Number of 100 ms buckets per patch.

    Returns:
        ``patches`` with shape ``[batch, patches, patch_size * features]`` and
        ``patch_valid`` with shape ``[batch, patches]``. Invalid patch vectors are
        zeroed so forward-filled values cannot leak through unmasked paths.
    """

    if x.ndim != 3:
        raise ValueError(f"expected x rank 3 [batch,buckets,features], got {x.ndim}")
    if bucket_mask.ndim != 2:
        raise ValueError(f"expected bucket_mask rank 2 [batch,buckets], got {bucket_mask.ndim}")
    if x.shape[0] != bucket_mask.shape[0] or x.shape[1] != bucket_mask.shape[1]:
        raise ValueError(
            "x and bucket_mask must agree on batch and bucket dimensions: "
            f"{tuple(x.shape)} vs {tuple(bucket_mask.shape)}"
        )
    if x.shape[1] % patch_size != 0:
        raise ValueError(f"bucket length {x.shape[1]} is not divisible by patch_size={patch_size}")

    batch_size, bucket_count, feature_dim = x.shape
    patch_count = bucket_count // patch_size

    patches = x.reshape(batch_size, patch_count, patch_size, feature_dim)
    patches = patches.reshape(batch_size, patch_count, patch_size * feature_dim)

    patch_valid = bucket_mask.bool().reshape(batch_size, patch_count, patch_size).any(dim=-1)
    patches = patches.masked_fill(~patch_valid.unsqueeze(-1), 0.0)
    return patches, patch_valid


class TraceFoundationEncoder(nn.Module):
    """Shared patch encoder used by pretraining and downstream heads."""

    def __init__(self, config: TraceFoundationConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Sequential(
            nn.Linear(config.patch_dim, config.patch_hidden_dim),
            nn.GELU(),
            nn.Linear(config.patch_hidden_dim, config.d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        bucket_mask: torch.Tensor,
        *,
        masked_patch_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        patches, patch_valid = patchify_traces(x, bucket_mask, patch_size=self.config.patch_size)
        patch_tokens = self.patch_embedding(patches)

        if masked_patch_mask is not None:
            if masked_patch_mask.shape != patch_valid.shape:
                raise ValueError(
                    "masked_patch_mask must match patch validity shape: "
                    f"{tuple(masked_patch_mask.shape)} vs {tuple(patch_valid.shape)}"
                )
            mask_token = self.mask_token.expand(patch_tokens.shape[0], patch_tokens.shape[1], -1)
            patch_tokens = torch.where(masked_patch_mask.bool().unsqueeze(-1), mask_token, patch_tokens)

        cls_tokens = self.cls_token.expand(patch_tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]

        cls_valid = torch.ones((patch_valid.shape[0], 1), dtype=torch.bool, device=patch_valid.device)
        token_valid = torch.cat([cls_valid, patch_valid], dim=1)
        key_padding_mask = ~token_valid

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        return {
            "encoded": encoded,
            "cls": encoded[:, 0],
            "patch_tokens": encoded[:, 1:],
            "patch_valid": patch_valid,
            "patches": patches,
            "key_padding_mask": key_padding_mask,
        }


class MaskedPatchReconstructionModel(nn.Module):
    """Foundation encoder with a masked-patch reconstruction pretraining head."""

    def __init__(self, config: TraceFoundationConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TraceFoundationEncoder(config)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.d_model, config.patch_hidden_dim),
            nn.GELU(),
            nn.Linear(config.patch_hidden_dim, config.patch_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        bucket_mask: torch.Tensor,
        *,
        masked_patch_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(x, bucket_mask, masked_patch_mask=masked_patch_mask)
        reconstruction = self.reconstruction_head(outputs["patch_tokens"])
        outputs["reconstruction"] = reconstruction
        return outputs


class ThroughputRegressionModel(nn.Module):
    """Foundation encoder with a CLS-pooled throughput regression head."""

    def __init__(self, config: TraceFoundationConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TraceFoundationEncoder(config)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, x: torch.Tensor, bucket_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, bucket_mask)
        return self.head(outputs["cls"]).squeeze(-1)


class EarlyStopFoundationModel(nn.Module):
    """Foundation encoder with a CLS-pooled stop/continue classifier."""

    def __init__(self, config: TraceFoundationConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TraceFoundationEncoder(config)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, x: torch.Tensor, prefix_bucket_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, prefix_bucket_mask)
        return self.head(outputs["cls"]).squeeze(-1)


class SpeedTierClassificationModel(nn.Module):
    """Foundation encoder with a CLS-pooled speed-tier classification head."""

    def __init__(self, config: TraceFoundationConfig, num_classes: int) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.encoder = TraceFoundationEncoder(config)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, bucket_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, bucket_mask)
        return self.head(outputs["cls"])
