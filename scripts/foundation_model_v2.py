#!/usr/bin/env python3
"""Causal bucket-stem foundation v2 model for all-decision training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from foundation_model import TraceFoundationConfig


class CausalBucketFoundationV2(nn.Module):
    """Causal bucket stem followed by a causal decision-token Transformer."""

    def __init__(
        self,
        config: TraceFoundationConfig,
        *,
        bucket_hidden_dim: int = 128,
        stem_kernel_size: int = 5,
        num_speed_tiers: int = 5,
    ) -> None:
        super().__init__()
        if stem_kernel_size <= 0:
            raise ValueError("stem_kernel_size must be positive")
        if config.max_sequence_buckets % config.patch_size != 0:
            raise ValueError("max_sequence_buckets must be divisible by patch_size")

        self.config = config
        self.bucket_hidden_dim = bucket_hidden_dim
        self.stem_kernel_size = stem_kernel_size
        self.num_speed_tiers = num_speed_tiers

        self.bucket_conv = nn.Conv1d(
            in_channels=config.feature_dim,
            out_channels=bucket_hidden_dim,
            kernel_size=stem_kernel_size,
            stride=1,
            padding=0,
        )
        self.bucket_norm = nn.LayerNorm(bucket_hidden_dim)
        self.bucket_projection = nn.Sequential(
            nn.Linear(bucket_hidden_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.decision_position_embedding = nn.Parameter(torch.zeros(1, config.num_patches, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decision_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)

        self.throughput_mu_head = self._scalar_head()
        self.throughput_log_var_head = self._scalar_head()
        self.stop_head = self._scalar_head()
        self.final_mu_head = self._scalar_head()
        self.final_log_var_head = self._scalar_head()
        self.speed_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_speed_tiers),
        )
        self.reset_parameters()

    def _scalar_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, 1),
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.decision_position_embedding, mean=0.0, std=0.02)

    def to_v2_config_dict(self) -> dict[str, int]:
        return {
            "bucket_hidden_dim": int(self.bucket_hidden_dim),
            "stem_kernel_size": int(self.stem_kernel_size),
            "num_speed_tiers": int(self.num_speed_tiers),
        }

    def causal_mask(self, token_count: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((token_count, token_count), 0.0, device=device)
        future = torch.triu(torch.ones(token_count, token_count, dtype=torch.bool, device=device), diagonal=1)
        return mask.masked_fill(future, float("-inf"))

    def causal_bucket_stem(self, x: torch.Tensor, bucket_mask: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(~bucket_mask.bool().unsqueeze(-1), 0.0)
        channels_first = x.transpose(1, 2)
        channels_first = F.pad(channels_first, (self.stem_kernel_size - 1, 0))
        bucket_hidden = self.bucket_conv(channels_first).transpose(1, 2)
        bucket_hidden = F.gelu(bucket_hidden)
        bucket_hidden = self.bucket_norm(bucket_hidden)
        bucket_hidden = self.bucket_projection(bucket_hidden)
        return bucket_hidden.masked_fill(~bucket_mask.bool().unsqueeze(-1), 0.0)

    def forward(
        self,
        x: torch.Tensor,
        bucket_mask: torch.Tensor,
        *,
        decision_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bucket_tokens = self.causal_bucket_stem(x, bucket_mask)
        end_indices = torch.arange(
            self.config.patch_size - 1,
            self.config.max_sequence_buckets,
            self.config.patch_size,
            device=x.device,
        )
        decision_tokens = bucket_tokens.index_select(dim=1, index=end_indices)
        decision_tokens = decision_tokens + self.decision_position_embedding[:, : decision_tokens.shape[1], :]

        if decision_valid_mask is None:
            patch_bucket_mask = bucket_mask.bool().reshape(
                bucket_mask.shape[0], self.config.num_patches, self.config.patch_size
            )
            decision_valid = patch_bucket_mask.any(dim=-1)
        else:
            decision_valid = decision_valid_mask.bool()
        decision_tokens = decision_tokens.masked_fill(~decision_valid.unsqueeze(-1), 0.0)

        encoded = self.decision_encoder(
            decision_tokens,
            mask=self.causal_mask(decision_tokens.shape[1], decision_tokens.device),
        )
        encoded = self.norm(encoded)
        encoded = encoded.masked_fill(~decision_valid.unsqueeze(-1), 0.0)

        final_indices = torch.clamp(decision_valid.long().sum(dim=1) - 1, min=0)
        final_state = encoded[torch.arange(encoded.shape[0], device=encoded.device), final_indices]

        return {
            "decision_states": encoded,
            "decision_valid_mask": decision_valid,
            "patch_valid": decision_valid,
            "throughput_mu": self.throughput_mu_head(encoded).squeeze(-1),
            "throughput_log_var": self.throughput_log_var_head(encoded).squeeze(-1),
            "stop_logits": self.stop_head(encoded).squeeze(-1),
            "final_mu": self.final_mu_head(final_state).squeeze(-1),
            "final_log_var": self.final_log_var_head(final_state).squeeze(-1),
            "speed_logits": self.speed_head(final_state),
        }


def load_v1_throughput_head_weights(
    model: CausalBucketFoundationV2,
    checkpoint_state: dict[str, torch.Tensor],
) -> list[str]:
    """Copy a v1 throughput regression head into v2 throughput mean heads."""

    required_keys = [
        "head.0.weight",
        "head.0.bias",
        "head.3.weight",
        "head.3.bias",
    ]
    missing_source = [key for key in required_keys if key not in checkpoint_state]
    if missing_source:
        return missing_source

    with torch.no_grad():
        for head in (model.throughput_mu_head, model.final_mu_head):
            head[0].weight.copy_(checkpoint_state["head.0.weight"])
            head[0].bias.copy_(checkpoint_state["head.0.bias"])
            head[3].weight.copy_(checkpoint_state["head.3.weight"])
            head[3].bias.copy_(checkpoint_state["head.3.bias"])
    return []
