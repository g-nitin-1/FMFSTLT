#!/usr/bin/env python3
"""Causal patch-token foundation v1.5 model for multitask decision training."""

from __future__ import annotations

import torch
from torch import nn

from foundation_model import TraceFoundationConfig, patchify_traces


class CausalPatchFoundationV15(nn.Module):
    """All-decision causal patch encoder with throughput, uncertainty, and stop heads."""

    def __init__(self, config: TraceFoundationConfig, num_speed_tiers: int = 5) -> None:
        super().__init__()
        self.config = config
        self.num_speed_tiers = num_speed_tiers

        self.patch_embedding = nn.Sequential(
            nn.Linear(config.patch_dim, config.patch_hidden_dim),
            nn.GELU(),
            nn.Linear(config.patch_hidden_dim, config.d_model),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, config.num_patches, config.d_model))

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
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def causal_mask(self, token_count: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((token_count, token_count), 0.0, device=device)
        future = torch.triu(torch.ones(token_count, token_count, dtype=torch.bool, device=device), diagonal=1)
        return mask.masked_fill(future, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        bucket_mask: torch.Tensor,
        *,
        decision_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        patches, patch_valid = patchify_traces(x, bucket_mask, patch_size=self.config.patch_size)
        tokens = self.patch_embedding(patches)
        tokens = tokens.masked_fill(~patch_valid.bool().unsqueeze(-1), 0.0)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        tokens = tokens.masked_fill(~patch_valid.bool().unsqueeze(-1), 0.0)

        encoded = self.encoder(
            tokens,
            mask=self.causal_mask(tokens.shape[1], tokens.device),
        )
        encoded = self.norm(encoded)
        encoded = encoded.masked_fill(~patch_valid.bool().unsqueeze(-1), 0.0)

        if decision_valid_mask is None:
            final_valid = patch_valid.bool()
        else:
            final_valid = decision_valid_mask.bool()
        final_indices = torch.clamp(final_valid.long().sum(dim=1) - 1, min=0)
        final_state = encoded[torch.arange(encoded.shape[0], device=encoded.device), final_indices]

        return {
            "decision_states": encoded,
            "decision_valid_mask": final_valid,
            "patch_valid": patch_valid,
            "throughput_mu": self.throughput_mu_head(encoded).squeeze(-1),
            "throughput_log_var": self.throughput_log_var_head(encoded).squeeze(-1),
            "stop_logits": self.stop_head(encoded).squeeze(-1),
            "final_mu": self.final_mu_head(final_state).squeeze(-1),
            "final_log_var": self.final_log_var_head(final_state).squeeze(-1),
            "speed_logits": self.speed_head(final_state),
        }


def load_v1_encoder_weights(
    model: CausalPatchFoundationV15,
    checkpoint_state: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    """Load compatible v1 encoder weights into the causal v1.5 encoder.

    v1 checkpoints store the encoder under ``encoder.*`` and include a CLS
    position at index 0. v1.5 has no CLS token, so patch positions load from
    ``position_embedding[:, 1:]``.
    """

    translated: dict[str, torch.Tensor] = {}
    for key, value in checkpoint_state.items():
        if key.startswith("encoder.patch_embedding."):
            translated[key.removeprefix("encoder.")] = value
        elif key == "encoder.position_embedding":
            if value.shape[1] >= model.config.num_patches + 1:
                translated["position_embedding"] = value[:, 1 : model.config.num_patches + 1, :]
        elif key.startswith("encoder.encoder."):
            translated[key.replace("encoder.encoder.", "encoder.", 1)] = value
        elif key.startswith("encoder.norm."):
            translated[key.removeprefix("encoder.")] = value

    missing, unexpected = model.load_state_dict(translated, strict=False)
    expected_missing_prefixes = (
        "throughput_mu_head.",
        "throughput_log_var_head.",
        "stop_head.",
        "final_mu_head.",
        "final_log_var_head.",
        "speed_head.",
    )
    unexpected_filtered = list(unexpected)
    missing_filtered = [
        key
        for key in missing
        if not key.startswith(expected_missing_prefixes)
    ]
    return missing_filtered, unexpected_filtered


def load_v1_throughput_head_weights(
    model: CausalPatchFoundationV15,
    checkpoint_state: dict[str, torch.Tensor],
) -> list[str]:
    """Copy a v1 throughput regression head into v1.5 throughput mean heads."""

    required_keys = [
        "head.0.weight",
        "head.0.bias",
        "head.3.weight",
        "head.3.bias",
    ]
    missing_source = [key for key in required_keys if key not in checkpoint_state]
    if missing_source:
        return missing_source

    target_heads = (model.throughput_mu_head, model.final_mu_head)
    with torch.no_grad():
        for head in target_heads:
            head[0].weight.copy_(checkpoint_state["head.0.weight"])
            head[0].bias.copy_(checkpoint_state["head.0.bias"])
            head[3].weight.copy_(checkpoint_state["head.3.weight"])
            head[3].bias.copy_(checkpoint_state["head.3.bias"])
    return []
