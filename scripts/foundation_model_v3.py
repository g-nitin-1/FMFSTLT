#!/usr/bin/env python3
"""FMNet v3 — bucket-level causal Transformer foundation model.

Key design choices (vs v1 patch / v2 patch+causal-stem):
  - 100 bucket tokens, no patch compression.
  - Causal attention; same architecture for pretraining and fine-tuning.
  - Per-decision heads at end-buckets {4, 9, 14, ..., 99}.
  - Joint per-decision throughput + stop heads share the encoder.
  - Causal next-bucket prediction objective for self-supervised pretraining.

Tasks supported:
  - Pretraining: causal next-bucket prediction (predict bucket t+1 from 0..t).
  - Fine-tuning multi-task: per-decision throughput + per-decision stop +
    final throughput + speed-tier classification.

Information argument vs XGBoost prefix predictor:
  At decision d the foundation receives the entire prefix [0..5d+4]
  (5d+5 buckets) through causal attention.  XGBoost gets a 20-bucket
  flat window.  For d=10 the foundation sees 55 buckets, XGBoost 20.
  No patch compression, so per-bucket detail is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

SPEED_TIER_ORDER = ("0-25", "25-100", "100-200", "200-400", "400+")
SPEED_TIER_TO_INDEX = {tier: idx for idx, tier in enumerate(SPEED_TIER_ORDER)}
NUM_SPEED_TIERS = len(SPEED_TIER_ORDER)


@dataclass
class FMNetV3Config:
    feature_dim: int = 13
    max_buckets: int = 100
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 8
    ff_dim: int = 1024
    dropout: float = 0.15
    head_dropout: float = 0.10
    num_speed_tiers: int = NUM_SPEED_TIERS

    def to_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "max_buckets": self.max_buckets,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "head_dropout": self.head_dropout,
            "num_speed_tiers": self.num_speed_tiers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FMNetV3Config":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FMNetV3(nn.Module):
    """Bucket-level causal Transformer with multi-task heads."""

    def __init__(self, config: FMNetV3Config) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.bucket_projection = nn.Linear(config.feature_dim, d)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_buckets, d)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.norm = nn.LayerNorm(d)

        self.throughput_mu_head = _MLPHead(d, d, 1, config.head_dropout)
        self.throughput_logvar_head = _MLPHead(d, d, 1, config.head_dropout)
        self.stop_head = _MLPHead(d, d, 1, config.head_dropout)
        self.speed_tier_head = _MLPHead(d, d, config.num_speed_tiers, config.head_dropout)
        self.next_bucket_head = _MLPHead(d, d, config.feature_dim, config.head_dropout)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode(self, x_full: torch.Tensor) -> torch.Tensor:
        """Encode [B,T,feature_dim] -> [B,T,d_model] with causal attention."""
        B, T, F = x_full.shape
        if F != self.config.feature_dim:
            raise ValueError(
                f"feature dim mismatch: expected {self.config.feature_dim}, got {F}"
            )
        if T > self.config.max_buckets:
            raise ValueError(
                f"sequence length {T} exceeds max_buckets {self.config.max_buckets}"
            )
        h = self.bucket_projection(x_full) + self.position_embedding[:, :T, :]
        mask = _causal_mask(T, x_full.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.norm(h)
        return h

    def gather_decisions(
        self, hidden: torch.Tensor, decision_buckets: torch.Tensor
    ) -> torch.Tensor:
        """Gather hidden states at per-test decision bucket indices.

        hidden:            [B, T, d_model]
        decision_buckets:  [B, D] int (clamped to <T)
        returns:           [B, D, d_model]
        """
        B, D = decision_buckets.shape
        clamped = decision_buckets.clamp(min=0, max=hidden.shape[1] - 1).long()
        batch_idx = torch.arange(B, device=hidden.device).unsqueeze(1).expand(B, D)
        return hidden[batch_idx, clamped]

    # ------------------------------------------------------------------
    # Forward modes
    # ------------------------------------------------------------------

    def forward_pretrain(self, x_full: torch.Tensor) -> dict:
        """Pretraining forward: returns next-bucket predictions and hidden states.

        next_bucket_pred[t] is the model's prediction for bucket t+1, computed
        using only buckets 0..t (causal). Loss should align next_bucket_pred[:, :-1, :]
        against x_full[:, 1:, :].
        """
        hidden = self.encode(x_full)
        return {
            "hidden": hidden,
            "next_bucket_pred": self.next_bucket_head(hidden),
        }

    def forward_finetune(
        self,
        x_full: torch.Tensor,
        decision_buckets: torch.Tensor | None = None,
    ) -> dict:
        """Multi-task fine-tuning forward.

        Returns:
          throughput_mu / throughput_logvar / stop_logit  : [B, D]   if decision_buckets given
          final_throughput_mu / final_throughput_logvar   : [B]
          speed_tier_logits                               : [B, num_speed_tiers]
          next_bucket_pred                                : [B, T, feature_dim]
        """
        hidden = self.encode(x_full)
        out: dict = {"hidden": hidden}

        if decision_buckets is not None:
            decision_hidden = self.gather_decisions(hidden, decision_buckets)
            out["throughput_mu"] = self.throughput_mu_head(decision_hidden).squeeze(-1)
            out["throughput_logvar"] = self.throughput_logvar_head(decision_hidden).squeeze(-1)
            out["stop_logit"] = self.stop_head(decision_hidden).squeeze(-1)

        last_hidden = hidden[:, -1, :]
        out["final_throughput_mu"] = self.throughput_mu_head(last_hidden).squeeze(-1)
        out["final_throughput_logvar"] = self.throughput_logvar_head(last_hidden).squeeze(-1)
        out["speed_tier_logits"] = self.speed_tier_head(last_hidden)
        out["next_bucket_pred"] = self.next_bucket_head(hidden)
        return out

    def forward(self, x_full: torch.Tensor, decision_buckets: torch.Tensor | None = None) -> dict:
        return self.forward_finetune(x_full, decision_buckets)

    # ------------------------------------------------------------------
    # Convenience: load encoder weights only (for pretrain -> finetune)
    # ------------------------------------------------------------------

    def load_encoder_state(self, state_dict: dict, strict: bool = False) -> tuple[list, list]:
        """Load encoder/embedding weights from a (possibly pretrain-only) checkpoint.

        Heads in the checkpoint are loaded if their shapes match; otherwise skipped.
        Returns (missing_keys, unexpected_keys) for transparency.
        """
        own_state = self.state_dict()
        loadable = {}
        for k, v in state_dict.items():
            if k in own_state and own_state[k].shape == v.shape:
                loadable[k] = v
        result = self.load_state_dict(loadable, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys)


def speed_tier_to_index(tiers: list[str]) -> torch.Tensor:
    """Map list of tier strings -> int64 tensor of indices."""
    out = torch.empty(len(tiers), dtype=torch.long)
    for i, t in enumerate(tiers):
        if t not in SPEED_TIER_TO_INDEX:
            raise ValueError(f"unknown speed_tier {t!r}; expected one of {SPEED_TIER_ORDER}")
        out[i] = SPEED_TIER_TO_INDEX[t]
    return out


def log1p_mbps(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x.clamp_min(0.0))


def expm1_mbps(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x).clamp_min(0.0)


__all__ = [
    "FMNetV3",
    "FMNetV3Config",
    "SPEED_TIER_ORDER",
    "SPEED_TIER_TO_INDEX",
    "NUM_SPEED_TIERS",
    "speed_tier_to_index",
    "log1p_mbps",
    "expm1_mbps",
]
