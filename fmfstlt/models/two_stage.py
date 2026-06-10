#!/usr/bin/env python3
"""FMNet two-stage model with TurboTest's Stage 1 to Stage 2 flow.

Architecture:
  raw trace [100,13]
    -> shared causal bucket-level encoder (reused from FMNet v3)
    -> per-decision throughput head (Stage 1)  -> stage1_mu / stage1_logvar
    -> Stage 2 policy module (small causal Transformer over 20 decision tokens)
       reads ONLY Stage 1 outputs + minimal metadata
       (optionally also h_decision when --include-h-decision is set)
    -> per-decision stop logit

Strict variant (h_decision = off):
    Stage 2 input per decision = 4 dims:
      [stage1_mu, stage1_logvar, elapsed_norm, observed_norm]
    This is the strict TurboTest analogue: Stage 2 sees only Stage 1's outputs
    plus decision metadata, no encoder representations directly.

Richer variant (h_decision = on):
    Stage 2 input per decision = 4 + d_model = 260 dims:
      strict features concatenated with the encoder's per-decision hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fmfstlt.models.fmnet_v3 import (
    SPEED_TIER_TO_INDEX,
    FMNetV3,
    FMNetV3Config,
    expm1_mbps,
    log1p_mbps,
)


@dataclass
class Stage2PolicyConfig:
    max_decisions: int = 20
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.15

    def to_dict(self) -> dict:
        return {
            "max_decisions": self.max_decisions,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Stage2PolicyConfig:
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


class Stage2PolicyModule(nn.Module):
    """Small causal Transformer over 20 per-decision feature vectors."""

    def __init__(self, *, input_dim: int, config: Stage2PolicyConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        d = config.d_model

        self.input_projection = nn.Linear(input_dim, d)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_decisions, d))
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(d)

        self.stop_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, 1),
        )

    def forward(self, decision_features: torch.Tensor) -> torch.Tensor:
        """decision_features: [B, T, input_dim] -> stop_logit [B, T]."""
        B, T, _ = decision_features.shape
        h = self.input_projection(decision_features) + self.position_embedding[:, :T, :]
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=decision_features.device),
            diagonal=1,
        )
        h = self.encoder(h, mask=causal_mask)
        h = self.norm(h)
        return self.stop_head(h).squeeze(-1)


@dataclass
class FMNetTwoStageConfig:
    encoder: FMNetV3Config
    policy: Stage2PolicyConfig
    include_h_decision: bool = False

    def to_dict(self) -> dict:
        return {
            "encoder": self.encoder.to_dict(),
            "policy": self.policy.to_dict(),
            "include_h_decision": self.include_h_decision,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FMNetTwoStageConfig:
        return cls(
            encoder=FMNetV3Config.from_dict(d["encoder"]),
            policy=Stage2PolicyConfig.from_dict(d["policy"]),
            include_h_decision=bool(d.get("include_h_decision", False)),
        )


class FMNetTwoStage(nn.Module):
    """Single foundation model: shared encoder + Stage 1 head + Stage 2 policy module.

    All parameters are part of one model (one set of weights). No external XGBoost.
    """

    def __init__(self, config: FMNetTwoStageConfig) -> None:
        super().__init__()
        self.config = config
        self.include_h_decision = config.include_h_decision
        self.encoder_model = FMNetV3(config.encoder)

        stage2_input_dim = 4 + (config.encoder.d_model if config.include_h_decision else 0)
        self.policy = Stage2PolicyModule(
            input_dim=stage2_input_dim,
            config=config.policy,
        )

    # ------------------------------------------------------------------
    # Phase 1: encoder + Stage 1 head only
    # ------------------------------------------------------------------

    def forward_stage1(
        self,
        x_full: torch.Tensor,
        decision_buckets: torch.Tensor,
    ) -> dict:
        hidden = self.encoder_model.encode(x_full)
        decision_hidden = self.encoder_model.gather_decisions(hidden, decision_buckets)
        stage1_mu = self.encoder_model.throughput_mu_head(decision_hidden).squeeze(-1)
        stage1_logvar = self.encoder_model.throughput_logvar_head(decision_hidden).squeeze(-1)
        last_hidden = hidden[:, -1, :]
        final_mu = self.encoder_model.throughput_mu_head(last_hidden).squeeze(-1)
        final_logvar = self.encoder_model.throughput_logvar_head(last_hidden).squeeze(-1)
        return {
            "stage1_mu": stage1_mu,  # [B, D]
            "stage1_logvar": stage1_logvar,  # [B, D]
            "final_throughput_mu": final_mu,  # [B]
            "final_throughput_logvar": final_logvar,  # [B]
            "decision_hidden": decision_hidden,  # [B, D, encoder.d_model]
            "hidden": hidden,  # [B, T, encoder.d_model]
        }

    # ------------------------------------------------------------------
    # Phase 2 / 3: encoder + Stage 1 + Stage 2 policy module
    # ------------------------------------------------------------------

    def forward_full(
        self,
        x_full: torch.Tensor,
        decision_buckets: torch.Tensor,
        decision_elapsed_ms: torch.Tensor,
        observed_buckets_seen: torch.Tensor,
        detach_stage1: bool = True,
    ) -> dict:
        s1 = self.forward_stage1(x_full, decision_buckets)

        stage1_mu = s1["stage1_mu"]
        stage1_logvar = s1["stage1_logvar"]
        decision_hidden = s1["decision_hidden"]

        if detach_stage1:
            stage1_mu_in = stage1_mu.detach()
            stage1_logvar_in = stage1_logvar.detach()
            decision_hidden_in = decision_hidden.detach()
        else:
            stage1_mu_in = stage1_mu
            stage1_logvar_in = stage1_logvar
            decision_hidden_in = decision_hidden

        elapsed_norm = decision_elapsed_ms.float() / 10000.0  # 10s = 1.0
        observed_norm = observed_buckets_seen.float() / 100.0  # 100 buckets = 1.0

        features = [
            stage1_mu_in.unsqueeze(-1),
            stage1_logvar_in.unsqueeze(-1),
            elapsed_norm.unsqueeze(-1),
            observed_norm.unsqueeze(-1),
        ]
        if self.include_h_decision:
            features.append(decision_hidden_in)

        decision_features = torch.cat(features, dim=-1)  # [B, D, input_dim]
        stop_logit = self.policy(decision_features)  # [B, D]

        out = dict(s1)
        out["stop_logit"] = stop_logit
        out["decision_features"] = decision_features
        return out

    # ------------------------------------------------------------------
    # Convenience: load encoder weights from FMNet v3 pretrain ckpt
    # ------------------------------------------------------------------

    def load_encoder_state(self, state_dict: dict) -> tuple[list, list]:
        return self.encoder_model.load_encoder_state(state_dict, strict=False)

    def freeze_encoder(self) -> None:
        for p in self.encoder_model.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder_model.parameters():
            p.requires_grad = True


__all__ = [
    "FMNetTwoStage",
    "FMNetTwoStageConfig",
    "Stage2PolicyModule",
    "Stage2PolicyConfig",
    "FMNetV3Config",
    "log1p_mbps",
    "expm1_mbps",
    "SPEED_TIER_TO_INDEX",
]
