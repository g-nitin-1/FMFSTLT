from __future__ import annotations

import torch

from fmfstlt.models import (
    FMNetTwoStage,
    FMNetTwoStageConfig,
    FMNetV3,
    FMNetV3Config,
    Stage2PolicyConfig,
)


def small_encoder_config() -> FMNetV3Config:
    return FMNetV3Config(
        feature_dim=3,
        max_buckets=10,
        d_model=16,
        num_heads=4,
        num_layers=2,
        ff_dim=32,
        dropout=0.0,
        head_dropout=0.0,
        num_speed_tiers=5,
    )


def test_fmnet_v3_is_causal() -> None:
    torch.manual_seed(7)
    model = FMNetV3(small_encoder_config()).eval()

    original = torch.randn(2, 10, 3)
    changed_future = original.clone()
    changed_future[:, 5:, :] += 100.0

    with torch.no_grad():
        original_hidden = model.encode(original)
        changed_hidden = model.encode(changed_future)

    torch.testing.assert_close(original_hidden[:, :5], changed_hidden[:, :5])


def test_two_stage_output_shapes() -> None:
    config = FMNetTwoStageConfig(
        encoder=small_encoder_config(),
        policy=Stage2PolicyConfig(
            max_decisions=2,
            d_model=8,
            num_heads=2,
            num_layers=1,
            ff_dim=16,
            dropout=0.0,
        ),
        include_h_decision=True,
    )
    model = FMNetTwoStage(config).eval()

    x_full = torch.randn(3, 10, 3)
    decision_buckets = torch.tensor([[4, 9], [4, 9], [4, 9]])
    elapsed_ms = torch.tensor([[500, 1000], [500, 1000], [500, 1000]])
    observed = torch.tensor([[5, 10], [5, 10], [5, 10]])

    with torch.no_grad():
        outputs = model.forward_full(
            x_full,
            decision_buckets,
            elapsed_ms,
            observed,
            detach_stage1=True,
        )

    assert outputs["stage1_mu"].shape == (3, 2)
    assert outputs["stage1_logvar"].shape == (3, 2)
    assert outputs["stop_logit"].shape == (3, 2)
    assert outputs["decision_features"].shape == (3, 2, 20)


def test_detached_stage1_blocks_policy_gradients() -> None:
    config = FMNetTwoStageConfig(
        encoder=small_encoder_config(),
        policy=Stage2PolicyConfig(
            max_decisions=2,
            d_model=8,
            num_heads=2,
            num_layers=1,
            ff_dim=16,
            dropout=0.0,
        ),
        include_h_decision=True,
    )
    model = FMNetTwoStage(config)
    outputs = model.forward_full(
        torch.randn(2, 10, 3),
        torch.tensor([[4, 9], [4, 9]]),
        torch.tensor([[500, 1000], [500, 1000]]),
        torch.tensor([[5, 10], [5, 10]]),
        detach_stage1=True,
    )
    outputs["stop_logit"].sum().backward()

    assert any(parameter.grad is not None for parameter in model.policy.parameters())
    assert all(parameter.grad is None for parameter in model.encoder_model.parameters())
