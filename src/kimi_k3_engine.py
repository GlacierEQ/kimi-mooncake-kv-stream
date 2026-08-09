#!/usr/bin/env python3
"""Local architecture arithmetic inspired by public long-context/MoE patterns.

The defaults are scenario assumptions. This module does not execute a Kimi model,
validate KDA/AttnRes/MoonEP fidelity, or measure model quality or throughput.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EVIDENCE_STATE = "MODELED_KIMI_ARCHITECTURE_SCENARIO_NOT_MODEL_EXECUTION"


@dataclass(frozen=True)
class KimiK3Config:
    context_window: int = 1_048_576
    layers: int = 64
    heads: int = 32
    head_dim: int = 128
    fixed_state_tokens: int = 16_384
    num_experts: int = 256
    redundant_experts_per_rank: int = 4
    baseline_imbalance_ratio: float = 1.45
    modeled_redundant_imbalance_ratio: float = 1.01

    def __post_init__(self) -> None:
        for name in (
            "context_window",
            "layers",
            "heads",
            "head_dim",
            "fixed_state_tokens",
            "num_experts",
            "redundant_experts_per_rank",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("baseline_imbalance_ratio", "modeled_redundant_imbalance_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")


class KimiK3Engine:
    """Evaluate explicit long-context and expert-routing scenario assumptions."""

    def __init__(self, config: KimiK3Config | None = None):
        self.config = config or KimiK3Config()

    def compute_kda_memory_savings(self, sequence_length: int) -> dict:
        """Compare a linear FP16 KV baseline with a fixed-state scenario."""

        if type(sequence_length) is not int or sequence_length < 1:
            raise ValueError("sequence_length must be an integer >= 1")
        seq_len = min(sequence_length, self.config.context_window)
        baseline_kv_bytes = (
            2
            * self.config.layers
            * self.config.heads
            * self.config.head_dim
            * seq_len
        )
        fixed_state_bytes = (
            2
            * self.config.layers
            * self.config.heads
            * self.config.head_dim
            * self.config.fixed_state_tokens
        )
        reduction = max(
            0.0,
            (1.0 - fixed_state_bytes / baseline_kv_bytes) * 100.0,
        )
        return {
            "sequence_length": seq_len,
            "modeled_baseline_kv_mb": round(baseline_kv_bytes / (1024 * 1024), 2),
            "modeled_fixed_state_mb": round(fixed_state_bytes / (1024 * 1024), 2),
            "modeled_storage_reduction_percent": round(reduction, 2),
            "fixed_state_tokens": self.config.fixed_state_tokens,
            "evidence_state": EVIDENCE_STATE,
        }

    def compute_attnres_scaling_gain(self, depth_layers: int = 64) -> dict:
        """Return a simple sqrt(depth) variance-ratio scenario, not a training metric."""

        if type(depth_layers) is not int or depth_layers < 1:
            raise ValueError("depth_layers must be an integer >= 1")
        baseline_norm = math.sqrt(depth_layers)
        modeled_normalized_variance = 1.0
        return {
            "depth_layers": depth_layers,
            "modeled_baseline_variance_norm": round(baseline_norm, 4),
            "modeled_normalized_variance": modeled_normalized_variance,
            "modeled_variance_ratio": round(
                baseline_norm / modeled_normalized_variance, 4
            ),
            "evidence_state": EVIDENCE_STATE,
        }

    def run_moonep_expert_dispatch(self, active_tokens: int = 32_768) -> dict:
        """Compare configured expert-imbalance assumptions; no live dispatch occurs."""

        if type(active_tokens) is not int or active_tokens < 1:
            raise ValueError("active_tokens must be an integer >= 1")
        baseline = self.config.baseline_imbalance_ratio
        modeled = self.config.modeled_redundant_imbalance_ratio
        reduction = max(0.0, (baseline - modeled) / baseline * 100.0)
        return {
            "active_tokens": active_tokens,
            "baseline_imbalance_assumption": baseline,
            "modeled_redundant_imbalance_assumption": modeled,
            "modeled_imbalance_reduction_percent": round(reduction, 2),
            "evidence_state": EVIDENCE_STATE,
        }
