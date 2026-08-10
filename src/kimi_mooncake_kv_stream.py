"""Deterministic disaggregated KV-cache scenario model.

This module does not execute Kimi/Mooncake infrastructure, contact Moonshot AI,
measure TTFT, or perform network/RDMA transfer. It applies explicit local timing
assumptions to prompt/prefix counts so cache-reuse arithmetic remains testable.
"""

from __future__ import annotations

import math
from typing import Any

EVIDENCE_STATE = "MODELED_DISAGGREGATED_KV_SCENARIO_NOT_KIMI_RUNTIME"


class KimiMooncakeKVStream:
    """Model cache-reuse and TTFT scenarios from explicit assumptions."""

    def __init__(
        self,
        disaggregated_nodes: int = 128,
        cache_hit_target: float = 0.95,
        uncached_ms_per_100k_tokens: float = 12.0,
        cached_fixed_overhead_ms: float = 1.8,
    ) -> None:
        if type(disaggregated_nodes) is not int or disaggregated_nodes < 1:
            raise ValueError("disaggregated_nodes must be an integer >= 1")
        if not math.isfinite(cache_hit_target) or not 0.0 <= cache_hit_target <= 1.0:
            raise ValueError("cache_hit_target must be finite and within [0, 1]")
        if (
            not math.isfinite(uncached_ms_per_100k_tokens)
            or uncached_ms_per_100k_tokens <= 0
        ):
            raise ValueError("uncached_ms_per_100k_tokens must be finite and > 0")
        if not math.isfinite(cached_fixed_overhead_ms) or cached_fixed_overhead_ms < 0:
            raise ValueError("cached_fixed_overhead_ms must be finite and >= 0")
        self.disaggregated_nodes = disaggregated_nodes
        self.cache_hit_target = cache_hit_target
        self.uncached_ms_per_100k_tokens = uncached_ms_per_100k_tokens
        self.cached_fixed_overhead_ms = cached_fixed_overhead_ms

    def process_prefill_request(
        self, prompt_tokens: int = 2_097_152, matched_prefix_tokens: int = 1_966_080
    ) -> dict[str, Any]:
        """Return modeled cache reuse and TTFT from configured assumptions."""

        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError("prompt_tokens must be an integer >= 1")
        if type(matched_prefix_tokens) is not int or matched_prefix_tokens < 0:
            raise ValueError("matched_prefix_tokens must be an integer >= 0")
        if matched_prefix_tokens > prompt_tokens:
            raise ValueError("matched_prefix_tokens must not exceed prompt_tokens")

        cache_hit_ratio = matched_prefix_tokens / prompt_tokens
        uncached_tokens = prompt_tokens - matched_prefix_tokens
        ttft_uncached_ms = (
            prompt_tokens / 100_000.0 * self.uncached_ms_per_100k_tokens
        )
        ttft_cached_ms = (
            uncached_tokens / 100_000.0 * self.uncached_ms_per_100k_tokens
            + self.cached_fixed_overhead_ms
        )
        modeled_reduction = max(
            0.0,
            (1.0 - ttft_cached_ms / ttft_uncached_ms) * 100.0,
        )

        return {
            "prompt_tokens": prompt_tokens,
            "cached_prefix_tokens": matched_prefix_tokens,
            "uncached_tokens": uncached_tokens,
            "cache_hit_ratio": round(cache_hit_ratio, 4),
<<<<<<< HEAD
            "cache_hit_target": self.cache_hit_target,
            "modeled_ttft_uncached_ms": round(ttft_uncached_ms, 4),
            "modeled_ttft_cached_ms": round(ttft_cached_ms, 4),
            "modeled_ttft_reduction_percent": round(modeled_reduction, 2),
            "assumed_uncached_ms_per_100k_tokens": self.uncached_ms_per_100k_tokens,
            "assumed_cached_fixed_overhead_ms": self.cached_fixed_overhead_ms,
            "configured_nodes": self.disaggregated_nodes,
            "evidence_state": EVIDENCE_STATE,
        }
=======
            "ttft_uncached_ms": round(ttft_uncached_ms, 2),
            "ttft_optimized_ms": round(ttft_cached_ms, 2),
            "latency_reduction_percent": round((1.0 - (ttft_cached_ms / ttft_uncached_ms)) * 100.0, 2),
            "status": "MOONCAKE_STREAM_OPTIMAL"
            }
>>>>>>> cbce7ad (chore: Hyper Excellence Activation & structural matrix alignment)
