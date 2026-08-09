"""Regression tests for the deterministic disaggregated KV scenario model."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kimi_mooncake_kv_stream import EVIDENCE_STATE, KimiMooncakeKVStream


class TestDisaggregatedKVScenario(unittest.TestCase):
    def test_prefill_cache_reuse_model(self):
        stream = KimiMooncakeKVStream()
        result = stream.process_prefill_request(
            prompt_tokens=2_097_152,
            matched_prefix_tokens=1_966_080,
        )

        self.assertGreater(result["cache_hit_ratio"], 0.9)
        self.assertGreater(result["modeled_ttft_reduction_percent"], 0.0)
        self.assertEqual(result["evidence_state"], EVIDENCE_STATE)
        self.assertNotIn("answer", result)
        self.assertNotIn("status", result)

    def test_zero_prefix_is_valid_scenario(self):
        result = KimiMooncakeKVStream().process_prefill_request(
            prompt_tokens=100,
            matched_prefix_tokens=0,
        )
        self.assertEqual(result["cache_hit_ratio"], 0.0)
        self.assertEqual(result["uncached_tokens"], 100)

    def test_invalid_assumptions_fail_closed(self):
        with self.assertRaises(ValueError):
            KimiMooncakeKVStream(disaggregated_nodes=0)
        with self.assertRaises(ValueError):
            KimiMooncakeKVStream(cache_hit_target=float("nan"))
        with self.assertRaises(ValueError):
            KimiMooncakeKVStream().process_prefill_request(prompt_tokens=0)
        with self.assertRaises(ValueError):
            KimiMooncakeKVStream().process_prefill_request(
                prompt_tokens=10,
                matched_prefix_tokens=11,
            )


if __name__ == "__main__":
    unittest.main()
