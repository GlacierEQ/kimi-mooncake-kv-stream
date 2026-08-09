"""Regression tests for the Kimi-inspired local architecture scenario model."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kimi_k3_engine import EVIDENCE_STATE, KimiK3Config, KimiK3Engine


class TestKimiArchitectureScenario(unittest.TestCase):
    def setUp(self):
        self.engine = KimiK3Engine()

    def test_fixed_state_storage_scenario(self):
        result = self.engine.compute_kda_memory_savings(sequence_length=1_000_000)
        self.assertGreater(result["modeled_storage_reduction_percent"], 0.0)
        self.assertEqual(result["evidence_state"], EVIDENCE_STATE)

    def test_variance_ratio_scenario(self):
        result = self.engine.compute_attnres_scaling_gain(depth_layers=64)
        self.assertEqual(result["modeled_variance_ratio"], 8.0)
        self.assertEqual(result["evidence_state"], EVIDENCE_STATE)

    def test_expert_imbalance_scenario(self):
        result = self.engine.run_moonep_expert_dispatch(active_tokens=32_768)
        self.assertGreater(result["modeled_imbalance_reduction_percent"], 20.0)
        self.assertEqual(result["evidence_state"], EVIDENCE_STATE)

    def test_invalid_scenario_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            KimiK3Config(context_window=0)
        with self.assertRaises(ValueError):
            KimiK3Config(baseline_imbalance_ratio=float("inf"))
        with self.assertRaises(ValueError):
            self.engine.compute_kda_memory_savings(sequence_length=0)
        with self.assertRaises(ValueError):
            self.engine.compute_attnres_scaling_gain(depth_layers=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.engine.run_moonep_expert_dispatch(active_tokens=0)


if __name__ == "__main__":
    unittest.main()
