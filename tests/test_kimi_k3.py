"""Test suite for Kimi K3 Engine (KDA, AttnRes, MoonEP)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kimi_k3_engine import KimiK3Engine, KimiK3Config

class TestKimiK3Engine(unittest.TestCase):

    def setUp(self):
        self.engine = KimiK3Engine()

    def test_kda_memory_savings(self):
        res = self.engine.compute_kda_memory_savings(sequence_length=1_000_000)
        self.assertEqual(res["status"], "KDA_LINEAR_ATTENTION_OPTIMAL")
        self.assertGreater(res["memory_reduction_percent"], 0.0)

    def test_attnres_scaling(self):
        res = self.engine.compute_attnres_scaling_gain(depth_layers=64)
        self.assertEqual(res["status"], "ATTNRES_STABILITY_PASS")
        self.assertEqual(res["gradient_stability_gain"], 8.0)

    def test_moonep_dispatch(self):
        res = self.engine.run_moonep_expert_dispatch(active_tokens=32768)
        self.assertEqual(res["status"], "MOONEP_BALANCED_DISPATCH_OK")
        self.assertGreater(res["throughput_boost_percent"], 20.0)

if __name__ == "__main__":
    unittest.main()
