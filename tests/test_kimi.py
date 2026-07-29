"""Test suite for Kimi Mooncake KV Stream solution."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kimi_mooncake_kv_stream import KimiMooncakeKVStream

class TestKimiMooncakeKVStream(unittest.TestCase):

    def test_mooncake_prefill(self):
        stream = KimiMooncakeKVStream()
        res = stream.process_prefill_request(prompt_tokens=2_097_152, matched_prefix_tokens=1_966_080)
        
        self.assertEqual(res["status"], "MOONCAKE_STREAM_OPTIMAL")
        self.assertTrue(res["latency_reduction_percent"] > 90.0)

if __name__ == "__main__":
    unittest.main()
