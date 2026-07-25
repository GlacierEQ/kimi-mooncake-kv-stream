# Kimi Mooncake KV Stream

> **Production Solution for Kimi Moonshot AI 2M+ Context Prefill Latency**

## Overview
Disaggregated prefill/decoding KV-cache streaming and prefix cache-hit predictor for Moonshot AI Kimi K1.5.

## Verification
```bash
PYTHONPATH=src python3 tests/test_kimi.py
python3 mastermind_sidecar.py
```
