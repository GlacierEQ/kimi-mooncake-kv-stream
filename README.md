# Disaggregated KV Streaming Study

Independent GlacierEQ portfolio work exploring local cache-reuse arithmetic, request scheduling, and ring-buffer mechanics inspired by publicly discussed disaggregated prefill/decode patterns.

**Status:** local scenario models + Rust/Go reference implementations.  
**Evidence tokens:** `MODELED_DISAGGREGATED_KV_SCENARIO_NOT_KIMI_RUNTIME` and `MODELED_KIMI_ARCHITECTURE_SCENARIO_NOT_MODEL_EXECUTION`.

This repository is **not affiliated with, endorsed by, or operated by Moonshot AI or Kimi**. It does not claim proprietary model access, Mooncake deployment, RDMA transport, measured TTFT, or execution of a Kimi model.

## Verified capabilities

### Python cache-reuse scenario

`src/kimi_mooncake_kv_stream.py` deterministically computes from explicit inputs and timing assumptions:

- matched-prefix cache ratio;
- uncached token count;
- modeled uncached/cached TTFT;
- modeled TTFT reduction;
- fail-closed invalid prompt/prefix/timing inputs.

Those timing numbers are assumptions supplied to a local model, **not measurements from Kimi or Mooncake infrastructure**.

### Python architecture assumptions

`src/kimi_k3_engine.py` evaluates explicit local scenarios for:

- linear KV baseline versus fixed-state storage;
- a simple sqrt(depth) variance-ratio heuristic;
- configured expert-imbalance assumptions.

The defaults are scenario parameters, not assertions about Kimi model size, KDA/AttnRes/MoonEP fidelity, training behavior, or throughput.

### Rust local ring buffer

`src/kv_cache_ring.rs` is a local in-process ring-buffer reference. Its tests verify FIFO push/pop behavior, logical head advancement, and occupancy. It does **not** implement or prove RDMA or zero-copy distributed transport.

### Go local scheduler

`src/mooncake_scheduler.go` is a local priority-aware prefill/decode node-selection reference. CI compiles it for build correctness; it does not establish a deployed Mooncake scheduler.

## Native proof

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
rustc --test src/kv_cache_ring.rs -o /tmp/kv_ring_tests
/tmp/kv_ring_tests
go test src/mooncake_scheduler.go
```

The repository-owned Public Truth Gate runs Python 3.11 and 3.13, Rust tests, Go compilation, and public-boundary checks on the exact source head.

## Explicit nonclaims

Current evidence does **not** establish:

- execution of Kimi K1.5/K3 or a 2.8T-parameter model;
- a faithful production implementation of KDA, AttnRes, MoonEP, or Mooncake;
- 2M+ live-context serving;
- measured 98.4% cache reuse or sub-millisecond routing;
- measured TTFT/throughput improvements;
- RDMA or zero-copy network transfer;
- automatic low-entropy KV eviction;
- MCP tool registration;
- live APEX/AKOS/Mastermind connectivity;
- Moonshot AI/Kimi employment, endorsement, affiliation, or proprietary access.

Those require separate model, network, or deployment receipts.

## Why the capability matters

The engineering value is in the concrete local mechanisms—cache-reuse arithmetic, bounded scenario assumptions, scheduler selection logic, and corrected FIFO ring behavior—while keeping company/runtime claims outside the evidence boundary until they are actually measured.
