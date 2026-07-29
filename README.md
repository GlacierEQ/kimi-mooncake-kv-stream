# Kimi Mooncake KV Stream — Disaggregated KV Cache Engine 🌙

> **Rust lock-free ring buffer & Go disaggregated prefill/decode scheduler for Kimi K3 Mooncake architecture.**

[![Rust](https://img.shields.io/badge/Rust-Lock--Free-orange)]()
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Domain](https://img.shields.io/badge/Domain-Disaggregated%20LLM-blue)]()

---

## 🎯 For Recruiters & Hiring Managers

This repository implements **Kimi's Mooncake Disaggregated Prefill/Decode Engine** — separating compute-heavy prefill nodes from memory-heavy decode nodes for massive LLM inference throughput. It demonstrates:

- **Rust lock-free KV ring buffer** handling RDMA-style zero-copy tensor transfers
- **Go priority-aware scheduler** routing requests based on node memory pressure and KV cache size
- **Python Kimi K3 engine interface** supporting 2.8T linear attention streams
- **Cache pressure mitigation** with automatic eviction of low-entropy KV cache entries

**Why this matters**: Disaggregating prefill and decode phases is the modern standard for scaling long-context LLM serving architectures.

---

## 🔬 For Engineers & Technical Reviewers

### Core Components

| Component | Language | Purpose |
|---|---|---|
| `src/kv_cache_ring.rs` | Rust | Atomic lock-free ring buffer for zero-copy KV streaming |
| `src/mooncake_scheduler.go` | Go | Priority-aware disaggregated node scheduler |
| `src/kimi_k3_engine.py` | Python | Kimi K3 linear attention model harness |
| `tests/` | Python | End-to-end Mooncake streaming simulation |

---

## 🤖 ML/AI & Programmatic Mesh Integration

- **MCP Tool**: `mooncake_stats()` — inspect prefill/decode node utilization
- **Mastermind Sidecar**: Fully connected to APEX Highway mesh
- **SHA-256 Integrity**: Tracked in `.integrity/file_hashes.json`

---

## ⚡ Quick Start

```bash
python3 src/kimi_k3_engine.py
python3 tests/test_kimi_k3.py
```
