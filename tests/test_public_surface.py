from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_public_surface_uses_real_paths_and_evidence_tokens() -> None:
    text = README.read_text(encoding="utf-8")
    for path in (
        "src/kimi_mooncake_kv_stream.py",
        "src/kimi_k3_engine.py",
        "src/kv_cache_ring.rs",
        "src/mooncake_scheduler.go",
    ):
        assert path in text
    assert "MODELED_DISAGGREGATED_KV_SCENARIO_NOT_KIMI_RUNTIME" in text
    assert "MODELED_KIMI_ARCHITECTURE_SCENARIO_NOT_MODEL_EXECUTION" in text


def test_public_surface_excludes_unverified_runtime_claims() -> None:
    text = README.read_text(encoding="utf-8").casefold()
    forbidden = (
        "handling rdma-style zero-copy tensor transfers",
        "supporting 2.8t linear attention streams",
        "automatic eviction of low-entropy kv cache entries",
        "mcp tool: `mooncake_stats()`",
        "fully connected to apex highway mesh",
    )
    assert all(marker not in text for marker in forbidden)


def test_public_surface_declares_non_affiliation_and_transport_boundary() -> None:
    text = README.read_text(encoding="utf-8").casefold()
    assert "not affiliated with, endorsed by, or operated by moonshot ai or kimi" in text
    assert "does not claim proprietary model access" in text
    assert "does not implement or prove rdma or zero-copy distributed transport" in text
