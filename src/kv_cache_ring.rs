/// Kimi Mooncake KV Cache Ring Buffer — Zero-Copy Disaggregated Cache
/// Implements a lock-free ring buffer for KV tensor streaming across
/// disaggregated prefill/decode nodes with RDMA-style semantics.

use std::sync::atomic::{AtomicU64, Ordering};

const RING_CAPACITY: usize = 65536;

#[repr(C)]
pub struct KVCacheEntry {
    pub layer_id: u32,
    pub head_id: u32,
    pub seq_pos: u64,
    pub key_norm: f32,
    pub value_norm: f32,
    pub timestamp_ns: u64,
}

pub struct KVCacheRing {
    buffer: Vec<KVCacheEntry>,
    head: AtomicU64,
    tail: AtomicU64,
    capacity: usize,
    total_evictions: AtomicU64,
}

impl KVCacheRing {
    pub fn new() -> Self {
        let mut buffer = Vec::with_capacity(RING_CAPACITY);
        for _ in 0..RING_CAPACITY {
            buffer.push(KVCacheEntry {
                layer_id: 0, head_id: 0, seq_pos: 0,
                key_norm: 0.0, value_norm: 0.0, timestamp_ns: 0,
            });
        }
        KVCacheRing {
            buffer,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            capacity: RING_CAPACITY,
            total_evictions: AtomicU64::new(0),
        }
    }

    /// Push a KV entry into the ring. Returns true if successful, evicts oldest if full.
    pub fn push(&mut self, entry: KVCacheEntry) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        let next_tail = (tail + 1) % self.capacity as u64;

        if next_tail == head {
            // Ring full — evict oldest entry
            self.head.store((head + 1) % self.capacity as u64, Ordering::Release);
            self.total_evictions.fetch_add(1, Ordering::Relaxed);
        }

        let idx = tail as usize % self.capacity;
        self.buffer[idx] = entry;
        self.tail.store(next_tail, Ordering::Release);
        true
    }

    /// Pop oldest KV entry from the ring
    pub fn pop(&self) -> Option<&KVCacheEntry> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        Some(&self.buffer[head as usize % self.capacity])
    }

    /// Compute cache pressure ratio (0.0 = empty, 1.0 = full)
    pub fn pressure(&self) -> f64 {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let used = if tail >= head { tail - head } else { self.capacity as u64 - head + tail };
        used as f64 / self.capacity as f64
    }

    pub fn eviction_count(&self) -> u64 {
        self.total_evictions.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if tail >= head { (tail - head) as usize } else { self.capacity - head as usize + tail as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop_cycle() {
        let mut ring = KVCacheRing::new();
        assert!(ring.is_empty());
        ring.push(KVCacheEntry {
            layer_id: 0, head_id: 0, seq_pos: 42,
            key_norm: 1.0, value_norm: 0.5, timestamp_ns: 100,
        });
        assert_eq!(ring.len(), 1);
        let entry = ring.pop().unwrap();
        assert_eq!(entry.seq_pos, 42);
    }

    #[test]
    fn test_pressure_calculation() {
        let mut ring = KVCacheRing::new();
        assert_eq!(ring.pressure(), 0.0);
        for i in 0..1000 {
            ring.push(KVCacheEntry {
                layer_id: 0, head_id: 0, seq_pos: i,
                key_norm: 1.0, value_norm: 1.0, timestamp_ns: i,
            });
        }
        assert!(ring.pressure() > 0.0);
    }
}
