/// Local KV-cache ring-buffer reference.
///
/// This module exercises atomic index bookkeeping in process. It does not perform
/// RDMA, zero-copy network transfer, or distributed prefill/decode coordination.
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

impl Default for KVCacheRing {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCacheRing {
    pub fn new() -> Self {
        let mut buffer = Vec::with_capacity(RING_CAPACITY);
        for _ in 0..RING_CAPACITY {
            buffer.push(KVCacheEntry {
                layer_id: 0,
                head_id: 0,
                seq_pos: 0,
                key_norm: 0.0,
                value_norm: 0.0,
                timestamp_ns: 0,
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

    /// Push an entry into the local ring, evicting the oldest slot when full.
    pub fn push(&mut self, entry: KVCacheEntry) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        let next_tail = (tail + 1) % self.capacity as u64;

        if next_tail == head {
            self.head
                .store((head + 1) % self.capacity as u64, Ordering::Release);
            self.total_evictions.fetch_add(1, Ordering::Relaxed);
        }

        let idx = tail as usize % self.capacity;
        self.buffer[idx] = entry;
        self.tail.store(next_tail, Ordering::Release);
        true
    }

    /// Remove and return the oldest logical entry.
    pub fn pop(&self) -> Option<&KVCacheEntry> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let idx = head as usize % self.capacity;
        self.head
            .store((head + 1) % self.capacity as u64, Ordering::Release);
        Some(&self.buffer[idx])
    }

    /// Compute local ring occupancy ratio (0.0 = empty, approaching 1.0 = full).
    pub fn pressure(&self) -> f64 {
        self.len() as f64 / self.capacity as f64
    }

    pub fn eviction_count(&self) -> u64 {
        self.total_evictions.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if tail >= head {
            (tail - head) as usize
        } else {
            self.capacity - head as usize + tail as usize
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seq_pos: u64) -> KVCacheEntry {
        KVCacheEntry {
            layer_id: 0,
            head_id: 0,
            seq_pos,
            key_norm: 1.0,
            value_norm: 0.5,
            timestamp_ns: seq_pos,
        }
    }

    #[test]
    fn test_push_pop_cycle_advances_head() {
        let mut ring = KVCacheRing::new();
        assert!(ring.is_empty());
        ring.push(entry(42));
        assert_eq!(ring.len(), 1);
        let popped = ring.pop().unwrap();
        assert_eq!(popped.seq_pos, 42);
        assert!(ring.is_empty());
        assert!(ring.pop().is_none());
    }

    #[test]
    fn test_fifo_order() {
        let mut ring = KVCacheRing::new();
        ring.push(entry(1));
        ring.push(entry(2));
        assert_eq!(ring.pop().unwrap().seq_pos, 1);
        assert_eq!(ring.pop().unwrap().seq_pos, 2);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_pressure_calculation() {
        let mut ring = KVCacheRing::new();
        assert_eq!(ring.pressure(), 0.0);
        for i in 0..1000 {
            ring.push(entry(i));
        }
        assert!(ring.pressure() > 0.0);
        assert!(ring.pressure() < 1.0);
    }
}
