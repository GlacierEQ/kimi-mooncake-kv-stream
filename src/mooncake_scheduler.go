// Package mooncake implements a disaggregated prefill/decode scheduler
// for Kimi's Mooncake architecture with priority-aware request routing.
package mooncake

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// RequestPriority determines scheduling order in the prefill queue
type RequestPriority int

const (
	PriorityRealtime RequestPriority = iota
	PriorityInteractive
	PriorityBatch
	PriorityColdStart
)

// InferenceRequest represents a single LLM inference request
type InferenceRequest struct {
	ID            string
	Priority      RequestPriority
	SequenceLen   int
	MaxNewTokens  int
	PrefillNode   string
	DecodeNode    string
	KVCacheSizeMB float64
	CreatedAt     time.Time
	StartedAt     time.Time
}

// NodeMetrics tracks health and capacity for a single compute node
type NodeMetrics struct {
	NodeID          string
	MemoryUsedGB    float64
	MemoryTotalGB   float64
	ActiveRequests  int
	AvgLatencyMs    float64
	KVCachePressure float64
}

// MooncakeScheduler implements the disaggregated prefill/decode scheduler
type MooncakeScheduler struct {
	mu             sync.RWMutex
	prefillNodes   map[string]*NodeMetrics
	decodeNodes    map[string]*NodeMetrics
	pendingQueue   []*InferenceRequest
	activeRequests map[string]*InferenceRequest
	totalScheduled uint64
	totalEvicted   uint64
}

// NewMooncakeScheduler creates a scheduler with prefill and decode node pools
func NewMooncakeScheduler() *MooncakeScheduler {
	return &MooncakeScheduler{
		prefillNodes:   make(map[string]*NodeMetrics),
		decodeNodes:    make(map[string]*NodeMetrics),
		pendingQueue:   make([]*InferenceRequest, 0, 1024),
		activeRequests: make(map[string]*InferenceRequest),
	}
}

// RegisterPrefillNode adds a prefill-capable node to the scheduler pool
func (s *MooncakeScheduler) RegisterPrefillNode(nodeID string, memoryGB float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prefillNodes[nodeID] = &NodeMetrics{
		NodeID:        nodeID,
		MemoryTotalGB: memoryGB,
	}
}

// RegisterDecodeNode adds a decode-capable node to the scheduler pool
func (s *MooncakeScheduler) RegisterDecodeNode(nodeID string, memoryGB float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decodeNodes[nodeID] = &NodeMetrics{
		NodeID:        nodeID,
		MemoryTotalGB: memoryGB,
	}
}

// Submit enqueues a new inference request with priority scheduling
func (s *MooncakeScheduler) Submit(req *InferenceRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	req.CreatedAt = time.Now()
	s.pendingQueue = append(s.pendingQueue, req)
	// Sort by priority (lower = higher priority)
	sort.Slice(s.pendingQueue, func(i, j int) bool {
		return s.pendingQueue[i].Priority < s.pendingQueue[j].Priority
	})
	return nil
}

// selectBestNode picks the node with lowest memory pressure
func (s *MooncakeScheduler) selectBestNode(nodes map[string]*NodeMetrics) *NodeMetrics {
	var best *NodeMetrics
	bestScore := math.MaxFloat64
	for _, n := range nodes {
		pressure := n.MemoryUsedGB / math.Max(n.MemoryTotalGB, 1.0)
		score := pressure*0.6 + float64(n.ActiveRequests)*0.3 + n.KVCachePressure*0.1
		if score < bestScore {
			bestScore = score
			best = n
		}
	}
	return best
}

// ScheduleNext assigns the highest-priority pending request to optimal nodes
func (s *MooncakeScheduler) ScheduleNext() (*InferenceRequest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.pendingQueue) == 0 {
		return nil, fmt.Errorf("no pending requests")
	}
	req := s.pendingQueue[0]
	s.pendingQueue = s.pendingQueue[1:]

	prefill := s.selectBestNode(s.prefillNodes)
	decode := s.selectBestNode(s.decodeNodes)
	if prefill == nil || decode == nil {
		return nil, fmt.Errorf("insufficient nodes: prefill=%v decode=%v", prefill, decode)
	}

	req.PrefillNode = prefill.NodeID
	req.DecodeNode = decode.NodeID
	req.StartedAt = time.Now()

	prefill.ActiveRequests++
	decode.ActiveRequests++
	kvSize := float64(req.SequenceLen) * 2.0 * 128.0 * 80.0 / (1024 * 1024 * 1024) // approx GiB
	req.KVCacheSizeMB = kvSize * 1024
	decode.MemoryUsedGB += kvSize

	s.activeRequests[req.ID] = req
	s.totalScheduled++
	return req, nil
}

// Stats returns current scheduler statistics
func (s *MooncakeScheduler) Stats() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return map[string]interface{}{
		"pending_count":    len(s.pendingQueue),
		"active_count":     len(s.activeRequests),
		"total_scheduled":  s.totalScheduled,
		"total_evicted":    s.totalEvicted,
		"prefill_nodes":    len(s.prefillNodes),
		"decode_nodes":     len(s.decodeNodes),
	}
}
