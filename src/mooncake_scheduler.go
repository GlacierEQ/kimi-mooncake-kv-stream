// Package mooncake implements a local disaggregated prefill/decode scheduling reference.
// It performs in-process queue and node-selection logic only; it does not contact Kimi,
// Moonshot AI, or a distributed inference service.
package mooncake

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// RequestPriority determines scheduling order in the local queue.
type RequestPriority int

const (
	PriorityRealtime RequestPriority = iota
	PriorityInteractive
	PriorityBatch
	PriorityColdStart
)

// InferenceRequest represents one local scheduling request.
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

// NodeMetrics holds local node-capacity inputs used by the selection heuristic.
type NodeMetrics struct {
	NodeID          string
	MemoryUsedGB    float64
	MemoryTotalGB   float64
	ActiveRequests  int
	AvgLatencyMs    float64
	KVCachePressure float64
}

// MooncakeScheduler is an in-process priority queue and node-selection model.
type MooncakeScheduler struct {
	mu             sync.RWMutex
	prefillNodes   map[string]*NodeMetrics
	decodeNodes    map[string]*NodeMetrics
	pendingQueue   []*InferenceRequest
	activeRequests map[string]*InferenceRequest
	totalScheduled uint64
	totalEvicted   uint64
}

// NewMooncakeScheduler creates empty local prefill and decode node pools.
func NewMooncakeScheduler() *MooncakeScheduler {
	return &MooncakeScheduler{
		prefillNodes:   make(map[string]*NodeMetrics),
		decodeNodes:    make(map[string]*NodeMetrics),
		pendingQueue:   make([]*InferenceRequest, 0, 1024),
		activeRequests: make(map[string]*InferenceRequest),
	}
}

// RegisterPrefillNode adds a local prefill-capable node to the model.
func (s *MooncakeScheduler) RegisterPrefillNode(nodeID string, memoryGB float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prefillNodes[nodeID] = &NodeMetrics{NodeID: nodeID, MemoryTotalGB: memoryGB}
}

// RegisterDecodeNode adds a local decode-capable node to the model.
func (s *MooncakeScheduler) RegisterDecodeNode(nodeID string, memoryGB float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decodeNodes[nodeID] = &NodeMetrics{NodeID: nodeID, MemoryTotalGB: memoryGB}
}

// Submit validates and enqueues a request according to local priority.
func (s *MooncakeScheduler) Submit(req *InferenceRequest) error {
	if req == nil {
		return fmt.Errorf("request must not be nil")
	}
	if req.ID == "" {
		return fmt.Errorf("request ID must not be empty")
	}
	if req.SequenceLen < 1 {
		return fmt.Errorf("sequence length must be positive")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.activeRequests[req.ID]; exists {
		return fmt.Errorf("request already active: %s", req.ID)
	}
	for _, pending := range s.pendingQueue {
		if pending.ID == req.ID {
			return fmt.Errorf("request already pending: %s", req.ID)
		}
	}

	req.CreatedAt = time.Now()
	s.pendingQueue = append(s.pendingQueue, req)
	sort.SliceStable(s.pendingQueue, func(i, j int) bool {
		return s.pendingQueue[i].Priority < s.pendingQueue[j].Priority
	})
	return nil
}

// selectBestNode picks the node with the lowest local pressure score.
func (s *MooncakeScheduler) selectBestNode(nodes map[string]*NodeMetrics) *NodeMetrics {
	var best *NodeMetrics
	bestScore := math.MaxFloat64
	for _, node := range nodes {
		pressure := node.MemoryUsedGB / math.Max(node.MemoryTotalGB, 1.0)
		score := pressure*0.6 + float64(node.ActiveRequests)*0.3 + node.KVCachePressure*0.1
		if score < bestScore {
			bestScore = score
			best = node
		}
	}
	return best
}

// ScheduleNext assigns the highest-priority pending request to local modeled nodes.
func (s *MooncakeScheduler) ScheduleNext() (*InferenceRequest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.pendingQueue) == 0 {
		return nil, fmt.Errorf("no pending requests")
	}

	prefill := s.selectBestNode(s.prefillNodes)
	decode := s.selectBestNode(s.decodeNodes)
	if prefill == nil || decode == nil {
		// Preserve the pending request when capacity is unavailable.
		return nil, fmt.Errorf("insufficient nodes: prefill=%v decode=%v", prefill, decode)
	}

	req := s.pendingQueue[0]
	s.pendingQueue = s.pendingQueue[1:]
	req.PrefillNode = prefill.NodeID
	req.DecodeNode = decode.NodeID
	req.StartedAt = time.Now()

	prefill.ActiveRequests++
	decode.ActiveRequests++
	kvSizeGB := float64(req.SequenceLen) * 2.0 * 128.0 * 80.0 / (1024 * 1024 * 1024)
	req.KVCacheSizeMB = kvSizeGB * 1024
	decode.MemoryUsedGB += kvSizeGB

	s.activeRequests[req.ID] = req
	s.totalScheduled++
	return req, nil
}

// Stats returns local scheduler counters and pool sizes.
func (s *MooncakeScheduler) Stats() map[string]interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return map[string]interface{}{
		"pending_count":   len(s.pendingQueue),
		"active_count":    len(s.activeRequests),
		"total_scheduled": s.totalScheduled,
		"total_evicted":   s.totalEvicted,
		"prefill_nodes":   len(s.prefillNodes),
		"decode_nodes":    len(s.decodeNodes),
	}
}
