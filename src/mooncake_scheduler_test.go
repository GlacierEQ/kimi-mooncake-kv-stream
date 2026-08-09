package mooncake

import "testing"

func request(id string, priority RequestPriority) *InferenceRequest {
	return &InferenceRequest{ID: id, Priority: priority, SequenceLen: 1024}
}

func TestSchedulePreservesPendingRequestWithoutNodes(t *testing.T) {
	s := NewMooncakeScheduler()
	if err := s.Submit(request("req-1", PriorityInteractive)); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if _, err := s.ScheduleNext(); err == nil {
		t.Fatal("expected insufficient-node error")
	}
	if got := s.Stats()["pending_count"]; got != 1 {
		t.Fatalf("pending_count=%v, want 1", got)
	}
}

func TestPriorityAndNodeAssignment(t *testing.T) {
	s := NewMooncakeScheduler()
	s.RegisterPrefillNode("prefill-a", 64)
	s.RegisterDecodeNode("decode-a", 64)
	if err := s.Submit(request("batch", PriorityBatch)); err != nil {
		t.Fatalf("submit batch: %v", err)
	}
	if err := s.Submit(request("realtime", PriorityRealtime)); err != nil {
		t.Fatalf("submit realtime: %v", err)
	}

	next, err := s.ScheduleNext()
	if err != nil {
		t.Fatalf("schedule: %v", err)
	}
	if next.ID != "realtime" {
		t.Fatalf("scheduled %q, want realtime", next.ID)
	}
	if next.PrefillNode != "prefill-a" || next.DecodeNode != "decode-a" {
		t.Fatalf("unexpected node assignment: %#v", next)
	}
}

func TestSubmitRejectsInvalidOrDuplicateRequests(t *testing.T) {
	s := NewMooncakeScheduler()
	if err := s.Submit(nil); err == nil {
		t.Fatal("expected nil request rejection")
	}
	if err := s.Submit(&InferenceRequest{ID: "", SequenceLen: 1}); err == nil {
		t.Fatal("expected empty ID rejection")
	}
	if err := s.Submit(&InferenceRequest{ID: "bad", SequenceLen: 0}); err == nil {
		t.Fatal("expected invalid sequence rejection")
	}
	if err := s.Submit(request("dup", PriorityBatch)); err != nil {
		t.Fatalf("initial submit: %v", err)
	}
	if err := s.Submit(request("dup", PriorityRealtime)); err == nil {
		t.Fatal("expected duplicate request rejection")
	}
}
