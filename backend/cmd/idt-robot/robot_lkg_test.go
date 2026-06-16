package main

import (
	"encoding/json"
	"mineio/internal/common"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func newTestService() *RobotService {
	s := &RobotService{
		slamRatePctPerSec: 0.05,
		slamSlowPct:       0.3,
	}
	s.state = common.RobotState{
		ID:               "test-robot",
		Name:             "Test Robot",
		Online:           true,
		Connected:        true,
		BatteryPct:       85.0,
		MappingProgress:  42.0,
		HazardsDetected:  []common.Hazard{},
		LastUpdated:      time.Now(),
	}
	return s
}

// T1: GET /state returns current data and stale=false when not in fail mode
func TestStateNotStaleWhenHealthy(t *testing.T) {
	s := newTestService()
	s.lkgState = s.state
	s.lkgAt = time.Now().Add(-5 * time.Second)

	req := httptest.NewRequest("GET", "/state", nil)
	w := httptest.NewRecorder()
	s.handleState(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp struct {
		Stale bool `json:"stale"`
	}
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Stale {
		t.Error("stale should be false when service is healthy")
	}
}

// T2: GET /state returns 503 when simFail=true and no LKG exists
func TestState503WhenFailNoLKG(t *testing.T) {
	s := newTestService()
	s.simFail = true
	// lkgAt is zero value — no LKG stored

	req := httptest.NewRequest("GET", "/state", nil)
	w := httptest.NewRecorder()
	s.handleState(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

// T3: GET /state returns stale data with stale=true when simFail=true and LKG exists
func TestStateReturnsLKGWhenFailed(t *testing.T) {
	s := newTestService()
	s.simFail = true
	s.lkgState = s.state
	s.lkgState.MappingProgress = 55.0
	s.lkgAt = time.Now().Add(-3 * time.Second)

	req := httptest.NewRequest("GET", "/state", nil)
	w := httptest.NewRecorder()
	s.handleState(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 (stale), got %d", w.Code)
	}
	var resp struct {
		MappingProgress float64 `json:"mappingProgress"`
		Stale           bool    `json:"stale"`
		StaleSinceMs    int64   `json:"staleSinceMs"`
	}
	json.NewDecoder(w.Body).Decode(&resp)
	if !resp.Stale {
		t.Error("stale should be true when returning LKG data")
	}
	if resp.MappingProgress != 55.0 {
		t.Errorf("expected LKG mappingProgress=55.0, got %f", resp.MappingProgress)
	}
	if resp.StaleSinceMs <= 0 {
		t.Error("staleSinceMs should be positive")
	}
}

// T4: LKG is updated on each successful simulateTick()
func TestLKGUpdatedOnSimulateTick(t *testing.T) {
	s := newTestService()
	initialProgress := s.state.MappingProgress

	s.simulateTick()

	if s.lkgAt.IsZero() {
		t.Error("lkgAt should be set after simulate tick")
	}
	// lkgState should reflect post-tick state
	if s.lkgState.MappingProgress == initialProgress && s.state.MappingProgress != initialProgress {
		t.Error("lkgState not updated after tick")
	}
}

// T5: On recovery (simFail=false), /state returns fresh data with stale=false
func TestStateNotStaleAfterRecovery(t *testing.T) {
	s := newTestService()
	s.simFail = true
	s.lkgState = s.state
	s.lkgAt = time.Now().Add(-2 * time.Second)

	s.simFail = false

	req := httptest.NewRequest("GET", "/state", nil)
	w := httptest.NewRecorder()
	s.handleState(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp struct {
		Stale bool `json:"stale"`
	}
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Stale {
		t.Error("stale should be false after recovery")
	}
}
