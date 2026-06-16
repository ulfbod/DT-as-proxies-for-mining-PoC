package main

import (
	"encoding/json"
	"mineio/internal/common"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func newTestGasService() *GasService {
	s := &GasService{}
	s.state = common.GasSensorState{
		ID:        "test-gas",
		Name:      "Test Gas Sensor",
		Online:    true,
		Connected: true,
		GasLevels: common.GasLevels{
			CH4: 0.1, CO: 5.0, CO2: 0.04, O2: 20.9, NO2: 0.5,
		},
		Alerts:            []common.GasAlert{},
		EnvironmentStatus: "safe",
		LastUpdated:       time.Now(),
	}
	return s
}

// T1: GET /state returns current data and stale=false when not in fail mode
func TestGasStateNotStaleWhenHealthy(t *testing.T) {
	s := newTestGasService()
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
func TestGasState503WhenFailNoLKG(t *testing.T) {
	s := newTestGasService()
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
func TestGasStateReturnsLKGWhenFailed(t *testing.T) {
	s := newTestGasService()
	s.simFail = true
	s.lkgState = s.state
	s.lkgState.GasLevels.CH4 = 0.75 // distinguish from initial value
	s.lkgAt = time.Now().Add(-3 * time.Second)

	req := httptest.NewRequest("GET", "/state", nil)
	w := httptest.NewRecorder()
	s.handleState(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 (stale), got %d", w.Code)
	}
	var resp struct {
		GasLevels struct {
			CH4 float64 `json:"ch4"`
		} `json:"gasLevels"`
		Stale        bool  `json:"stale"`
		StaleSinceMs int64 `json:"staleSinceMs"`
	}
	json.NewDecoder(w.Body).Decode(&resp)
	if !resp.Stale {
		t.Error("stale should be true when returning LKG data")
	}
	if resp.GasLevels.CH4 != 0.75 {
		t.Errorf("expected LKG CH4=0.75, got %f", resp.GasLevels.CH4)
	}
	if resp.StaleSinceMs <= 0 {
		t.Error("staleSinceMs should be positive")
	}
}

// T4: LKG is updated on each successful simulateTick()
func TestGasLKGUpdatedOnSimulateTick(t *testing.T) {
	s := newTestGasService()

	s.simulateTick()

	if s.lkgAt.IsZero() {
		t.Error("lkgAt should be set after simulate tick")
	}
}

// T5: On recovery (simFail=false), /state returns fresh data with stale=false
func TestGasStateNotStaleAfterRecovery(t *testing.T) {
	s := newTestGasService()
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
