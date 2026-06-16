package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"mineio/internal/common"
	"net/http"
	"os"
	"sync"
	"time"
)

type RobotService struct {
	mu                sync.RWMutex
	state             common.RobotState
	ah                *common.ArrowheadClient
	simFail           bool    // returns 503 on /state
	degraded          bool    // slows SLAM rate, adds latency
	slamSlowPct       float64 // fraction of normal SLAM speed when degraded (default 0.3)
	latencyMs         int     // extra response delay when degraded
	slamRatePctPerSec float64 // % per second; default 0.05
	lkgMu             sync.RWMutex
	lkgState          common.RobotState
	lkgAt             time.Time
}

func main() {
	id := envOrDefault("IDT_ID", "idt1a")
	name := envOrDefault("IDT_NAME", "Inspection Robot A")
	port := envOrDefault("PORT", "8101")
	ahURL := envOrDefault("ARROWHEAD_URL", "http://localhost:8000")
	host := envOrDefault("HOST", "localhost")

	log.Printf("[%s] Starting %s on :%s", id, name, port)

	svc := &RobotService{
		ah:                common.NewArrowheadClient(ahURL, id),
		slamSlowPct:       0.3,
		slamRatePctPerSec: 0.05,
		state: common.RobotState{
			ID:               id,
			Name:             name,
			Online:           true,
			Connected:        true,
			Position:         common.Position{X: rand.Float64() * 50, Y: rand.Float64() * 50, Z: 0},
			BatteryPct:       100.0,
			MappingProgress:  0.0,
			SlamActive:       false,
			NavigationStatus: "idle",
			HazardsDetected:  []common.Hazard{},
			AreaCoveredSqm:   0,
			LastUpdated:      time.Now(),
		},
	}

	portInt := 8101
	fmt.Sscanf(port, "%d", &portInt)

	common.RegisterWithRetry(svc.ah, common.RegisterRequest{
		ID:           id,
		Name:         name,
		Address:      host,
		Port:         portInt,
		ServiceType:  "iDT",
		Capabilities: []string{"mapping", "localization", "hazard-detection", "navigation", "slam"},
		Metadata:     map[string]string{"machine": "inspection-robot"},
	}, 10)

	go svc.simulate()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		common.WriteJSON(w, 200, map[string]interface{}{"status": "ok", "id": id, "timestamp": time.Now()})
	})
	mux.HandleFunc("/state", svc.handleState)
	mux.HandleFunc("/map", svc.handleMap)
	mux.HandleFunc("/hazards", svc.handleHazards)
	mux.HandleFunc("/navigate", svc.handleNavigate)
	mux.HandleFunc("/slam/start", svc.handleSlamStart)
	mux.HandleFunc("/slam/stop", svc.handleSlamStop)
	mux.HandleFunc("/connectivity", svc.handleConnectivity)
	mux.HandleFunc("/online", svc.handleOnline)
	mux.HandleFunc("/hazard/inject", svc.handleHazardInject)
	mux.HandleFunc("/hazard/clear", svc.handleHazardClear)
	mux.HandleFunc("/simulate/reset", svc.handleSimulateReset)
	mux.HandleFunc("/simulate/slam-rate", svc.handleSimulateSlamRate)
	mux.HandleFunc("/simulate/fail", svc.handleSimulateFail)
	mux.HandleFunc("/simulate/degrade", svc.handleSimulateDegrade)
	mux.HandleFunc("/simulate/recover", svc.handleSimulateRecover)

	handler := common.CORSMiddleware(mux)
	log.Fatal(http.ListenAndServe(":"+port, handler))
}

func (s *RobotService) simulateTick() {
	hazardTypes := []string{"loose-rock", "misfire", "structural"}
	severities := []string{"low", "medium", "high"}
	s.mu.Lock()
	if s.state.Online && s.state.Connected {
		// Battery drain
		if s.state.BatteryPct > 0 {
			s.state.BatteryPct = math.Max(0, s.state.BatteryPct-0.01)
		}
		// Position drift when navigating
		if s.state.NavigationStatus == "navigating" {
			s.state.Position.X += (rand.Float64() - 0.5) * 2
			s.state.Position.Y += (rand.Float64() - 0.5) * 2
			s.state.Position.X = math.Max(0, math.Min(100, s.state.Position.X))
			s.state.Position.Y = math.Max(0, math.Min(100, s.state.Position.Y))
		}
		// SLAM / mapping progress (slowed when degraded)
		if s.state.SlamActive && s.state.MappingProgress < 100 {
			rate := s.slamRatePctPerSec
			if s.degraded {
				rate = s.slamRatePctPerSec * s.slamSlowPct
			}
			s.state.MappingProgress = math.Min(100, s.state.MappingProgress+rate)
			s.state.AreaCoveredSqm = s.state.MappingProgress * 50
		}
		// Random hazard detection (low probability)
		if rand.Float64() < 0.005 && len(s.state.HazardsDetected) < 5 {
			h := common.Hazard{
				ID:         fmt.Sprintf("haz-%d", time.Now().UnixNano()),
				Type:       hazardTypes[rand.Intn(len(hazardTypes))],
				Severity:   severities[rand.Intn(len(severities))],
				Position:   common.Position{X: rand.Float64() * 100, Y: rand.Float64() * 100, Z: 0},
				DetectedAt: time.Now(),
				Cleared:    false,
			}
			s.state.HazardsDetected = append(s.state.HazardsDetected, h)
			log.Printf("[%s] Hazard detected: %s (%s)", s.state.ID, h.Type, h.Severity)
		}
		s.state.LastUpdated = time.Now()
		snapshot := s.state
		s.mu.Unlock()
		if !s.simFail {
			s.lkgMu.Lock()
			s.lkgState = snapshot
			s.lkgAt = time.Now()
			s.lkgMu.Unlock()
		}
	} else {
		s.mu.Unlock()
	}
}

func (s *RobotService) simulate() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		s.simulateTick()
	}
}

func (s *RobotService) handleState(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	fail := s.simFail
	latMs := s.latencyMs
	degraded := s.degraded
	state := s.state
	s.mu.RUnlock()

	if fail {
		s.lkgMu.RLock()
		hasLKG := !s.lkgAt.IsZero()
		lkg := s.lkgState
		lkgAt := s.lkgAt
		s.lkgMu.RUnlock()
		if !hasLKG {
			common.WriteError(w, 503, "robot failure simulated")
			return
		}
		staleSinceMs := time.Since(lkgAt).Milliseconds()
		type staleResp struct {
			common.RobotState
			SimDegraded  bool  `json:"simDegraded"`
			Stale        bool  `json:"stale"`
			StaleSinceMs int64 `json:"staleSinceMs"`
		}
		common.WriteJSON(w, 200, staleResp{lkg, degraded, true, staleSinceMs})
		return
	}

	if degraded && latMs > 0 {
		time.Sleep(time.Duration(latMs) * time.Millisecond)
	}
	type resp struct {
		common.RobotState
		SimDegraded bool `json:"simDegraded"`
		Stale       bool `json:"stale"`
	}
	common.WriteJSON(w, 200, resp{state, degraded, false})
}

func (s *RobotService) handleMap(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	prog := s.state.MappingProgress
	area := s.state.AreaCoveredSqm
	slam := s.state.SlamActive
	s.mu.RUnlock()

	// Generate simple waypoint grid based on progress
	waypoints := []map[string]float64{}
	n := int(prog / 10)
	for i := 0; i < n; i++ {
		waypoints = append(waypoints, map[string]float64{
			"x": float64(i%10) * 10,
			"y": float64(i/10) * 10,
		})
	}
	common.WriteJSON(w, 200, map[string]interface{}{
		"coverage":   prog,
		"areaSqm":    area,
		"slamActive": slam,
		"waypoints":  waypoints,
	})
}

func (s *RobotService) handleHazards(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	hazards := s.state.HazardsDetected
	s.mu.RUnlock()
	if hazards == nil {
		hazards = []common.Hazard{}
	}
	common.WriteJSON(w, 200, map[string]interface{}{"hazards": hazards, "count": len(hazards)})
}

func (s *RobotService) handleNavigate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	var body struct {
		X float64 `json:"x"`
		Y float64 `json:"y"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	s.mu.Lock()
	s.state.NavigationStatus = "navigating"
	s.mu.Unlock()
	log.Printf("[%s] Navigating to (%.1f, %.1f)", s.state.ID, body.X, body.Y)
	common.WriteJSON(w, 200, map[string]interface{}{"status": "navigating", "target": body})
}

func (s *RobotService) handleSlamStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	s.mu.Lock()
	s.state.SlamActive = true
	s.state.NavigationStatus = "mapping"
	s.mu.Unlock()
	log.Printf("[%s] SLAM started", s.state.ID)
	common.WriteJSON(w, 200, map[string]string{"status": "slam started"})
}

func (s *RobotService) handleSlamStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	s.mu.Lock()
	s.state.SlamActive = false
	s.state.NavigationStatus = "idle"
	s.mu.Unlock()
	log.Printf("[%s] SLAM stopped", s.state.ID)
	common.WriteJSON(w, 200, map[string]string{"status": "slam stopped"})
}

func (s *RobotService) handleConnectivity(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		common.WriteError(w, 405, "PUT required")
		return
	}
	var body struct {
		Connected bool `json:"connected"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	s.mu.Lock()
	s.state.Connected = body.Connected
	s.mu.Unlock()
	log.Printf("[%s] Connected set to %v", s.state.ID, body.Connected)
	common.WriteJSON(w, 200, map[string]bool{"connected": body.Connected})
}

func (s *RobotService) handleOnline(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		common.WriteError(w, 405, "PUT required")
		return
	}
	var body struct {
		Online bool `json:"online"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	s.mu.Lock()
	s.state.Online = body.Online
	s.mu.Unlock()
	log.Printf("[%s] Online set to %v", s.state.ID, body.Online)
	common.WriteJSON(w, 200, map[string]bool{"online": body.Online})
}

func (s *RobotService) handleHazardInject(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	var body struct {
		Type     string `json:"type"`
		Severity string `json:"severity"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	if body.Type == "" {
		body.Type = "loose-rock"
	}
	if body.Severity == "" {
		body.Severity = "high"
	}
	h := common.Hazard{
		ID:         fmt.Sprintf("haz-%d", time.Now().UnixNano()),
		Type:       body.Type,
		Severity:   body.Severity,
		Position:   common.Position{X: rand.Float64() * 100, Y: rand.Float64() * 100, Z: 0},
		DetectedAt: time.Now(),
		Cleared:    false,
	}
	s.mu.Lock()
	s.state.HazardsDetected = append(s.state.HazardsDetected, h)
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] Hazard injected: %s (%s)", id, h.Type, h.Severity)
	common.WriteJSON(w, 201, h)
}

func (s *RobotService) handleHazardClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	var body struct {
		ID string `json:"id"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	s.mu.Lock()
	now := time.Now()
	for i := range s.state.HazardsDetected {
		if s.state.HazardsDetected[i].ID == body.ID || body.ID == "" {
			s.state.HazardsDetected[i].Cleared = true
			s.state.HazardsDetected[i].ClearedAt = &now
		}
	}
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] Hazard cleared: %s", id, body.ID)
	common.WriteJSON(w, 200, map[string]string{"status": "cleared"})
}

func (s *RobotService) handleSimulateReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	s.mu.Lock()
	s.state.BatteryPct = 100.0
	s.state.MappingProgress = 0.0
	s.state.AreaCoveredSqm = 0.0
	s.state.SlamActive = false
	s.state.NavigationStatus = "idle"
	s.state.HazardsDetected = []common.Hazard{}
	s.state.LastUpdated = time.Now()
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] State reset to initial values.", id)
	common.WriteJSON(w, 200, map[string]string{"status": "reset"})
}

// handleSimulateSlamRate sets the SLAM rate so that 100% coverage is reached in durationSec seconds.
// Body: {"durationSec": 30}
func (s *RobotService) handleSimulateSlamRate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	var body struct {
		DurationSec float64 `json:"durationSec"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	if body.DurationSec <= 0 {
		body.DurationSec = 30
	}
	rate := 100.0 / body.DurationSec
	s.mu.Lock()
	s.slamRatePctPerSec = rate
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] SLAM rate set to %.4f%%/s (100%% in %.0fs)", id, rate, body.DurationSec)
	common.WriteJSON(w, 200, map[string]interface{}{"status": "ok", "slamRatePctPerSec": rate, "durationSec": body.DurationSec})
}

func (s *RobotService) handleSimulateFail(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	s.mu.Lock()
	s.simFail = true
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] FAILURE MODE activated – /state will return 503", id)
	common.WriteJSON(w, 200, map[string]string{"status": "failure mode activated"})
}

// handleSimulateDegrade slows SLAM and adds response latency.
// Body: {"slamSlowPct": 0.3, "latencyMs": 150}
func (s *RobotService) handleSimulateDegrade(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	var body struct {
		SlamSlowPct float64 `json:"slamSlowPct"` // 0–1, fraction of normal rate
		LatencyMs   int     `json:"latencyMs"`
	}
	body.SlamSlowPct = 0.3
	body.LatencyMs = 150
	json.NewDecoder(r.Body).Decode(&body)
	if body.SlamSlowPct <= 0 || body.SlamSlowPct > 1 {
		body.SlamSlowPct = 0.3
	}
	s.mu.Lock()
	s.degraded = true
	s.slamSlowPct = body.SlamSlowPct
	s.latencyMs = body.LatencyMs
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] DEGRADE MODE: slamSlowPct=%.2f latencyMs=%d", id, body.SlamSlowPct, body.LatencyMs)
	common.WriteJSON(w, 200, map[string]interface{}{
		"status": "degraded", "slamSlowPct": body.SlamSlowPct, "latencyMs": body.LatencyMs,
	})
}

func (s *RobotService) handleSimulateRecover(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	s.mu.Lock()
	s.simFail = false
	s.degraded = false
	s.slamSlowPct = 0.3
	s.latencyMs = 0
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] RECOVERED – normal operation restored", id)
	common.WriteJSON(w, 200, map[string]string{"status": "recovered"})
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
