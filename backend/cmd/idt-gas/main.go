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

// Gas threshold constants
const (
	thresholdCH4    = 1.0  // %
	thresholdCO     = 25.0 // ppm
	thresholdO2Low  = 19.5 // %
	thresholdO2High = 23.0 // %
	thresholdNO2    = 3.0  // ppm
)

type simDegradeParams struct {
	active         bool
	noiseFactor    float64 // multiplier for noise amplitude (>1 = noisier/less accurate)
	extraLatencyMs int     // artificial response delay
}

type GasService struct {
	mu       sync.RWMutex
	state    common.GasSensorState
	ah       *common.ArrowheadClient
	simFail  bool             // when true, all state endpoints return HTTP 503
	degrade  simDegradeParams // when active, adds noise and latency
	lkgMu    sync.RWMutex
	lkgState common.GasSensorState
	lkgAt    time.Time
}

func main() {
	id := envOrDefault("IDT_ID", "idt2a")
	name := envOrDefault("IDT_NAME", "Gas Sensing Unit A")
	port := envOrDefault("PORT", "8201")
	ahURL := envOrDefault("ARROWHEAD_URL", "http://localhost:8000")
	host := envOrDefault("HOST", "localhost")

	log.Printf("[%s] Starting %s on :%s", id, name, port)

	svc := &GasService{
		ah: common.NewArrowheadClient(ahURL, id),
		state: common.GasSensorState{
			ID:        id,
			Name:      name,
			Online:    true,
			Connected: true,
			Position:  common.Position{X: rand.Float64() * 100, Y: rand.Float64() * 100, Z: 0},
			GasLevels: common.GasLevels{
				CH4: 0.1, CO: 5.0, CO2: 0.04, O2: 20.9, NO2: 0.5,
			},
			Alerts:            []common.GasAlert{},
			EnvironmentStatus: "safe",
			LastUpdated:       time.Now(),
		},
	}

	portInt := 8201
	fmt.Sscanf(port, "%d", &portInt)

	common.RegisterWithRetry(svc.ah, common.RegisterRequest{
		ID:           id,
		Name:         name,
		Address:      host,
		Port:         portInt,
		ServiceType:  "iDT",
		Capabilities: []string{"gas-measurement", "gas-alert", "environment-monitoring"},
		Metadata:     map[string]string{"machine": "gas-sensor"},
	}, 10)

	go svc.simulate()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		common.WriteJSON(w, 200, map[string]interface{}{"status": "ok", "id": id, "timestamp": time.Now()})
	})
	mux.HandleFunc("/state", svc.handleState)
	mux.HandleFunc("/measurements", svc.handleMeasurements)
	mux.HandleFunc("/alerts", svc.handleAlerts)
	mux.HandleFunc("/connectivity", svc.handleConnectivity)
	mux.HandleFunc("/online", svc.handleOnline)
	mux.HandleFunc("/simulate/gas", svc.handleSimulateGas)
	mux.HandleFunc("/simulate/spike", svc.handleSimulateSpike)
	// QoS failure injection endpoints
	mux.HandleFunc("/simulate/fail", svc.handleSimulateFail)
	mux.HandleFunc("/simulate/degrade", svc.handleSimulateDegrade)
	mux.HandleFunc("/simulate/recover", svc.handleSimulateRecover)

	handler := common.CORSMiddleware(mux)
	log.Fatal(http.ListenAndServe(":"+port, handler))
}

func (s *GasService) simulateTick() {
	s.mu.Lock()
	if s.state.Online && s.state.Connected && !s.simFail {
		g := &s.state.GasLevels
		nf := 1.0
		if s.degrade.active {
			nf = s.degrade.noiseFactor
		}
		// Fluctuate gas levels with realistic noise (amplified when degraded)
		g.CH4 = clamp(g.CH4+(rand.Float64()-0.48)*0.02*nf, 0, 5)
		g.CO = clamp(g.CO+(rand.Float64()-0.48)*0.5*nf, 0, 200)
		g.CO2 = clamp(g.CO2+(rand.Float64()-0.5)*0.005*nf, 0.03, 5)
		g.O2 = clamp(g.O2+(rand.Float64()-0.5)*0.05*nf, 16, 25)
		g.NO2 = clamp(g.NO2+(rand.Float64()-0.48)*0.05*nf, 0, 20)
		s.evaluateAlerts()
		s.state.LastUpdated = time.Now()
		snapshot := s.state
		s.mu.Unlock()
		s.lkgMu.Lock()
		s.lkgState = snapshot
		s.lkgAt = time.Now()
		s.lkgMu.Unlock()
	} else {
		s.mu.Unlock()
	}
}

func (s *GasService) simulate() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		s.simulateTick()
	}
}

// evaluateAlerts must be called with s.mu held (write lock).
func (s *GasService) evaluateAlerts() {
	now := time.Now()
	type check struct {
		gas       string
		level     float64
		threshold float64
		exceeded  bool
	}
	checks := []check{
		{"CH4", s.state.GasLevels.CH4, thresholdCH4, s.state.GasLevels.CH4 > thresholdCH4},
		{"CO", s.state.GasLevels.CO, thresholdCO, s.state.GasLevels.CO > thresholdCO},
		{"O2-low", s.state.GasLevels.O2, thresholdO2Low, s.state.GasLevels.O2 < thresholdO2Low},
		{"O2-high", s.state.GasLevels.O2, thresholdO2High, s.state.GasLevels.O2 > thresholdO2High},
		{"NO2", s.state.GasLevels.NO2, thresholdNO2, s.state.GasLevels.NO2 > thresholdNO2},
	}
	for _, c := range checks {
		found := false
		for i := range s.state.Alerts {
			if s.state.Alerts[i].Gas == c.gas && s.state.Alerts[i].Active {
				found = true
				if !c.exceeded {
					s.state.Alerts[i].Active = false
				} else {
					s.state.Alerts[i].Level = c.level
				}
				break
			}
		}
		if !found && c.exceeded {
			s.state.Alerts = append(s.state.Alerts, common.GasAlert{
				ID:        fmt.Sprintf("alert-%s-%d", c.gas, now.UnixNano()),
				Gas:       c.gas,
				Level:     c.level,
				Threshold: c.threshold,
				Location:  s.state.Position,
				Timestamp: now,
				Active:    true,
			})
			log.Printf("[%s] Gas alert: %s = %.3f (threshold %.3f)", s.state.ID, c.gas, c.level, c.threshold)
		}
	}
	var active, inactive []common.GasAlert
	for _, a := range s.state.Alerts {
		if a.Active {
			active = append(active, a)
		} else {
			inactive = append(inactive, a)
		}
	}
	if len(inactive) > 20 {
		inactive = inactive[len(inactive)-20:]
	}
	s.state.Alerts = append(active, inactive...)
	hasActive := len(active) > 0
	hasCritical := false
	for _, a := range active {
		if a.Gas == "CH4" || a.Gas == "CO" || a.Gas == "O2-low" {
			hasCritical = true
			break
		}
	}
	switch {
	case hasCritical:
		s.state.EnvironmentStatus = "danger"
	case hasActive:
		s.state.EnvironmentStatus = "warning"
	default:
		s.state.EnvironmentStatus = "safe"
	}
}

// failCheck returns true and writes 503 if simFail mode is active.
func (s *GasService) failCheck(w http.ResponseWriter) bool {
	s.mu.RLock()
	fail := s.simFail
	s.mu.RUnlock()
	if fail {
		common.WriteError(w, 503, "sensor failure simulated")
		return true
	}
	return false
}

// degradeLatency adds an artificial delay when degraded.
func (s *GasService) degradeLatency() {
	s.mu.RLock()
	d := s.degrade
	s.mu.RUnlock()
	if d.active && d.extraLatencyMs > 0 {
		time.Sleep(time.Duration(d.extraLatencyMs) * time.Millisecond)
	}
}

func (s *GasService) handleState(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	fail := s.simFail
	degraded := s.degrade.active
	state := s.state
	s.mu.RUnlock()

	if fail {
		s.lkgMu.RLock()
		hasLKG := !s.lkgAt.IsZero()
		lkg := s.lkgState
		lkgAt := s.lkgAt
		s.lkgMu.RUnlock()
		if !hasLKG {
			common.WriteError(w, 503, "sensor failure simulated")
			return
		}
		staleSinceMs := time.Since(lkgAt).Milliseconds()
		type staleResp struct {
			common.GasSensorState
			SimDegraded  bool  `json:"simDegraded"`
			Stale        bool  `json:"stale"`
			StaleSinceMs int64 `json:"staleSinceMs"`
		}
		common.WriteJSON(w, 200, staleResp{lkg, degraded, true, staleSinceMs})
		return
	}

	s.degradeLatency()
	type resp struct {
		common.GasSensorState
		SimDegraded bool `json:"simDegraded"`
		Stale       bool `json:"stale"`
	}
	common.WriteJSON(w, 200, resp{state, degraded, false})
}

func (s *GasService) handleMeasurements(w http.ResponseWriter, r *http.Request) {
	if s.failCheck(w) {
		return
	}
	s.degradeLatency()
	s.mu.RLock()
	levels := s.state.GasLevels
	alerts := s.state.Alerts
	ts := s.state.LastUpdated
	s.mu.RUnlock()
	if alerts == nil {
		alerts = []common.GasAlert{}
	}
	common.WriteJSON(w, 200, map[string]interface{}{
		"gasLevels": levels, "alerts": alerts, "timestamp": ts,
	})
}

func (s *GasService) handleAlerts(w http.ResponseWriter, r *http.Request) {
	if s.failCheck(w) {
		return
	}
	s.mu.RLock()
	alerts := s.state.Alerts
	s.mu.RUnlock()
	active := []common.GasAlert{}
	for _, a := range alerts {
		if a.Active {
			active = append(active, a)
		}
	}
	common.WriteJSON(w, 200, map[string]interface{}{"alerts": active, "count": len(active)})
}

func (s *GasService) handleConnectivity(w http.ResponseWriter, r *http.Request) {
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
	common.WriteJSON(w, 200, map[string]bool{"connected": body.Connected})
}

func (s *GasService) handleOnline(w http.ResponseWriter, r *http.Request) {
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
	common.WriteJSON(w, 200, map[string]bool{"online": body.Online})
}

func (s *GasService) handleSimulateGas(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	var body struct {
		CH4 *float64 `json:"ch4"`
		CO  *float64 `json:"co"`
		CO2 *float64 `json:"co2"`
		O2  *float64 `json:"o2"`
		NO2 *float64 `json:"no2"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		common.WriteError(w, 400, "invalid JSON")
		return
	}
	s.mu.Lock()
	if body.CH4 != nil {
		s.state.GasLevels.CH4 = *body.CH4
	}
	if body.CO != nil {
		s.state.GasLevels.CO = *body.CO
	}
	if body.CO2 != nil {
		s.state.GasLevels.CO2 = *body.CO2
	}
	if body.O2 != nil {
		s.state.GasLevels.O2 = *body.O2
	}
	if body.NO2 != nil {
		s.state.GasLevels.NO2 = *body.NO2
	}
	s.evaluateAlerts()
	levels := s.state.GasLevels
	s.mu.Unlock()
	log.Printf("[%s] Gas levels overridden via /simulate/gas", s.state.ID)
	common.WriteJSON(w, 200, map[string]interface{}{"status": "applied", "gasLevels": levels})
}

func (s *GasService) handleSimulateSpike(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	s.mu.Lock()
	s.state.GasLevels.CH4 = 2.5
	s.state.GasLevels.CO = 80.0
	s.state.GasLevels.O2 = 17.0
	s.state.GasLevels.NO2 = 8.0
	s.state.GasLevels.CO2 = 1.2
	s.evaluateAlerts()
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] Gas spike simulated!", id)
	common.WriteJSON(w, 200, map[string]string{"status": "spike triggered", "message": "dangerous gas levels injected"})
}

// handleSimulateFail makes all read endpoints return HTTP 503.
func (s *GasService) handleSimulateFail(w http.ResponseWriter, r *http.Request) {
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

// handleSimulateDegrade degrades sensor accuracy and adds latency.
// Body: {"noiseFactor": 5.0, "latencyMs": 100}
// noiseFactor>1 = noisier readings (lower accuracy). latencyMs = extra response delay.
func (s *GasService) handleSimulateDegrade(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	var body struct {
		NoiseFactor float64 `json:"noiseFactor"` // default 5.0
		LatencyMs   int     `json:"latencyMs"`   // default 100
	}
	body.NoiseFactor = 5.0
	body.LatencyMs = 100
	json.NewDecoder(r.Body).Decode(&body)
	if body.NoiseFactor <= 0 {
		body.NoiseFactor = 5.0
	}
	s.mu.Lock()
	s.degrade = simDegradeParams{active: true, noiseFactor: body.NoiseFactor, extraLatencyMs: body.LatencyMs}
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] DEGRADE MODE: noiseFactor=%.1f latencyMs=%d", id, body.NoiseFactor, body.LatencyMs)
	common.WriteJSON(w, 200, map[string]interface{}{
		"status": "degraded", "noiseFactor": body.NoiseFactor, "latencyMs": body.LatencyMs,
	})
}

// handleSimulateRecover clears failure and degradation modes.
func (s *GasService) handleSimulateRecover(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		common.WriteError(w, 405, "POST required")
		return
	}
	s.mu.Lock()
	s.simFail = false
	s.degrade = simDegradeParams{}
	id := s.state.ID
	s.mu.Unlock()
	log.Printf("[%s] RECOVERED – normal operation restored", id)
	common.WriteJSON(w, 200, map[string]string{"status": "recovered"})
}

func clamp(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
