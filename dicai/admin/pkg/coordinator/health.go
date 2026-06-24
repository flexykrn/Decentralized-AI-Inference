package coordinator

import (
	"time"
)

// HealthChecker monitors provider health and handles dropouts
type HealthChecker struct {
	registry      *ProviderRegistry
	checkInterval time.Duration
	timeout       time.Duration
	stopCh        chan struct{}
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(registry *ProviderRegistry) *HealthChecker {
	return &HealthChecker{
		registry:      registry,
		checkInterval: 5 * time.Second,
		timeout:       15 * time.Second,
		stopCh:        make(chan struct{}),
	}
}

// Start begins the health check loop
func (h *HealthChecker) Start() {
	go h.loop()
}

// Stop halts the health check loop
func (h *HealthChecker) Stop() {
	close(h.stopCh)
}

func (h *HealthChecker) loop() {
	ticker := time.NewTicker(h.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			h.checkAll()
		case <-h.stopCh:
			return
		}
	}
}

func (h *HealthChecker) checkAll() {
	providers := h.registry.List()
	now := time.Now()

	for _, provider := range providers {
		elapsed := now.Sub(provider.LastHeartbeat)

		if elapsed > h.timeout*4 {
			// Offline: no heartbeat for 60+ seconds
			provider.Status = "offline"
		} else if elapsed > h.timeout {
			// Unhealthy: no heartbeat for 15+ seconds
			provider.Status = "unhealthy"
		}
		// else: healthy (heartbeat within 15 seconds)
	}
}

// GetHealthSummary returns counts of providers by status
func (h *HealthChecker) GetHealthSummary() map[string]int {
	providers := h.registry.List()
	summary := map[string]int{
		"healthy":    0,
		"unhealthy":  0,
		"offline":    0,
		"total":      len(providers),
	}

	for _, p := range providers {
		summary[p.Status]++
	}

	return summary
}

// IsHealthy returns true if provider is responding to heartbeats
func (h *HealthChecker) IsHealthy(id string) bool {
	provider, ok := h.registry.Get(id)
	if !ok {
		return false
	}
	return provider.Status == "healthy"
}
