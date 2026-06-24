package coordinator

import (
	"sync"
	"time"
)

// Provider represents a compute node in the distributed network
type Provider struct {
	ID            string            `json:"id"`
	Address       string            `json:"address"`
	Port          int               `json:"port"`
	Memory        int               `json:"memory"`        // GB
	Backend       string            `json:"backend"`       // mlx, cuda, directml, cpu
	Layers        []int             `json:"layers"`        // assigned layer indices
	Latency       float64           `json:"latency"`       // ms
	Status        string            `json:"status"`        // healthy, unhealthy, offline
	LastHeartbeat time.Time         `json:"last_heartbeat"`
	Capabilities  map[string]string `json:"capabilities"`
}

// LayerAssignment maps a layer range to a provider
type LayerAssignment struct {
	ProviderID  string `json:"provider_id"`
	LayerStart  int    `json:"layer_start"`
	LayerEnd    int    `json:"layer_end"`
	ModelID     string `json:"model_id"`
}

// ProviderRegistry manages all registered providers
type ProviderRegistry struct {
	mu        sync.RWMutex
	providers map[string]*Provider
	assignments map[string][]LayerAssignment // model_id -> assignments
}

// NewProviderRegistry creates a new registry
func NewProviderRegistry() *ProviderRegistry {
	return &ProviderRegistry{
		providers:   make(map[string]*Provider),
		assignments: make(map[string][]LayerAssignment),
	}
}

// Register adds a new provider to the registry
func (r *ProviderRegistry) Register(p *Provider) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	p.Status = "healthy"
	p.LastHeartbeat = time.Now()
	r.providers[p.ID] = p
	return nil
}

// Unregister removes a provider from the registry
func (r *ProviderRegistry) Unregister(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.providers, id)
}

// Get retrieves a provider by ID
func (r *ProviderRegistry) Get(id string) (*Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.providers[id]
	return p, ok
}

// List returns all registered providers
func (r *ProviderRegistry) List() []*Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	providers := make([]*Provider, 0, len(r.providers))
	for _, p := range r.providers {
		providers = append(providers, p)
	}
	return providers
}

// UpdateHeartbeat updates the last heartbeat time for a provider
func (r *ProviderRegistry) UpdateHeartbeat(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	p, ok := r.providers[id]
	if !ok {
		return false
	}
	p.LastHeartbeat = time.Now()
	p.Status = "healthy"
	return true
}

// GetHealthy returns all providers with healthy status
func (r *ProviderRegistry) GetHealthy() []*Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	healthy := make([]*Provider, 0)
	for _, p := range r.providers {
		if p.Status == "healthy" {
			healthy = append(healthy, p)
		}
	}
	return healthy
}

// StoreAssignments saves layer assignments for a model
func (r *ProviderRegistry) StoreAssignments(modelID string, assignments []LayerAssignment) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.assignments[modelID] = assignments
}

// GetAssignments retrieves layer assignments for a model
func (r *ProviderRegistry) GetAssignments(modelID string) ([]LayerAssignment, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	assignments, ok := r.assignments[modelID]
	return assignments, ok
}
