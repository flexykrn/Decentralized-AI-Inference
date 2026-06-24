package coordinator

import (
	"fmt"
	"sort"
)

// AssignmentEngine distributes model layers across providers
type AssignmentEngine struct {
	registry *ProviderRegistry
}

// NewAssignmentEngine creates a new assignment engine
func NewAssignmentEngine(registry *ProviderRegistry) *AssignmentEngine {
	return &AssignmentEngine{registry: registry}
}

// UniformAssignment distributes layers evenly across all healthy providers
func (e *AssignmentEngine) UniformAssignment(modelID string, totalLayers int) ([]LayerAssignment, error) {
	providers := e.registry.GetHealthy()
	if len(providers) == 0 {
		return nil, fmt.Errorf("no healthy providers available")
	}
	
	numProviders := len(providers)
	layersPerProvider := totalLayers / numProviders
	remainder := totalLayers % numProviders
	
	assignments := make([]LayerAssignment, 0, numProviders)
	currentLayer := 0
	
	for i, provider := range providers {
		// Distribute remainder across first N providers
		extra := 0
		if i < remainder {
			extra = 1
		}
		
		layerStart := currentLayer
		layerEnd := currentLayer + layersPerProvider + extra - 1
		
		assignment := LayerAssignment{
			ProviderID: provider.ID,
			LayerStart:   layerStart,
			LayerEnd:     layerEnd,
			ModelID:      modelID,
		}
		assignments = append(assignments, assignment)
		
		// Update provider's assigned layers
		provider.Layers = make([]int, 0, layerEnd-layerStart+1)
		for j := layerStart; j <= layerEnd; j++ {
			provider.Layers = append(provider.Layers, j)
		}
		
		currentLayer = layerEnd + 1
	}
	
	// Store assignments
	e.registry.StoreAssignments(modelID, assignments)
	
	return assignments, nil
}

// GetPipelineOrder returns providers in layer order for inference
func (e *AssignmentEngine) GetPipelineOrder(modelID string) ([]*Provider, error) {
	assignments, ok := e.registry.GetAssignments(modelID)
	if !ok {
		return nil, fmt.Errorf("no assignments found for model %s", modelID)
	}
	
	// Sort by layer start
	sort.Slice(assignments, func(i, j int) bool {
		return assignments[i].LayerStart < assignments[j].LayerStart
	})
	
	// Get providers in order
	pipeline := make([]*Provider, 0, len(assignments))
	for _, assignment := range assignments {
		provider, ok := e.registry.Get(assignment.ProviderID)
		if !ok {
			return nil, fmt.Errorf("provider %s not found", assignment.ProviderID)
		}
		pipeline = append(pipeline, provider)
	}
	
	return pipeline, nil
}

// ValidateCoverage checks if all layers are covered by assignments
func (e *AssignmentEngine) ValidateCoverage(modelID string, totalLayers int) error {
	assignments, ok := e.registry.GetAssignments(modelID)
	if !ok {
		return fmt.Errorf("no assignments found for model %s", modelID)
	}
	
	// Check for gaps
	expectedStart := 0
	for _, assignment := range assignments {
		if assignment.LayerStart != expectedStart {
			return fmt.Errorf("layer gap detected: expected %d, got %d", expectedStart, assignment.LayerStart)
		}
		expectedStart = assignment.LayerEnd + 1
	}
	
	if expectedStart != totalLayers {
		return fmt.Errorf("incomplete coverage: expected %d layers, got %d", totalLayers, expectedStart)
	}
	
	return nil
}
