package coordinator

// AssignmentEngine assigns model layers to providers
type AssignmentEngine struct {
	registry *ProviderRegistry
}

// NewAssignmentEngine creates a new assignment engine
func NewAssignmentEngine(registry *ProviderRegistry) *AssignmentEngine {
	return &AssignmentEngine{registry: registry}
}

// UniformAssignment distributes layers uniformly across providers
func (e *AssignmentEngine) UniformAssignment(modelID string, totalLayers int) ([]LayerAssignment, error) {
	providers := e.registry.GetHealthy()
	if len(providers) == 0 {
		return nil, nil
	}

	layersPerProvider := totalLayers / len(providers)
	remainder := totalLayers % len(providers)
	assignments := make([]LayerAssignment, 0, len(providers))
	currentLayer := 0

	for i, p := range providers {
		count := layersPerProvider
		if i < remainder {
			count++
		}
		if count == 0 {
			continue
		}
		start := currentLayer
		end := currentLayer + count - 1
		assignments = append(assignments, LayerAssignment{
			ProviderID: p.ID,
			LayerStart: start,
			LayerEnd:   end,
			ModelID:    modelID,
		})
		p.Layers = []int{start, end}
		currentLayer = end + 1
	}

	e.registry.StoreAssignments(modelID, assignments)
	return assignments, nil
}
