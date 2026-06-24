package blockchain

// Blockchain is the interface for blockchain interactions
type Blockchain interface {
	RecordTask(providerID string, tokens int) error
	VerifyProvider(providerID string) (bool, error)
}

// Simulator is a simulated blockchain for POC
type Simulator struct{}

// NewSimulator creates a new blockchain simulator
func NewSimulator() *Simulator {
	return &Simulator{}
}

// RecordTask simulates recording a task on the blockchain
func (s *Simulator) RecordTask(providerID string, tokens int) error {
	return nil
}

// VerifyProvider simulates verifying a provider
func (s *Simulator) VerifyProvider(providerID string) (bool, error) {
	return true, nil
}
