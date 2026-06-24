package api

import (
	"net/http"
	"time"

	"github.com/dicai/dicai-admin/pkg/coordinator"
	"github.com/gin-gonic/gin"
)

// Server handles HTTP API requests
type Server struct {
	router     *gin.Engine
	registry   *coordinator.ProviderRegistry
	assignment *coordinator.AssignmentEngine
	health     *coordinator.HealthChecker
}

// NewServer creates a new API server
func NewServer(registry *coordinator.ProviderRegistry) *Server {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())

	server := &Server{
		router:     router,
		registry:   registry,
		assignment: coordinator.NewAssignmentEngine(registry),
		health:     coordinator.NewHealthChecker(registry),
	}

	server.setupRoutes()
	server.health.Start()

	return server
}

func (s *Server) setupRoutes() {
	// Health check
	s.router.GET("/health", s.handleHealth)

	// Provider management
	s.router.POST("/api/v1/providers/register", s.handleRegisterProvider)
	s.router.POST("/api/v1/providers/heartbeat", s.handleHeartbeat)
	s.router.GET("/api/v1/providers", s.handleListProviders)
	s.router.GET("/api/v1/providers/:id", s.handleGetProvider)
	s.router.DELETE("/api/v1/providers/:id", s.handleUnregisterProvider)

	// Model deployment
	s.router.POST("/api/v1/models/:id/deploy", s.handleDeployModel)
	s.router.GET("/api/v1/models/:id/assignments", s.handleGetAssignments)

	// Inference
	s.router.POST("/api/v1/inference", s.handleInference)

	// OpenAI-compatible API
	s.router.POST("/v1/chat/completions", s.handleChatCompletion)
	s.router.GET("/v1/models", s.handleListModels)
}

// Run starts the HTTP server
func (s *Server) Run(addr string) error {
	return s.router.Run(addr)
}

// handleHealth returns admin health status
func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "ok",
		"timestamp": time.Now().Unix(),
	})
}

// ProviderRegistrationRequest is the request body for registering a provider
type ProviderRegistrationRequest struct {
	ID           string            `json:"id" binding:"required"`
	Address      string            `json:"address"`
	Port         int               `json:"port"`
	Memory       int               `json:"memory"`
	Backend      string            `json:"backend"`
	Capabilities map[string]string `json:"capabilities"`
}

func (s *Server) handleRegisterProvider(c *gin.Context) {
	var req ProviderRegistrationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	provider := &coordinator.Provider{
		ID:           req.ID,
		Address:      req.Address,
		Port:         req.Port,
		Memory:       req.Memory,
		Backend:      req.Backend,
		Capabilities: req.Capabilities,
	}

	if err := s.registry.Register(provider); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":     provider.ID,
		"status": "registered",
	})
}

func (s *Server) handleHeartbeat(c *gin.Context) {
	var req struct {
		ID string `json:"id" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if !s.registry.UpdateHeartbeat(req.ID) {
		c.JSON(http.StatusNotFound, gin.H{"error": "provider not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func (s *Server) handleListProviders(c *gin.Context) {
	providers := s.registry.List()
	c.JSON(http.StatusOK, providers)
}

func (s *Server) handleGetProvider(c *gin.Context) {
	id := c.Param("id")
	provider, ok := s.registry.Get(id)
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "provider not found"})
		return
	}
	c.JSON(http.StatusOK, provider)
}

func (s *Server) handleUnregisterProvider(c *gin.Context) {
	id := c.Param("id")
	s.registry.Unregister(id)
	c.JSON(http.StatusOK, gin.H{"status": "unregistered"})
}

// DeployModelRequest is the request body for deploying a model
type DeployModelRequest struct {
	TotalLayers int `json:"total_layers"`
}

func (s *Server) handleDeployModel(c *gin.Context) {
	modelID := c.Param("id")

	var req DeployModelRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		// Default to 80 layers if not specified
		req.TotalLayers = 80
	}

	assignments, err := s.assignment.UniformAssignment(modelID, req.TotalLayers)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"model_id":    modelID,
		"assignments": assignments,
	})
}

func (s *Server) handleGetAssignments(c *gin.Context) {
	modelID := c.Param("id")
	assignments, ok := s.registry.GetAssignments(modelID)
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found"})
		return
	}
	c.JSON(http.StatusOK, assignments)
}

// InferenceRequest is the request body for inference
type InferenceRequest struct {
	Model   string `json:"model" binding:"required"`
	Prompt  string `json:"prompt" binding:"required"`
	MaxTokens int  `json:"max_tokens"`
}

func (s *Server) handleInference(c *gin.Context) {
	var req InferenceRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TODO: Route to pipeline
	// For now, return mock response
	c.JSON(http.StatusOK, gin.H{
		"model":    req.Model,
		"prompt":   req.Prompt,
		"response": "Hello! This is a mock response from the distributed inference network.",
		"latency":  "150ms",
		"tokens":   12,
	})
}

// ChatCompletionRequest matches OpenAI's API format
type ChatCompletionRequest struct {
	Model    string    `json:"model" binding:"required"`
	Messages []Message `json:"messages" binding:"required"`
	MaxTokens int     `json:"max_tokens"`
	Stream   bool     `json:"stream"`
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse matches OpenAI's API format
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (s *Server) handleChatCompletion(c *gin.Context) {
	var req ChatCompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TODO: Route to pipeline
	// For now, return mock response
	response := ChatCompletionResponse{
		ID:      "dicai-" + time.Now().Format("20060102-150405"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: "Hello! This is a mock response from DiCAI distributed inference network.",
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     len(req.Messages),
			CompletionTokens: 12,
			TotalTokens:      len(req.Messages) + 12,
		},
	}

	c.JSON(http.StatusOK, response)
}

func (s *Server) handleListModels(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data": []gin.H{
			{
				"id":       "llama-3-70b",
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "dicai",
			},
		},
	})
}
