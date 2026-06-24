package api

import (
	"net/http"
	"time"

	"github.com/dicai/dicai-admin/pkg/blockchain"
	"github.com/dicai/dicai-admin/pkg/coordinator"
	"github.com/dicai/dicai-admin/pkg/metrics"
	"github.com/gin-gonic/gin"
)

// Handler holds the API handlers
type Handler struct {
	coord   *coordinator.ProviderRegistry
	bc      blockchain.Blockchain
	metrics *metrics.Registry
}

// NewHandler creates a new API handler
func NewHandler(coord *coordinator.ProviderRegistry, bc blockchain.Blockchain, m *metrics.Registry) *Handler {
	return &Handler{
		coord:   coord,
		bc:      bc,
		metrics: m,
	}
}

// RegisterRoutes registers all API routes
func (h *Handler) RegisterRoutes(r *gin.Engine) {
	r.GET("/health", h.healthCheck)
	r.GET("/v1/providers", h.listProviders)
	r.POST("/v1/providers/register", h.registerProvider)
	r.POST("/v1/chat/completions", h.chatCompletions)
	r.POST("/v1/completions", h.completions)
}

func (h *Handler) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "healthy"})
}

func (h *Handler) listProviders(c *gin.Context) {
	providers := h.coord.List()
	c.JSON(http.StatusOK, gin.H{"providers": providers})
}

func (h *Handler) registerProvider(c *gin.Context) {
	var p coordinator.Provider
	if err := c.ShouldBindJSON(&p); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := h.coord.Register(&p); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "registered", "id": p.ID})
}

func (h *Handler) chatCompletions(c *gin.Context) {
	var req ChatCompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Simulate distributed inference response
	resp := ChatCompletionResponse{
		ID:      "chatcmpl-dicai-" + randomString(8),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: "This is a simulated distributed inference response from DiCAI.",
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     countTokens(req.Messages),
			CompletionTokens: 12,
			TotalTokens:      countTokens(req.Messages) + 12,
		},
	}
	c.JSON(http.StatusOK, resp)
}

func (h *Handler) completions(c *gin.Context) {
	var req CompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	resp := CompletionResponse{
		ID:      "cmpl-dicai-" + randomString(8),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []TextChoice{
			{
				Index:        0,
				Text:         "This is a simulated completion from DiCAI.",
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     len(req.Prompt) / 4,
			CompletionTokens: 10,
			TotalTokens:      len(req.Prompt)/4 + 10,
		},
	}
	c.JSON(http.StatusOK, resp)
}
