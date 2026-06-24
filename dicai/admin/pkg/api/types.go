package api

import (
	"math/rand"
	"time"
)

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest mirrors the OpenAI chat completions request
type ChatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
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

// ChatCompletionResponse mirrors the OpenAI chat completions response
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// CompletionRequest mirrors the OpenAI completions request
type CompletionRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream,omitempty"`
}

// TextChoice represents a text completion choice
type TextChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// CompletionResponse mirrors the OpenAI completions response
type CompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []TextChoice `json:"choices"`
	Usage   Usage        `json:"usage"`
}

func randomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

func countTokens(messages []Message) int {
	// Very rough token estimation
	total := 0
	for _, m := range messages {
		total += len(m.Content) / 4
	}
	return total
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
