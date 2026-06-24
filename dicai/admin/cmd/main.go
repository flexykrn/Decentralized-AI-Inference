package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/dicai/dicai-admin/pkg/api"
	"github.com/dicai/dicai-admin/pkg/coordinator"
)

func main() {
	var (
		addr   = flag.String("addr", ":8080", "HTTP server address")
		apiURL = flag.String("api-url", "http://localhost:8080", "Admin API URL for providers")
	)
	flag.Parse()

	log.Println("Starting DiCAI Admin Coordinator...")
	log.Printf("API URL: %s", *apiURL)
	log.Printf("HTTP server: %s", *addr)

	// Create registry
	registry := coordinator.NewProviderRegistry()

	// Create API server
	server := api.NewServer(registry)

	// Handle graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		log.Println("Shutting down...")
		os.Exit(0)
	}()

	// Start HTTP server
	log.Printf("Admin coordinator ready at http://localhost%s", *addr)
	if err := server.Run(*addr); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
