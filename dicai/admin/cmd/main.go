package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/dicai/dicai-admin/pkg/api"
	"github.com/dicai/dicai-admin/pkg/blockchain"
	"github.com/dicai/dicai-admin/pkg/coordinator"
	"github.com/dicai/dicai-admin/pkg/metrics"
	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

func main() {
	port := flag.String("port", "8080", "Admin API port")
	metricsPort := flag.String("metrics-port", "9090", "Metrics port")
	logLevel := flag.String("log-level", "info", "Log level")
	flag.Parse()

	// Setup logging
	level, err := logrus.ParseLevel(*logLevel)
	if err != nil {
		logrus.SetLevel(logrus.InfoLevel)
	} else {
		logrus.SetLevel(level)
	}
	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
	})

	logrus.Info("Starting DiCAI Admin Coordinator")
	logrus.Infof("Version: 0.1.0, Port: %s, Metrics: %s", *port, *metricsPort)

	// Initialize components
	coord := coordinator.New()
	bc := blockchain.NewSimulator() // Simulated for POC
	metrics := metrics.New()

	// Setup API server
	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(loggingMiddleware())

	apiHandler := api.NewHandler(coord, bc, metrics)
	apiHandler.RegisterRoutes(router)

	// Start metrics server
	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", metrics.Handler())
		logrus.Infof("Metrics server starting on :%s", *metricsPort)
		if err := http.ListenAndServe(":"+*metricsPort, mux); err != nil {
			logrus.Errorf("Metrics server failed: %v", err)
		}
	}()

	// Start main API server
	logrus.Infof("Admin API starting on :%s", *port)
	if err := router.Run(":" + *port); err != nil {
		logrus.Fatalf("Admin API failed: %v", err)
	}
}

func loggingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path

		c.Next()

		latency := time.Since(start)
		status := c.Writer.Status()

		logrus.WithFields(logrus.Fields{
			"status":  status,
			"latency": latency,
			"path":    path,
			"method":  c.Request.Method,
		}).Info("Request")
	}
}
