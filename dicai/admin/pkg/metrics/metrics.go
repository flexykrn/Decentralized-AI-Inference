package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Registry holds prometheus metrics
type Registry struct {
	requestsTotal   *prometheus.CounterVec
	requestDuration *prometheus.HistogramVec
}

// New creates a new metrics registry
func New() *Registry {
	r := &Registry{
		requestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "dicai_requests_total",
				Help: "Total number of requests",
			},
			[]string{"method", "status"},
		),
		requestDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "dicai_request_duration_seconds",
				Help:    "Request duration in seconds",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method"},
		),
	}
	prometheus.MustRegister(r.requestsTotal)
	prometheus.MustRegister(r.requestDuration)
	return r
}

// Handler returns the metrics HTTP handler
func (r *Registry) Handler() http.Handler {
	return promhttp.Handler()
}
