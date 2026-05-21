# Orchestrator

This directory contains the Go-based Control Plane for the Oracle distributed system.

It powers the backend of the **Neon Command Center**, utilizing Go's lightweight Goroutines and channels to manage cluster telemetry. It is responsible for:
- Maintaining a real-time heartbeat across all active training pods.
- Aggregating distributed metrics (Loss, FPS, Epoch progress) without blocking the Python training loop.
- Serving the WebSocket connections that feed the React frontend (`dashboard/`).