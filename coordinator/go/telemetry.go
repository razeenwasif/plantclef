package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
    "path/filepath"
	"sync"
	"time"
)

type HWStats struct {
	Util  float64 `json:"util"`
	Temp  float64 `json:"temp"`
	Pwr   float64 `json:"pwr"`
	VramU float64 `json:"vram_u"`
	VramR float64 `json:"vram_r"`
	VramT float64 `json:"vram_t"`
}

type Heartbeat struct {
	Pod    string  `json:"pod"`
	Rank   int     `json:"rank"`
	Epoch  int     `json:"epoch"`
	Step   int     `json:"step"`
	Loss   float64 `json:"loss"`
	FPS    float64 `json:"fps"`
	Status string  `json:"status"`
	HW     HWStats `json:"hw"`
	Time   float64 `json:"time"`
}

type NodeStatus struct {
	LastPulse Heartbeat `json:"last_pulse"`
	LastSeen  time.Time `json:"last_seen"`
	Online    bool      `json:"online"`
}

var (
	clusterState = make(map[string]*NodeStatus)
	stateMutex   sync.RWMutex
)

func startPulsarReceiver() {
	addr, err := net.ResolveUDPAddr("udp", ":9001")
	if err != nil {
		fmt.Printf("[Pulsar] Error: %v\n", err)
		return
	}

	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		fmt.Printf("[Pulsar] Error: %v\n", err)
		return
	}
	defer conn.Close()

	fmt.Println("📡 [Pulsar] Receiver active on UDP :9001")
	buf := make([]byte, 2048)

	for {
		n, _, err := conn.ReadFromUDP(buf)
		if err != nil { continue }

		var hb Heartbeat
		if err := json.Unmarshal(buf[:n], &hb); err == nil {
			stateMutex.Lock()
			// Use Pod + Rank as unique key for multi-mission support
			key := fmt.Sprintf("%s_%d", hb.Pod, hb.Rank)
			clusterState[key] = &NodeStatus{
				LastPulse: hb,
				LastSeen:  time.Now(),
				Online:    true,
			}
			stateMutex.Unlock()
		}
	}
}

func startTelemetryAPI() {
	// 1. JSON API for Current State
	http.HandleFunc("/api/state", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		stateMutex.RLock()
		defer stateMutex.RUnlock()
		now := time.Now()
		for _, status := range clusterState {
			if now.Sub(status.LastSeen) > 30*time.Second { status.Online = false }
		}
		json.NewEncoder(w).Encode(clusterState)
	})

	// 2. Mission Control: Launch Commands
	http.HandleFunc("/api/launch", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" { return }
		
		var req struct {
			Command string `json:"command"`
			Phase   string `json:"phase"`
			Seed    int    `json:"seed"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		
		fmt.Printf("🎮 [Dashboard] Remote Launch Triggered: %s %s (Seed %d)\n", req.Command, req.Phase, req.Seed)
		
		// Execute the project's coordinator CLI. The path is configurable
		// via CLUSTER_CLI_PATH so this binary stays project-neutral.
		cliPath := getEnv("CLUSTER_CLI_PATH", "./coordinator.py")
		cmdStr := fmt.Sprintf("%s %s --phase %s --seed %d", cliPath, req.Command, req.Phase, req.Seed)
		go exec.Command("bash", "-c", cmdStr).Run()
		
		w.WriteHeader(200)
	})

	// 3. Registry & Logs API
	http.HandleFunc("/api/registry", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		http.ServeFile(w, r, "/workspace/PlantCLEF2026/models/registry.json")
	})

	http.HandleFunc("/api/logs", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		out, _ := exec.Command("tail", "-n", "100", "/workspace/PlantCLEF2026/reports/logs/rank_0.log").Output()
		w.Write(out)
	})

	// 4. Claude AI Crash Analysis
	http.HandleFunc("/api/claude/analyze", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			return
		}

		apiKey := os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			http.Error(w, `{"error":"ANTHROPIC_API_KEY not set"}`, 500)
			return
		}

		var req struct {
			Logs string `json:"logs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"bad request"}`, 400)
			return
		}

		systemPrompt := `You are an expert in distributed deep learning, PyTorch, CUDA, NCCL, and the PlantCLEF 2026 training pipeline.
You will be given crash logs from a training or inference run. Your job is to:
1. Identify the root cause in 1-2 sentences.
2. Provide the exact code fix (file path + line numbers if inferable, or the corrected snippet).
3. Add one sentence on how to prevent this class of error in future.
Be concise. Use markdown code blocks for fixes.`

		payload := map[string]interface{}{
			"model":      "claude-3-5-sonnet-20241022",
			"max_tokens": 1024,
			"system":     systemPrompt,
			"messages": []map[string]string{
				{"role": "user", "content": "Analyze this crash and provide the fix:\n\n" + req.Logs},
			},
		}

		body, _ := json.Marshal(payload)
		httpReq, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
		httpReq.Header.Set("x-api-key", apiKey)
		httpReq.Header.Set("anthropic-version", "2023-06-01")
		httpReq.Header.Set("content-type", "application/json")

		client := &http.Client{Timeout: 60 * time.Second}
		resp, err := client.Do(httpReq)
		if err != nil {
			http.Error(w, `{"error":"anthropic api unreachable"}`, 502)
			return
		}
		defer resp.Body.Close()

		var apiResp struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		respBody, _ := io.ReadAll(resp.Body)
		if err := json.Unmarshal(respBody, &apiResp); err != nil || len(apiResp.Content) == 0 {
			errMsg := "empty response"
			if apiResp.Error != nil {
				errMsg = apiResp.Error.Message
			}
			json.NewEncoder(w).Encode(map[string]string{"error": errMsg})
			return
		}

		json.NewEncoder(w).Encode(map[string]string{"analysis": apiResp.Content[0].Text})
	})

    // 5. Reports & Visualizations
	http.Handle("/api/reports/", http.StripPrefix("/api/reports/", http.FileServer(http.Dir("./reports"))))

    // 5. Static File Server for Dashboard
    cwd, _ := os.Getwd()
    distPath := filepath.Join(cwd, "dashboard/dist")
    
    if _, err := os.Stat(distPath); os.IsNotExist(err) {
        // Try one level up if running from orchestrator/
        distPath = filepath.Join(cwd, "..", "dashboard/dist")
    }
    
    // Final check to ensure index.html exists
    if _, err := os.Stat(filepath.Join(distPath, "index.html")); err != nil {
        fmt.Printf("⚠️  [Warning] Dashboard index.html not found at: %s\n", distPath)
    } else {
        fmt.Printf("📂 [Dashboard] Serving static files from: %s\n", distPath)
    }
    
	fs := http.FileServer(http.Dir(distPath))
	http.Handle("/", fs)

    apiPort := getEnv("CLUSTER_API_PORT", "9000")
	fmt.Printf("📊 [Telemetry] Mission Control API active on http://0.0.0.0:%s\n", apiPort)
	http.ListenAndServe("0.0.0.0:"+apiPort, nil)
}
