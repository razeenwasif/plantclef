package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
)

// --- CONFIGURATION ---
var (
	MasterAddr   = getMasterAddr()
	TotalWorkers = getTotalWorkers()
	
	// Default Inference Settings Optimized for RTX PRO 6000 Blackwell
	InfArgs = []string{
		"--batch-size", "128",
		"--num-workers", "12",
		"--aggregation", "bayesian_veg",
		"--checkpoint", "/workspace/working/PlantCLEF2026/models/dinov3_plantnet_v2_retrain/phase_a_best.pth",
		"--checkpoint", "/workspace/working/best.pt",
		"--submission-name", "ensemble_optimized.csv",
	}
)

func getMasterAddr() string {
	ip := getEnv("CLUSTER_MASTER_IP", "")
	if ip == "" {
		content, err := os.ReadFile("/workspace/PlantCLEF2026/coordinator/.master_ip")
		if err == nil {
			ip = strings.TrimSpace(string(content))
		}
	}
	if ip == "" {
		ip = "127.0.0.1"
	}
	return ip
}

func getTotalWorkers() int {
	if val := os.Getenv("CLUSTER_TOTAL_WORKERS"); val != "" {
		var res int
		fmt.Sscanf(val, "%d", &res)
		return res
	}
	if val := os.Getenv("CLUSTER_NNODES"); val != "" {
		var res int
		fmt.Sscanf(val, "%d", &res)
		if res > 0 {
			return res - 1
		}
	}
	return 0 // Default to local inference
}

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func startTelemetryServer() {
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		logPath := "/workspace/PlantCLEF2026/reports/logs/rank_0.log"
		if _, err := os.Stat(logPath); err == nil {
			out, err := exec.Command("tail", "-n", "10", logPath).Output()
			if err != nil {
				fmt.Fprintf(w, "inf_status{state=\"offline\"} 0\n")
				return
			}

			fmt.Fprintf(w, "inf_status{state=\"running\"} 1\n")

			lines := strings.Split(string(out), "\n")
			for i := len(lines) - 1; i >= 0; i-- {
				line := lines[i]
				if strings.Contains(line, "img/s") {
					if idx := strings.Index(line, "img/s"); idx != -1 {
						parts := strings.Fields(line[:idx])
						if len(parts) > 0 {
							fmt.Fprintf(w, "inf_speed_imgs %s\n", parts[len(parts)-1])
						}
					}
					if idx := strings.Index(line, "%|"); idx != -1 {
						progParts := strings.Fields(line[:idx])
						fmt.Fprintf(w, "inf_progress_percent %s\n", progParts[len(progParts)-1])
					}
					break
				}
			}
		} else {
			fmt.Fprintf(w, "inf_status{state=\"initializing\"} 0\n")
		}
	})

	fmt.Println("📊 [Telemetry] Inference Metrics active on :8080/metrics")
	go http.ListenAndServe(":8080", nil)
}

func main() {
	role := "sprint"
	if len(os.Args) > 1 {
		role = os.Args[1]
	}

	fmt.Println("🛡️  COORDINATOR: Initializing Inference Orchestrator...")
	startTelemetryServer()

	for {
		if role == "master" {
			runMaster()
		} else if role == "worker" {
			rank := getEnv("CLUSTER_NODE_RANK", "1")
			runWorker(rank)
		} else {
			fmt.Println("🚀 [Sprint] Launching local standalone inference...")
			launchInference("sprint")
			monitorLoop("sprint", nil)
			break // Sprint runs once and exits
		}
		fmt.Println("⚠️ [Hub] Cluster reset. Re-initializing in 5s...")
		time.Sleep(5 * time.Second)
	}
}

func runMaster() {
	fmt.Printf("🧠 [Master] Inference Hub active on :9999 (External: %s)\n", MasterAddr)
	lc := net.ListenConfig{KeepAlive: 30 * time.Second}
	ln, err := lc.Listen(context.Background(), "tcp", ":9999")
	if err != nil {
		fmt.Printf("[Error] Hub Port Blocked: %v\n", err)
		return
	}
	defer ln.Close()

	conns := make([]net.Conn, 0)
	for len(conns) < TotalWorkers {
		conn, err := ln.Accept()
		if err != nil {
			continue
		}
		if tcpConn, ok := conn.(*net.TCPConn); ok {
			tcpConn.SetKeepAlive(true)
			tcpConn.SetKeepAlivePeriod(30 * time.Second)
		}
		conns = append(conns, conn)
		fmt.Printf("[Hub] Worker %d/%d connected from %s\n", len(conns), TotalWorkers, conn.RemoteAddr())
	}

	fmt.Println("🚀 CLUSTER SYNCED. Sending NUKE + GO signals...")
	launchInference("master")
	time.Sleep(5 * time.Second)

	for _, conn := range conns {
		conn.Write([]byte("NK"))
		time.Sleep(1 * time.Second)
		conn.Write([]byte("GO"))
	}

	monitorLoop("master", conns)
}

func runWorker(rank string) {
	fullAddr := MasterAddr + ":9999"
	fmt.Printf("📡 [Worker %s] Connecting to Master Hub at %s...\n", rank, fullAddr)
	var conn net.Conn
	var err error

	d := net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	for {
		conn, err = d.DialContext(context.Background(), "tcp", fullAddr)
		if err == nil {
			break
		}
		time.Sleep(2 * time.Second)
	}

	if tcpConn, ok := conn.(*net.TCPConn); ok {
		tcpConn.SetKeepAlive(true)
		tcpConn.SetKeepAlivePeriod(30 * time.Second)
	}

	fmt.Printf("[Worker %s] Connected. Waiting for commands...\n", rank)
	buf := make([]byte, 2)

	_, err = conn.Read(buf)
	if err != nil {
		return
	}
	if string(buf) == "NK" {
		fmt.Printf("☢️ [Worker %s] NUKE RECEIVED. Sanitizing local pod...\n", rank)
		exec.Command("bash", "-c", "pkill -9 -f torchrun; pkill -9 -f run_inference; fuser -k 29505/tcp").Run()
	}

	_, err = conn.Read(buf)
	if err != nil {
		return
	}
	fmt.Printf("🏁 [Worker %s] GO RECEIVED. Launching compute plane.\n", rank)
	launchInference("worker")

	go func() {
		hbBuf := make([]byte, 2)
		for {
			_, err := conn.Read(hbBuf)
			if err != nil {
				fmt.Printf("⚠️ [Worker %s] Hub connection lost. Signaling reset.\n", rank)
				break
			}
		}
		exec.Command("bash", "-c", "pkill -9 -f torchrun; pkill -9 -f run_inference").Run()
	}()

	monitorLoop("worker", nil)
}

func launchInference(role string) {
	exec.Command("bash", "-c", "pkill -9 -f torchrun; pkill -9 -f run_inference; fuser -k 29505/tcp").Run()
	time.Sleep(2 * time.Second)

	rankVar := os.Getenv("CLUSTER_NODE_RANK")
	if rankVar == "" {
		if role == "master" {
			rankVar = "0"
		} else {
			rankVar = "1"
		}
	}

	envCmd := fmt.Sprintf("export CLUSTER_NODE_RANK=%s && ", rankVar) +
		fmt.Sprintf("export CLUSTER_MASTER_IP=%s && ", MasterAddr) +
		"export PYTHONPATH=$PYTHONPATH:/workspace/PlantCLEF2026:/workspace/PlantCLEF2026/src && " +
		"export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && export NCCL_DEBUG=INFO"

	// Hand off to the project's inference CLI. The path is configurable via
	// CLUSTER_CLI_PATH so this binary doesn't bake in any one project's
	// entry-point name.
	cliPath := getEnv("CLUSTER_CLI_PATH", "./coordinator.py")
	cmdStr := fmt.Sprintf("%s && %s infer --ensemble --role %s", envCmd, cliPath, role)

	cmd := exec.Command("bash", "-c", cmdStr)
	cmd.Dir = "/workspace/PlantCLEF2026"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Start()
}

func monitorLoop(role string, conns []net.Conn) {
	fmt.Printf("🛡️  [Watchdog] Monitoring inference process...\n")
	time.Sleep(10 * time.Second)
	for {
		if !isProcessActive() {
			fmt.Println("🛑 Inference process exited.")
			return
		}
		if role == "master" && len(conns) > 0 {
			for _, c := range conns {
				_, err := c.Write([]byte("HB"))
				if err != nil {
					fmt.Println("[Hub] Worker connection dropped. Triggering global restart.")
					for _, closeC := range conns {
						closeC.Close()
					}
					return
				}
			}
		}
		time.Sleep(10 * time.Second)
	}
}

func isProcessActive() bool {
	out, _ := exec.Command("bash", "-c", "pgrep -f run_inference").Output()
	return len(out) > 0
}
