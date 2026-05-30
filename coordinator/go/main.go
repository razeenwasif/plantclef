package main

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"strings"
	"time"
)

// --- CONFIGURATION ---
var (
	MasterAddr   = getMasterAddr()
	TotalWorkers = getTotalWorkers()
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
	return ip + ":9999"
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
	return 0 // Default to single-node sprint
}

func getEnv(key, fallback string) string {
    if value, ok := os.LookupEnv(key); ok { return value }
    return fallback
}

func getEnvInt(key string, fallback int) int {
    if value, ok := os.LookupEnv(key); ok {
        var result int
        fmt.Sscanf(value, "%d", &result)
        return result
    }
    return fallback
}

func getLocalIP() string {
	addrs, _ := net.InterfaceAddrs()
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipv4 := ipnet.IP.To4(); ipv4 != nil {
				if strings.HasPrefix(ipv4.String(), "10.") { return ipv4.String() }
			}
		}
	}
	return "127.0.0.1"
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: ./coordinator <phase|hub> [extra_args...]")
		return
	}
	phase := os.Args[1]
	extraArgs := os.Args[2:]

	localIP := getLocalIP()
	masterFile := "/workspace/PlantCLEF2026/coordinator/.master_ip"
	masterIP := getEnv("CLUSTER_MASTER_IP", "")
	isMaster := false

	// --- 1. Role Identification ---
	if phase == "hub" {
		isMaster = true
		masterIP = localIP
		fmt.Println("🛰️  COORDINATOR-HUB: Initializing Always-On Control Plane...")
	} else if phase != "sprint" {
		if masterIP != "" {
			isMaster = (localIP == masterIP)
		} else {
			content, err := os.ReadFile(masterFile)
			if err != nil {
				os.WriteFile(masterFile, []byte(localIP), 0644)
				isMaster = true
				masterIP = localIP
			} else {
				masterIP = strings.TrimSpace(string(content))
				if localIP == masterIP { isMaster = true }
			}
		}
	}

	if masterIP != "" {
		MasterAddr = masterIP + ":9999"
		os.Setenv("CLUSTER_MASTER_IP", masterIP)
	}

	// --- 2. Telemetry Start ---
	go startPulsarReceiver()
	go startTelemetryAPI()

	if phase == "hub" {
		fmt.Printf("✅ COORDINATOR HUB ACTIVE. Dashboard: http://%s:8080\n", localIP)
		fmt.Println("[*] Standing by for remote mission dispatch...")
		select {} // Block forever
	}

	// --- 3. Execution Loop ---
	if isMaster {
		fmt.Println("🛡️  COORDINATOR: [Master Hub] Initializing Control Plane...")
	} else {
		rank := os.Getenv("CLUSTER_NODE_RANK")
		fmt.Printf("🛡️  COORDINATOR: [Worker %s] Initializing Compute Plane...\n", rank)
	}

	for {
		// Clean start
		exec.Command("bash", "-c", "pkill -9 -f torchrun; pkill -9 -f run_phase; pkill -9 -f run_inference; fuser -k 29505/tcp").Run()
		
		if phase == "sprint" {
			launchTraining(phase, "sprint", extraArgs)
			monitorLoop("sprint", nil)
			os.Exit(0)
		} else if isMaster {
			runMaster(phase, extraArgs)
		} else {
			rank := os.Getenv("CLUSTER_NODE_RANK")
			runWorker(phase, rank, extraArgs)
		}
		time.Sleep(5 * time.Second)
	}
}

func runMaster(phase string, extraArgs []string) {
	fmt.Printf("🧠 [Master] Hub active on :9999 (External: %s)\n", MasterAddr)
	ln, err := net.Listen("tcp", ":9999")
	if err != nil { return }
	defer ln.Close()

	conns := make([]net.Conn, 0)
	for len(conns) < TotalWorkers {
		conn, err := ln.Accept()
		if err != nil { continue }
		conns = append(conns, conn)
		fmt.Printf("[Hub] Worker %d/%d connected from %s\n", len(conns), TotalWorkers, conn.RemoteAddr())
	}

	fmt.Println("🚀 CLUSTER SYNCED. Sending NUKE + GO signals...")
	launchTraining(phase, "master", extraArgs)
	for _, conn := range conns {
		conn.Write([]byte("NK"))
		time.Sleep(500 * time.Millisecond)
		conn.Write([]byte("GO"))
	}
	monitorLoop("master", conns)
}

func runWorker(phase, rank string, extraArgs []string) {
	fmt.Printf("📡 [Worker %s] Connecting to Master Hub at %s...\n", rank, MasterAddr)
	var conn net.Conn
	for {
		var err error
		conn, err = net.Dial("tcp", MasterAddr)
		if err == nil { break }
		time.Sleep(2 * time.Second)
	}
	fmt.Printf("[Worker %s] Connected. Waiting for commands...\n", rank)
	buf := make([]byte, 2)
	conn.Read(buf) // NK
	conn.Read(buf) // GO
	launchTraining(phase, "worker", extraArgs)
	monitorLoop("worker", nil)
}

func launchTraining(phase, role string, extraArgs []string) {
	argsStr := strings.Join(extraArgs, " ")
	cmdStr := fmt.Sprintf("./coordinator/scripts/launch.sh %s %s %s", phase, role, argsStr)
	cmd := exec.Command("bash", "-c", cmdStr)
	cmd.Dir = "/workspace/PlantCLEF2026"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Start()
}

func monitorLoop(role string, conns []net.Conn) {
	time.Sleep(10 * time.Second)
	for {
		if !isTrainingActive() { return }
		if role == "master" {
			for _, c := range conns { c.Write([]byte("HB")) }
		}
		time.Sleep(10 * time.Second)
	}
}

func isTrainingActive() bool {
	cmd := exec.Command("bash", "-c", "pgrep -f torchrun || pgrep -f run_inference")
	out, _ := cmd.Output()
	return len(out) > 0
}
