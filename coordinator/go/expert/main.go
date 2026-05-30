package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
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
    if value, ok := os.LookupEnv(key); ok {
        return value
    }
    return fallback
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: ./expert_launcher <phase> [extra_args...]")
		return
	}
	phase := os.Args[1]
	extraArgs := os.Args[2:]

	fmt.Println("🚀 [Expert] Launching standalone expert trainer...")
	
	// --- PRE-SYNC NUKE ---
	exec.Command("bash", "-c", "pkill -9 -f torchrun; pkill -9 -f run_phase; pkill -9 -f run_inference; fuser -k 29505/tcp").Run()
	
	// Direct launch without network hub
	launchTraining(phase, "sprint", extraArgs)
}

func launchTraining(phase, role string, extraArgs []string) {
	rankVar := os.Getenv("CLUSTER_NODE_RANK")
	if rankVar == "" { rankVar = "0" }
	
	envCmd := fmt.Sprintf("source /workspace/pytorch_env/bin/activate && export CLUSTER_NODE_RANK=%s && ", rankVar) +
			  "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && " +
			  "export NCCL_DEBUG=WARN"

	launchPath := "/workspace/PlantCLEF2026/coordinator/launch.sh"
	argsStr := strings.Join(extraArgs, " ")
	cmdStr := fmt.Sprintf("%s && %s %s %s %s", envCmd, launchPath, phase, role, argsStr)

	cmd := exec.Command("bash", "-c", cmdStr)
	cmd.Dir = "/workspace/PlantCLEF2026"
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()
}
