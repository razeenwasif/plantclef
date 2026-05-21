import socket
import json
import os
import time
import subprocess

class PulsarHeartbeat:
    """
    Ultra-Low Latency Telemetry Emitter.
    Broadcasts real-time training metrics to the Go Orchestrator via UDP.
    """
    def __init__(self, port: int = 9001):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        master_ip = os.environ.get("ORACLE_MASTER_IP", "127.0.0.1")
        self.target = (master_ip, port)
        self.pod_name = os.environ.get("ORACLE_NAME", "unknown_pod")
        self.rank = int(os.environ.get("RANK", 0))

    def pulse(self, epoch: int, step: int, loss: float, throughput: float, status: str = "active"):
        """Emits a single heartbeat packet with hardware vitals."""
        import torch
        
        gpu_util = 0
        gpu_temp = 0
        gpu_pwr = 0
        vram_used = 0
        vram_total = 1
        
        try:
            # ORACLE: High-Speed Vitals Probing
            # Use torch.cuda for fast memory metrics
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            vram_reserved = torch.cuda.memory_reserved() / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            
            # Use nvidia-smi for silicon vitals (Temp, Load, Power)
            # This is cached by the OS, so it's low-latency
            cmd = "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits"
            res = subprocess.check_output(cmd.split(), text=True).strip().split(',')
            gpu_util = float(res[0])
            gpu_temp = float(res[1])
            gpu_pwr = float(res[2])
        except Exception:
            pass

        data = {
            "pod": self.pod_name,
            "rank": self.rank,
            "epoch": epoch,
            "step": step,
            "loss": round(loss, 4),
            "fps": round(throughput, 2),
            "status": status,
            "hw": {
                "util": gpu_util,
                "temp": gpu_temp,
                "pwr": gpu_pwr,
                "vram_u": round(vram_used, 2),
                "vram_r": round(vram_reserved, 2),
                "vram_t": round(vram_total, 2)
            },
            "time": time.time()
        }
        try:
            payload = json.dumps(data).encode('utf-8')
            self.sock.sendto(payload, self.target)
        except Exception:
            pass
