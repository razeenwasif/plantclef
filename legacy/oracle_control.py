import os
import sys
import socket
import subprocess
import time
import argparse
import threading
import json

# --- CONFIGURATION ---
PORT = 9999
DISCOVERY_PORT = 10000
TOTAL_WORKERS = 3
PHASES = ["p1", "p2a", "p2b"]

def get_local_ip():
    try:
        out = subprocess.check_output("ip -4 addr show podnet1", shell=True, text=True)
        for line in out.splitlines():
            if "inet " in line:
                return line.split()[1].split('/')[0]
    except:
        pass
    ips = subprocess.check_output("hostname -I", shell=True, text=True).split()
    return ips[0] if ips else "127.0.0.1"

class OracleOrchestrator:
    def __init__(self, phases, role=None):
        self.phases = phases
        self.local_ip = get_local_ip()
        self.master_ip = self.elect_master()
        self.role = "master" if self.local_ip == self.master_ip else "worker"
        self.rank = 0 if self.role == "master" else None
        print(f"🛡️  ORACLE-UNIFIED: Role={self.role}, LocalIP={self.local_ip}, MasterIP={self.master_ip}")

    def elect_master(self):
        nodes = {self.local_ip}
        
        def listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', DISCOVERY_PORT))
            sock.settimeout(5)
            while True:
                try:
                    data, addr = sock.recvfrom(1024)
                    nodes.add(data.decode())
                except: break
        
        t = threading.Thread(target=listener, daemon=True)
        t.start()
        
        # Broadcast presence
        broadcaster = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        broadcaster.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for _ in range(5):
            broadcaster.sendto(self.local_ip.encode(), ('<broadcast>', DISCOVERY_PORT))
            time.sleep(1)
        
        t.join()
        return min(nodes)

    def run_master(self):
        print(f"🧠 [Master] Hub active on {PORT}. Waiting for {TOTAL_WORKERS} workers...")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # ORACLE: Enable TCP KeepAlives on the server socket
        server.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, 'TCP_KEEPIDLE'):
            server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        if hasattr(socket, 'TCP_KEEPINTVL'):
            server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30)
        if hasattr(socket, 'TCP_KEEPCNT'):
            server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 10)
            
        server.bind(('0.0.0.0', PORT))
        server.listen(TOTAL_WORKERS)

        workers = []
        while len(workers) < TOTAL_WORKERS:
            conn, addr = server.accept()
            # ORACLE: Also enable KeepAlives on the accepted connection
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            workers.append(conn)
            rank = len(workers)
            print(f"[Hub] Worker {rank}/{TOTAL_WORKERS} connected from {addr[0]}")
            # Assign rank to worker
            conn.sendall(f"ASSIGN_RANK:{rank}".encode())

        for phase in self.phases:
            print(f"\n🚀 STARTING PHASE: {phase.upper()}")
            # Signal workers to start phase
            for conn in workers:
                conn.sendall(f"START_PHASE:{phase}".encode())
            
            # Run locally
            self.execute_phase(phase)
            
            # Wait for workers to finish (simple sync: wait for them to signal 'DONE')
            finished = 0
            while finished < TOTAL_WORKERS:
                for conn in workers:
                    try:
                        msg = conn.recv(1024).decode()
                        if "DONE" in msg:
                            finished += 1
                            print(f"[Hub] Worker finished phase {phase}. ({finished}/{TOTAL_WORKERS})")
                    except:
                        pass
                time.sleep(5)
            
            print(f"✅ PHASE {phase.upper()} COMPLETE CLUSTER-WIDE")

    def run_worker(self):
        print(f"📡 [Worker] Connecting to Master at {self.master_ip}...")
        while True:
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # ORACLE: Enable KeepAlives on the client side
                client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                client.connect((self.master_ip, PORT))
                break
            except:
                time.sleep(2)

        while True:
            data = client.recv(1024).decode()
            if not data: break

            if data.startswith("ASSIGN_RANK:"):
                self.rank = int(data.split(":")[1])
                print(f"🆔 Assigned Rank: {self.rank}")

            elif data.startswith("START_PHASE:"):
                phase = data.split(":")[1]
                print(f"🏁 Starting Phase: {phase}")
                self.execute_phase(phase)
                client.sendall("DONE".encode())
                print(f"📤 Phase {phase} finished. Waiting for next command...")

    def execute_phase(self, phase):
        print(f"🛠️ [Compute] Sanitizing environment for {phase}...")
        subprocess.run("pkill -9 -f torchrun", shell=True)
        subprocess.run("fuser -k 29505/tcp", shell=True, capture_output=True)
        time.sleep(5)

        env = os.environ.copy()
        env["ORACLE_NODE_RANK"] = str(self.rank)
        env["ORACLE_MASTER_IP"] = self.master_ip
        env["PYTHONPATH"] = os.getcwd()
        
        launch_cmd = ["bash", "launch_oracle.sh", phase, self.role]
        if phase == "p2b" and self.role == "master":
            launch_cmd.append("--resume")
            launch_cmd.append("latest")

        print(f"🚀 [Launch] {' '.join(launch_cmd)}")
        process = subprocess.Popen(launch_cmd, env=env)
        process.wait()

def main():
    parser = argparse.ArgumentParser(description="ORACLE Unified Zero-Config Orchestrator")
    parser.add_argument("--phases", type=str, nargs="+", default=PHASES, help="Phases to run (e.g., --phases p2b)")
    args = parser.parse_args()

    orchestrator = OracleOrchestrator(phases=args.phases)
    if orchestrator.role == "master":
        orchestrator.run_master()
    else:
        orchestrator.run_worker()

if __name__ == "__main__":
    main()
