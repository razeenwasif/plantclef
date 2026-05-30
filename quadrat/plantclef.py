#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import signal
from pathlib import Path
from src.config import load_config
from src.config.registry import ModelRegistry, ModelArtifact

# Per-phase YAML configs (used when --config is not explicitly overridden)
_PHASE_CONFIGS = {
    "p1":          "configs/p1_extract.yaml",
    "p2a":         "configs/p2a_warmup.yaml",
    "p2b-expert":  "configs/p2b_teacher_bioclip.yaml",
    "p2b-student": "configs/p2b_student.yaml",
    "ad-td":       "configs/p2b_student.yaml",
    "swa":         "configs/p2b_student.yaml",
    "inf":         "configs/inference.yaml",
}
_PHASE_ALIASES = {
    "p2b": "p2b-student"
}
_LEGACY_DEFAULT = "configs/training.yaml"


class PlantCLEFCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PlantCLEF 2026 ML CLI")
        self.registry = ModelRegistry()

        subparsers = self.parser.add_subparsers(dest="command")

        # Command: train
        train_parser = subparsers.add_parser("train", help="Launch a training phase")
        train_parser.add_argument("--phase", choices=["p1", "p2a", "p2b", "p2b-expert", "p2b-student", "ad-td", "swa"],
                                  required=True, help="Phase to run")
        train_parser.add_argument("--role", choices=["master", "worker", "sprint"],
                                  default="sprint")
        train_parser.add_argument("--seed", type=int, default=42)
        train_parser.add_argument("--dataset", type=str, default=None, help="Named dataset config")
        train_parser.add_argument("--config", type=str, default=None,
                                  help="Override YAML config (default: per-phase config)")
        train_parser.add_argument("--rank", type=int,
                                  help="Force node rank (0=master, 1+=worker)")
        train_parser.add_argument("--force", action="store_true",
                                  help="Force-overwrite existing checkpoints or SWA models")
        train_parser.add_argument("--mode", choices=["cuda", "tpu", "auto"],
                                  default="auto",
                                  help="Accelerator: cuda (default on CUDA hosts), tpu, or auto-detect.")
        train_parser.add_argument("--cluster", type=str, default=None,
                                  help="Path to cluster.yaml. Derives NNODES, MASTER_IP, "
                                       "NODE_RANK, device count, and (optionally) --mode from "
                                       "the manifest. See configs/cluster.example.yaml.")
        train_parser.add_argument("--host-id", type=str, default=None,
                                  help="This host's id in the cluster manifest. "
                                       "Defaults to auto-detection by hostname.")

        # Command: infer
        infer_parser = subparsers.add_parser("infer", help="Run optimized inference")
        infer_parser.add_argument("role_pos", nargs="?",
                                  choices=["master", "worker", "sprint"], default=None,
                                  metavar="ROLE",
                                  help="Role shorthand positional (e.g. ./plantclef.py infer --ensemble sprint)")
        infer_parser.add_argument("--config", type=str, default=None)
        infer_parser.add_argument("--role", choices=["master", "worker", "sprint"],
                                  default="sprint")
        infer_parser.add_argument("--ensemble", action="store_true",
                                  help="Run full multi-model ensemble")

        # Command: registry
        reg_parser = subparsers.add_parser("registry", help="Model registry operations")
        reg_parser.add_argument("action", nargs="?",
                                choices=["list", "prune", "sync"], default=None)
        reg_parser.add_argument("--list",  dest="flag_list",  action="store_true")
        reg_parser.add_argument("--prune", dest="flag_prune", action="store_true")

        # Command: dashboard
        dash_parser = subparsers.add_parser("dashboard", help="Launch Neon Command Center")
        dash_parser.add_argument("--port", type=int, default=9000, help="Port to host on")

        # Command: cache
        cache_parser = subparsers.add_parser("cache", help="Build teacher feature cache")
        cache_parser.add_argument("--role",    choices=["master", "worker", "sprint"],
                                  default="sprint")
        cache_parser.add_argument("--seed",    type=int, default=42)
        cache_parser.add_argument("--rank",    type=int,
                                  help="Manual node rank (worker only)")
        cache_parser.add_argument("--res",     type=int, default=512,
                                  help="Teacher input resolution (default: 512)")
        cache_parser.add_argument("--mode",    choices=["cuda", "tpu", "auto"],
                                  default="auto",
                                  help="Accelerator: cuda (default on CUDA hosts), tpu, or auto-detect.")

        # Command: infer (also accelerator-aware)
        infer_parser.add_argument("--mode",    choices=["cuda", "tpu", "auto"],
                                  default="auto",
                                  help="Accelerator: cuda (default on CUDA hosts), tpu, or auto-detect.")
        cache_parser.add_argument("--batch",   type=int, default=512,
                                  help="Per-GPU batch size (default: 512)")
        cache_parser.add_argument("--workers", type=int, default=0,
                                  help="DALI CPU threads (0 = auto-detect)")
        cache_parser.add_argument("--out-dir", dest="out_dir", type=str, default=None,
                                  help="Output directory for .npy cache (e.g. /dev/shm)")
        cache_parser.add_argument("--no-compile", dest="no_compile", action="store_true",
                                  help="Skip torch.compile (faster startup)")
        cache_parser.add_argument("--force",   action="store_true",
                                  help="Overwrite existing cache")
        cache_parser.add_argument("--resume",  action="store_true",
                                  help="Resume from done-mask checkpoint")

    def run(self):
        args, unknown = self.parser.parse_known_args()

        if args.command == "train":
            self._handle_train(args, unknown)
        elif args.command == "infer":
            self._handle_infer(args, unknown)
        elif args.command == "registry":
            self._handle_registry(args)
        elif args.command == "dashboard":
            self._handle_dashboard(args)
        elif args.command == "cache":
            self._handle_cache(args, unknown)
        else:
            self.parser.print_help()

    def _handle_cache(self, args, unknown):
        out_dir = args.out_dir or "/dev/shm"
        print(f"[plantclef] Teacher cache  role={args.role}  "
              f"res={args.res}  batch={args.batch}  out={out_dir}")
        env = os.environ.copy()
        if getattr(args, "rank", None) is not None:
            env["CLUSTER_NODE_RANK"] = str(args.rank)
        env["PLANTCLEF_SEED"] = str(args.seed)
        env["CLUSTER_MODE"] = getattr(args, "mode", "auto")
        env["OMP_NUM_THREADS"] = "32"

        cmd = [
            "./src/setup/launch.sh", "cache", args.role,
            "--teacher_res", str(args.res),
            "--batch_size",  str(args.batch),
            "--out_dir",     out_dir,
        ]
        if args.workers:    cmd += ["--num_workers", str(args.workers)]
        if args.no_compile: cmd.append("--no_compile")
        if args.force:      cmd.append("--force")
        if args.resume:     cmd.append("--resume")
        if unknown:         cmd += unknown
        subprocess.run(cmd, check=True, env=env)

    def _handle_dashboard(self, args):
        port = args.port or 9000
        print(f"[plantclef] Activating Neon Command Center on http://localhost:{port}")

        # 1. Start the Go API in the background
        print("[plantclef] Starting backend API...")
        cwd = "orchestrator"
        binary = "./coord"
        env = os.environ.copy()
        env["CLUSTER_API_PORT"] = "8081" # Keep API on a separate port

        try:
            # Launch Go backend (API only)
            api_proc = subprocess.Popen([binary, "hub"], cwd=cwd, env=env, 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 2. Start Python Static Server for the Frontend
            # The dashboard lives in the Oracle product repo, not here.
            # Point PLANTCLEF_DASHBOARD_DIST at its build output to serve it
            # locally. Default falls back to the original path for users who
            # keep ~/Oracle and ~/Research/plantclef side by side.
            dist_path = os.path.abspath(
                os.environ.get("PLANTCLEF_DASHBOARD_DIST", "../../Oracle/apps/oracle-plant/dist")
            )
            print(f"[plantclef] Serving dashboard files from {dist_path}...")

            if not os.path.exists(dist_path):
                print(f"[Error] Dashboard files not found at {dist_path}")
                print("        Set PLANTCLEF_DASHBOARD_DIST to the dashboard's dist/ path.")
                api_proc.terminate()
                return

            import http.server
            import socketserver

            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=dist_path, **kwargs)
                def log_message(self, format, *args):
                    return # Suppress logs to keep terminal clean

                def end_headers(self):
                    # Add CORS and no-cache headers
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                    super().end_headers()

            # Allow address reuse to prevent "Port already in use" errors on restart
            socketserver.TCPServer.allow_reuse_address = True

            with socketserver.TCPServer(("0.0.0.0", port), QuietHandler) as httpd:
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\n[plantclef] Shutting down...")
                    httpd.shutdown()
                    api_proc.terminate()
        except FileNotFoundError:
            print(f"[Error] Backend binary {binary} not found.")
        except Exception as e:
            print(f"[Error] Failed to start dashboard: {e}")

    def _handle_train(self, args, unknown):
        # Resolve aliases
        phase = _PHASE_ALIASES.get(args.phase, args.phase)
        
        # Resolve config: explicit flag > per-phase default > legacy fallback
        cfg = args.config or _PHASE_CONFIGS.get(phase, _LEGACY_DEFAULT)

        run_name = f"plantclef_s{args.seed}" if phase != "p1" else "phase1_extract"
        print(f"[plantclef] {phase} ({args.role})  seed={args.seed}  config={cfg}")

        self.registry.register(ModelArtifact(
            name=run_name, type="seed", path="", resolution=0, status="training"
        ))

        env = os.environ.copy()
        env["PLANTCLEF_SEED"] = str(args.seed)
        env["PLANTCLEF_NAME"] = f"plantclef_s{args.seed}"
        env["CLUSTER_MODE"] = getattr(args, "mode", "auto")
        if args.rank is not None:
            env["CLUSTER_NODE_RANK"] = str(args.rank)

        # ── Cluster manifest path: build a TrainingTask, let its
        # to_env() override the cluster-shape vars (NNODES, MASTER_IP,
        # NODE_RANK, device count, CLUSTER_MODE). Explicit --rank still
        # wins over the manifest-derived NODE_RANK. ──────────────────
        if getattr(args, "cluster", None):
            from src.cluster import build_task_from_args
            task = build_task_from_args(args, unknown)
            for k, v in task.to_env().items():
                env[k] = v
            if args.rank is not None:
                env["CLUSTER_NODE_RANK"] = str(args.rank)
            # If the manifest selected our role and the caller didn't
            # specify one, forward it to the launcher positional.
            if task.host and not (args.role and args.role != "sprint"):
                args.role = task.host.role
            print(f"[plantclef] Cluster: {task.manifest.name if task.manifest else '-'}  "
                  f"Host: {task.host.id if task.host else '-'}  "
                  f"Role: {args.role}  "
                  f"Devices: {task.host.device_count if task.host else '-'}  "
                  f"Mode: {env.get('CLUSTER_MODE')}")

        # All training phases (p1, p2a, p2b) now go through launch.sh
        # launch.sh handles the multi-seed looping for sprint roles internally.
        cmd = ["./src/setup/launch.sh", phase, args.role,
               "--seed", str(args.seed), "--config", cfg]
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        if args.force:
            cmd.append("--force")
        if unknown:
            cmd += unknown
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError:
            print(f"[plantclef] Phase {args.phase} failed. Check logs.")
            sys.exit(1)

    def _handle_infer(self, args, unknown):
        cfg  = args.config or _PHASE_CONFIGS.get("inf")
        role = args.role_pos if args.role_pos else args.role
        print(f"[plantclef] Launching Inference ({role})  config={cfg}")
        
        if args.ensemble:
            # Full pipeline (SWA -> Opt -> Calibrate -> Ensemble)
            cmd = ["./src/setup/launch.sh", "pipeline", role]
        else:
            # Single-model inference
            cmd = ["./src/setup/launch.sh", "inf", role,
                   "--config", cfg]
        
        if unknown:
            cmd += unknown

        env = os.environ.copy()
        env["CLUSTER_MODE"] = getattr(args, "mode", "auto")
        subprocess.run(cmd, env=env, check=True)

    def _handle_registry(self, args):
        action = args.action
        if not action:
            if args.flag_list:  action = "list"
            elif args.flag_prune: action = "prune"
            else: action = "list"
        if action == "list":
            models = self.registry.list()
            print("\n--- plantclef MODEL REGISTRY ---")
            if not models:
                print("  (empty)")
            for m in models:
                status = f"[{m.status.upper()}]"
                print(f"  {status:<12} {m.name:<30} res={m.resolution}  {m.path}")
        elif action == "prune":
            print("Cleaning up failed artifacts...")
            self.registry.prune()

if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n[plantclef] Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    cli = PlantCLEFCLI()
    cli.run()
