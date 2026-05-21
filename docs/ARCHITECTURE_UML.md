# PlantCLEF 2026: Complete System Architecture

This document provides a comprehensive mapping of all classes in the codebase, organized by their functional layers.

## 1. High-Level Class Diagram (Mermaid)

```mermaid
classDiagram
    %% --- PLATFORM ORCHESTRATION ---
    class OracleCLI {
        <<Entry Point>>
        +registry: ModelRegistry
        +train(phase, seed)
        +infer(ensemble)
        +registry_list()
    }
    class ModelRegistry {
        +path: Path
        +data: Dict[str, ModelArtifact]
        +register(artifact)
        +update_progress(name, epoch, step)
        +health_check()
    }
    class ModelArtifact {
        +name: str
        +type: str
        +status: str
        +accuracy: float
    }
    class OracleConfig {
        <<Dataclass>>
        +hardware: HardwareConfig
        +model: ModelConfig
        +training: TrainingConfig
        +postprocess: PostprocessConfig
    }

    %% --- ENGINES & BUILDERS ---
    class ModelRunner {
        +model: PlantEnsemble
        +cfg: ModelConfig
        +predict(tiles)
    }
    class PulsarHeartbeat {
        +sock: socket
        +pulse(epoch, step, loss, fps)
    }

    %% --- MODEL ARCHITECTURE ---
    class PlantEnsemble {
        +bioclip: PlantBioCLIP
        +dinov3: PlantViTBackbone
        +convnext: PlantConvNeXt
        +gcn_head: EcologicalGCNHead
        +forward(x)
        +apply_lora()
    }

    %% --- TELEMETRY & DASHBOARD ---
    class GoOrchestrator {
        <<Go Process>>
        +PulsarReceiver
        +TelemetryAPI
        +MissionControl
    }
    class ReactDashboard {
        <<Web UI>>
        +ClusterHUD
        +MissionControl
        +Terminal
        +HardwareHUD
    }

    %% --- RELATIONSHIPS ---
    OracleCLI --> OracleConfig : Loads
    OracleCLI --> ModelRegistry : Manages
    
    PlantEnsemble --> PulsarHeartbeat : Emits
    PulsarHeartbeat ..> GoOrchestrator : UDP Pulse
    
    GoOrchestrator --> ReactDashboard : Serves API/Static
    ReactDashboard --> GoOrchestrator : Dispatches Launch
    
    ModelRunner --> PlantEnsemble : Wraps
```

## 2. Mission Control Sequence Diagram (Mermaid)

This diagram traces the flow from a web-based "Launch" click to a multi-GPU training run.

```mermaid
sequenceDiagram
    participant U as User (Mobile/Web)
    participant D as React Dashboard
    participant G as Go Orchestrator (oracle_control)
    participant C as Oracle CLI (oracle.py)
    participant L as Launch Script (launch_oracle.sh)
    participant T as Training Loop (Python)
    participant P as Pulsar (Telemetry)

    U->>D: Click "Launch Phase P2B-Student"
    D->>G: POST /api/launch {phase: "p2b-student"}
    G->>C: Execute "./oracle.py train --phase p2b-student"
    C->>L: launch_oracle.sh p2b-student
    L->>T: torchrun phases.student_distillation.run
    loop Training Every 5 Steps
        T->>P: pulse(epoch, step, loss, fps)
        P-->>G: UDP Datagram (Port 9001)
        G->>G: Update Global Cluster State
        D->>G: GET /api/state (Polling)
        G-->>D: JSON Cluster Snapshot
        D-->>U: Neon Update (Loss curve/FPS)
    end
```

## 3. Technical Stack Trace (ASCII)

Use this map to navigate the code in the order of execution:

```text
[PHASE 0: CONFIG & REGISTRY]
src/config/schema.py (Dataclass Schema)
src/config/loader.py (Hardware Auto-Detection)
src/config/registry.py (Artifact Persistence)
  └── oracle.py (The Unified Nervous System)

[PHASE 1: DATA PREP]
launch_oracle.sh p1 -> phases/foundation_caching/run.py

[PHASE 2: TRAINING]
launch_oracle.sh <p2a|p2b-student|ad-td> -> phases/head_warmup/run.py | phases/student_distillation/run.py | phases/asymmetric_distillation/run.py
  ├── tools/infrastructure/pulsar.py (Heartbeat)
  └── orchestrator/ (Go Control Plane)

[PHASE 3: INFERENCE]
launch_oracle.sh pipeline -> phases/inference/run.py
  └── phases/inference/pipeline.py (Modular Flow)

[PHASE 4: OBSERVABILITY]
dashboard/ (React + Vite + Tailwind)
  └── orchestrator/telemetry.go (Go Mission Control API)
```

## 3. Core Class Descriptions

| Class | File | Responsibility |
| :--- | :--- | :--- |
| **`PlantEnsemble`** | `src/models/ensemble.py` | The central model. Fuses three backbones and manages LoRA adapters. |
| **`CUDASahiEngine`** | `src/models/sahi.py` | Handles high-resolution tiled inference and result aggregation. |
| **`AsymmetricLoss`** | `src/training/losses.py` | Optimized loss function for the 7,800-class long-tail distribution. |
| **`EcologicalGCNHead`** | `src/models/layers/gcn.py` | Generates classifier weights using an ecological knowledge graph. |
| **`PlantDALIPipeline`** | `src/data/dataloader.py` | GPU-accelerated JPEG decoding and image augmentation. |
| **`GPUEcologicalBuilder`** | `tools/data/build_ecological_db.py` | High-speed bioclimatic niche modeling using VRAM sampling. |
