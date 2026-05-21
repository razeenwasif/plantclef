import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field

@dataclass
class ModelArtifact:
    name: str
    type: str # 'seed', 'expert', 'ensemble'
    path: str
    resolution: int
    status: str # 'training', 'ready', 'failed', 'empty'
    accuracy: float = 0.0
    epoch: int = 0
    step: int = 0
    last_updated: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelRegistry:
    def __init__(self, registry_path: str = "models/registry.json"):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, ModelArtifact] = self._load()

    def _load(self) -> Dict[str, ModelArtifact]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, 'r') as f:
                raw = json.load(f)
                return {k: ModelArtifact(**v) for k, v in raw.items()}
        except Exception:
            return {}

    def save(self):
        with open(self.path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.data.items()}, f, indent=4)

    def register(self, artifact: ModelArtifact):
        self.data[artifact.name] = artifact
        self.save()

    def update_progress(self, name: str, epoch: int, step: int, status: str = "training"):
        if name in self.data:
            self.data[name].epoch = epoch
            self.data[name].step = step
            self.data[name].status = status
            self.data[name].last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save()

    def get_ready_checkpoints(self, types: list = None) -> list[str]:
        """Returns paths of all models marked as 'ready'."""
        paths = []
        for m in self.data.values():
            if m.status == 'ready' and (not types or m.type in types):
                if os.path.exists(m.path):
                    paths.append(m.path)
        return paths

    def health_check(self):
        """Verifies if files on disk match the registry."""
        for name, m in self.data.items():
            if not os.path.exists(m.path):
                m.status = "missing"
        self.save()
