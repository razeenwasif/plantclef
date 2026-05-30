import os
import json
import pytest
import torch
import socket
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.config import load_config, PlantCLEFConfig
from src.config.registry import ModelRegistry, ModelArtifact
from tools.infrastructure.pulsar import PulsarHeartbeat

# --- CONFIG TESTS ---

def test_config_defaults():
    """Verify that PlantCLEFConfig initializes with correct default schema."""
    config = load_config()
    assert config.model.num_classes == 7808
    assert config.model.resolution == 448

def test_config_yaml_merge(tmp_path):
    """Verify that YAML overrides correctly modify the config object."""
    yaml_content = """
architecture:
  resolution: 512
hyperparameters:
  batch_size: 64
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)
    
    config = load_config(str(config_file))
    assert config.model.resolution == 512
    assert config.training.batch_size == 64

@patch("src.config.loader.subprocess.check_output")
def test_hardware_probing(mock_smi):
    """Verify that hardware probing correctly identifies Blackwell/5090."""
    mock_smi.return_value = "NVIDIA RTX PRO 6000 Blackwell"
    
    # 1. Test Single-GPU
    with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
        config = load_config()
        assert config.hardware.preset == "blackwell"
        assert config.hardware.use_compile is True
        
    # 2. Test Multi-GPU
    with patch.dict(os.environ, {"WORLD_SIZE": "3"}):
        config = load_config()
        assert config.hardware.preset == "blackwell"
        assert config.hardware.use_compile is True

# --- REGISTRY TESTS ---

def test_registry_persistence(tmp_path):
    """Verify that ModelRegistry correctly saves and loads artifacts."""
    reg_path = tmp_path / "registry.json"
    registry = ModelRegistry(str(reg_path))
    
    artifact = ModelArtifact(
        name="test_run", type="seed", path="/tmp/model.pth", 
        resolution=448, status="ready", accuracy=85.5
    )
    registry.register(artifact)
    
    # Reload from disk
    new_registry = ModelRegistry(str(reg_path))
    assert "test_run" in new_registry.data
    assert new_registry.data["test_run"].accuracy == 85.5
    assert new_registry.data["test_run"].status == "ready"

def test_registry_updates():
    """Verify that progress updates work correctly."""
    registry = ModelRegistry("/tmp/mock_reg.json")
    artifact = ModelArtifact(name="run_1", type="seed", path="", resolution=0, status="training")
    registry.register(artifact)
    
    registry.update_progress("run_1", epoch=2, step=500, status="training")
    assert registry.data["run_1"].epoch == 2
    assert registry.data["run_1"].step == 500

# --- PULSAR TELEMETRY TESTS ---

@patch("socket.socket")
def test_pulsar_heartbeat(mock_sock):
    """Verify that Pulsar emits correctly structured JSON over UDP."""
    instance = mock_sock.return_value
    pulsar = PulsarHeartbeat()
    
    with patch("torch.cuda.memory_allocated", return_value=1024**3), \
         patch("torch.cuda.memory_reserved", return_value=2*1024**3), \
         patch("torch.cuda.get_device_properties") as mock_props, \
         patch("tools.infrastructure.pulsar.subprocess.check_output") as mock_smi:
        
        mock_smi.return_value = "50, 65, 300"
        mock_props.return_value.total_mem = 10 * 1024**3
        pulsar.pulse(epoch=1, step=10, loss=0.5, throughput=100.0)
    
    # Verify sendto was called
    assert instance.sendto.called
    args, _ = instance.sendto.call_args
    payload = json.loads(args[0].decode('utf-8'))
    
    assert payload["epoch"] == 1
    assert payload["loss"] == 0.5
    assert payload["hw"]["util"] == 50.0
    assert payload["hw"]["temp"] == 65.0
    assert payload["hw"]["vram_u"] == 1.0

# --- CLI INTEGRATION TESTS ---

@patch("plantclef.subprocess.run")
def test_plantclef_cli_dispatch(mock_run):
    """Verify that PlantCLEFCLI correctly dispatches shell commands."""
    from plantclef import PlantCLEFCLI
    import sys
    
    cli = PlantCLEFCLI()
    # Mock sys.argv - Using the correct flag-based syntax
    with patch.object(sys, 'argv', ['plantclef.py', 'train', '--phase', 'p2b', '--seed', '123']):
        cli.run()
    
    # Verify subprocess was called with correct args
    assert mock_run.called
    args, _ = mock_run.call_args
    cmd = args[0]
    assert any("launch.sh" in part for part in cmd)
    assert "p2b-student" in cmd

    assert "123" in cmd

if __name__ == "__main__":
    pytest.main([__file__])
