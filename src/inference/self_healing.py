"""
Async Self-Healing Neural Networks for Test-Time Adaptation.
Based on the concept of fixing model drift in real-time via a ReflexiveLayer.
"""
import torch
import torch.nn as nn
import threading
import queue
import logging

logger = logging.getLogger(__name__)

class ReflexiveLogitAdapter(nn.Module):
    """
    A lightweight adapter applied to the final logits.
    Uses a residual connection to apply a small perturbation to correct drift.
    """
    def __init__(self, num_classes):
        super().__init__()
        # Extremely lightweight: just a bias shift
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, logits):
        # Must handle 1D or 2D inputs
        return logits + self.scale * self.bias

class AsyncHealingEngine:
    def __init__(self, num_classes, learning_rate=0.01):
        self.adapter = ReflexiveLogitAdapter(num_classes)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=learning_rate)
        
        self._lock = threading.RLock()
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._heal_worker, daemon=True)
        self._worker.start()
        
        logger.info(f"[Self-Healing] Async Engine Initialized for {num_classes} classes.")

    def forward(self, logits):
        """Applies the adapter to the logits (thread-safe)."""
        device = logits.device
        with self._lock:
            self.adapter.eval()
            self.adapter.to(device)
            with torch.no_grad():
                return self.adapter(logits)

    def request_heal(self, original_logits, target_distribution):
        """
        Non-blocking request. 
        original_logits: The raw prediction from the vision backbone.
        target_distribution: The 'corrected' distribution from symbolic logic (e.g., AC-3).
        """
        self._queue.put({
            "logits": original_logits.clone().detach(),
            "target": target_distribution.clone().detach()
        })

    def _heal_worker(self):
        # Hybrid Loss: push adapter to match target distribution
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        while True:
            job = self._queue.get()
            try:
                with self._lock:
                    self.adapter.train()
                    
                    device = job["logits"].device
                    self.adapter.to(device)
                    self.optimizer.zero_grad()
                    
                    # Apply adapter
                    # Make sure it's 2D for loss
                    logits_2d = job["logits"].unsqueeze(0)
                    if logits_2d.dim() > 2:
                        logits_2d = logits_2d.view(1, -1)
                        
                    adapted_logits = self.adapter(logits_2d)
                    
                    target_probs = job["target"].unsqueeze(0)
                    if target_probs.dim() > 2:
                        target_probs = target_probs.view(1, -1)
                    
                    # Add a small smoothing to target to avoid log(0)
                    target_probs = torch.clamp(target_probs, min=1e-6)
                    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
                    
                    log_probs = torch.log_softmax(adapted_logits, dim=-1)
                    
                    loss = criterion(log_probs, target_probs)
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        # gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
                        self.optimizer.step()
            except Exception as e:
                logger.error(f"[Self-Healing] Worker error: {e}")
            finally:
                self._queue.task_done()
