"""
LLM Arbiter Module: Visual Chain-of-Thought (VCoT) Tie-Breaker and Veto System.
Optimized for Ollama integration within WSL2.
Models: Gemma 4 (Visual), Nemotron-3-Super (Logic)
"""
import torch
import numpy as np
from PIL import Image
import logging
import requests
import json
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class AgenticLLMArbiter:
    def __init__(self, use_gemma=True, use_nemotron=True, endpoint="http://localhost:11434/api/generate"):
        self.use_gemma = use_gemma
        self.use_nemotron = use_nemotron
        self.endpoint = endpoint
        # Model tags as configured in your local Ollama library
        self.gemma_tag = "gemma4" 
        self.nemotron_tag = "nemotron-3-super"

    def _query_ollama(self, prompt: str, model: str, image: Image.Image = None) -> str:
        """Helper to query local Ollama instance."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 128, "temperature": 0.2}
        }
        
        if image:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            payload["images"] = [img_str]

        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"[Arbiter] Ollama connection failed: {e}")
            return ""

    def gemma_tie_breaker(self, image: Image.Image, top_k_species: list[str], top_k_probs: list[float]) -> dict:
        """Uses Gemma 4 to resolve visually ambiguous predictions."""
        if not self.use_gemma: return {"status": "skipped"}
            
        prompt = f"""You are an expert botanist. Differentiate between:
1. {top_k_species[0]} (Confidence: {top_k_probs[0]:.2f})
2. {top_k_species[1]} (Confidence: {top_k_probs[1]:.2f})

Analyze the provided high-resolution crop specifically for morphological differences.
State your visual reasoning step-by-step and name the most likely species."""

        response = self._query_ollama(prompt, self.gemma_tag, image)
        logger.info(f"[Arbiter] Gemma 4 response: {response[:100]}...")
        
        # Simple heuristic for winner detection (In production, use structured JSON output)
        winner = top_k_species[0]
        if top_k_species[1].lower() in response.lower():
            winner = top_k_species[1]

        return {
            "status": "resolved",
            "selected_species": winner,
            "reasoning": response
        }

    def nemotron_logic_veto(self, species: str, prob: float, geo_ph: float, gdd: float, expected_ph: float, expected_gdd: float) -> dict:
        """Uses Nemotron-3 to mediate conflicts between Vision and Ecological Math."""
        if not self.use_nemotron: return {"status": "skipped"}
            
        prompt = f"""Evaluate this conflict:
- Vision Model says: {species} (Confidence: {prob:.2f})
- Environment: Soil pH={geo_ph}, GDD={gdd}
- Species Needs: pH~{expected_ph}, GDD~{expected_gdd}

Does the environment invalidate this prediction? Answer ACCEPT or VETO with reasoning."""

        response = self._query_ollama(prompt, self.nemotron_tag)
        action = "VETO" if "VETO" in response.upper() else "ACCEPT"
        
        return {
            "status": "resolved",
            "action": action,
            "reasoning": response
        }

    def process_predictions(self, image: Image.Image, logits: torch.Tensor, species_ids: list[str], metadata: dict = None) -> torch.Tensor:
        """Main entry point for the LLM Arbiter in the inference pipeline."""
        probs = torch.softmax(logits, dim=0)
        top_k_probs, top_k_idx = torch.topk(probs, k=3)
        
        top_k_species = [species_ids[i] for i in top_k_idx]
        top_k_probs_list = top_k_probs.tolist()
        
        modified_logits = logits.clone()

        if top_k_probs_list[0] - top_k_probs_list[1] < 0.15:
            result = self.gemma_tie_breaker(image, top_k_species, top_k_probs_list)
            if result.get("status") == "resolved":
                idx = species_ids.index(result["selected_species"])
                modified_logits[idx] += 5.0 

        if metadata and self.use_nemotron:
            geo_ph = metadata.get("ph", 7.0)
            gdd = metadata.get("gdd", 1000)
            expected_ph, expected_gdd = 5.0, 800 # Mock values
            
            result = self.nemotron_logic_veto(top_k_species[0], top_k_probs_list[0], geo_ph, gdd, expected_ph, expected_gdd)
            if result.get("action") == "VETO":
                idx = species_ids.index(top_k_species[0])
                modified_logits[idx] -= 20.0 

        return modified_logits
