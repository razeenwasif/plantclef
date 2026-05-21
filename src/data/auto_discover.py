import os
import subprocess
import logging
import yaml
from pathlib import Path
from src.inference.llm_arbiter import AgenticLLMArbiter

logger = logging.getLogger(__name__)

class AgenticDatasetFinder:
    """
    Combines rapid OS-level heuristics with the reasoning power of Gemma 4
    to autonomously locate and configure missing datasets across the system.
    """
    def __init__(self):
        self.arbiter = AgenticLLMArbiter()
        
    def fast_scan(self, dataset_name: str) -> list[str]:
        """Uses fast OS utilities to find candidate directories."""
        # Focus on WSL mounted drives and home dir
        search_roots = ["/mnt/c", "/mnt/d", "/mnt/e", "/home"]
        candidates = []
        
        # Heuristic: the folder probably contains part of the name
        # We take the first 6 chars to be safe (e.g., "plantn" from "plantnet300k")
        query = dataset_name[:6].lower()
        
        for root in search_roots:
            if not os.path.exists(root):
                continue
                
            try:
                # Use find to locate directories, maxdepth 6 to prevent infinite crawls
                cmd = f"find {root} -maxdepth 6 -type d -iname '*{query}*' 2>/dev/null | head -n 15"
                result = subprocess.check_output(cmd, shell=True, text=True).strip()
                if result:
                    for line in result.split('\n'):
                        # Exclude obvious non-dataset OS dirs
                        if "Recycle.Bin" not in line and "System Volume Information" not in line:
                            candidates.append(line)
            except subprocess.CalledProcessError:
                pass
                
        return candidates

    def get_dir_structure(self, path: str) -> str:
        """Returns a summary of the directory contents for the LLM."""
        try:
            items = os.listdir(path)
            subdirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]
            return f"Subdirs: {subdirs[:8]}..., Files: {files[:5]}..."
        except Exception as e:
            return str(e)

    def locate_and_save(self, dataset_name: str) -> dict:
        """Finds the dataset, asks Gemma 4 to verify, and saves the config."""
        logger.info(f"🕵️  Agentic Search Triggered for '{dataset_name}'. Scanning OS mounts...")
        candidates = self.fast_scan(dataset_name)
        
        if not candidates:
            logger.error("OS scan yielded no candidates.")
            return {}
            
        logger.info(f"Found {len(candidates)} candidates. Passing context to Gemma 4...")
        
        prompt = f"I am looking for the root directory of a machine learning dataset named '{dataset_name}'.\n"
        prompt += "Below are candidate paths found on the system and a preview of their contents:\n\n"
        
        for i, cand in enumerate(candidates):
            prompt += f"Candidate {i+1}:\nPath: {cand}\nContents: {self.get_dir_structure(cand)}\n\n"
            
        prompt += """Analyze these candidates. A dataset root usually contains 'train', 'val', 'test' folders, or a large number of class folders. If a candidate is just an empty parent folder or a documentation folder, do not select it. Select the folder that directly contains the images or the splits.
Which candidate is the actual root directory of the dataset?
Respond with ONLY the exact absolute path of the correct candidate. No markdown, no explanations."""

        response = self.arbiter._query_ollama(prompt, self.arbiter.gemma_tag)
        best_path = response.strip().strip("`").strip("'").strip('"')
        
        if os.path.exists(best_path):
            logger.info(f"✅ Gemma 4 identified verified path: {best_path}")
            
            # Determine structure
            ds_type = "flat"
            if os.path.exists(os.path.join(best_path, "train")) and os.path.exists(os.path.join(best_path, "val")):
                ds_type = "pre_split"
                
            cfg_update = {
                "type": ds_type,
                "root": best_path,
                "layout": "folder_per_class"
            }
            if ds_type == "pre_split":
                cfg_update["train_dir"] = "train"
                cfg_update["val_dir"] = "val"
                cfg_update["test_dir"] = "test"
                
            self._save_to_yaml(dataset_name, cfg_update)
            return cfg_update
        else:
            logger.warning(f"Gemma 4 hallucinated or failed to find a valid path. Response: {response}")
            return {}
            
    def _save_to_yaml(self, dataset_name: str, config: dict):
        yaml_path = "configs/datasets.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
            
        data[dataset_name] = config
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"💾 Saved Agentic Discovery results to {yaml_path}")
