import os
import shutil
import glob
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DynamicShardManager:
    """
    Manages a sliding window of WebDataset shards in RAM-disk (/dev/shm).
    Ideal for low-RAM devices that cannot hold the entire dataset in memory.
    """
    def __init__(self, 
                 source_dir: str = "/workspace/plantclef/shards",
                 target_dir: str = "/dev/shm/shards",
                 prefix: str = "train_",
                 window_size: int = 5,
                 num_shards: int = 1,
                 shard_id: int = 0):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.prefix = prefix
        self.window_size = window_size
        
        # Get all relevant shards from source
        all_shards = sorted(glob.glob(os.path.join(source_dir, f"{prefix}*.tar")))
        
        # Shard the shards (DDP style)
        if num_shards > 1:
            self.all_shards = [all_shards[i] for i in range(shard_id, len(all_shards), num_shards)]
        else:
            self.all_shards = all_shards
            
        self.current_idx = 0
        os.makedirs(target_dir, exist_ok=True)
        
    def prepare_next_window(self) -> List[str]:
        """
        Deletes old shards in target_dir and copies the next 'window_size' shards.
        Returns the list of local paths in target_dir.
        """
        # Cleanup
        existing = glob.glob(os.path.join(self.target_dir, f"{self.prefix}*.tar"))
        for f in existing:
            try:
                os.remove(f)
            except OSError:
                pass
        
        if self.current_idx >= len(self.all_shards):
            return []
            
        end_idx = min(self.current_idx + self.window_size, len(self.all_shards))
        batch = self.all_shards[self.current_idx:end_idx]
        self.current_idx = end_idx
        
        local_paths = []
        for remote_path in batch:
            local_path = os.path.join(self.target_dir, os.path.basename(remote_path))
            logger.info(f"[ShardManager] Copying {os.path.basename(remote_path)} to RAM...")
            shutil.copy(remote_path, local_path)
            local_paths.append(local_path)
            
        return local_paths

    def has_more(self) -> bool:
        return self.current_idx < len(self.all_shards)

    def reset(self):
        self.current_idx = 0
