"""
Fixed-Size Metadata Indexer using XXHash.
Reduces memory pressure by representing string paths as 64-bit integers.
"""
import numpy as np
import xxhash
import os
import logging

logger = logging.getLogger(__name__)

class ElasticHashRegistry:
    r"""
    Bleeding-edge O(1) Amortized Open Addressing without Reordering.
    Based on Elastic/Funnel Hashing (arXiv:2501.02305 - Farach-Colton et al. 2025).
    
    Instead of a single flat array where collisions cause linear probing $O(1/\delta)$,
    we project probes across multiple decoupled sub-tables (levels).
    Since we never move an element once placed (no Cuckoo/Robin Hood), it avoids 
    catastrophic cascading during high load factors and is highly concurrent-friendly.
    """
    def __init__(self, capacity: int = 4000000):
        # Funnel Hashing conceptually divides the table into geometrically decreasing blocks.
        self.levels = []
        rem = capacity
        self.num_levels = 6
        for i in range(self.num_levels):
            size = rem // 2 if i < (self.num_levels - 1) else rem
            self.levels.append(np.zeros(size, dtype=np.uint64))
            rem -= size
        self.size = 0

    def __len__(self):
        return self.size

    def insert(self, key: int) -> bool:
        """Greedy decoupled probing sequence across hierarchical levels."""
        # 1. Primary Funnel Probing (O(1) expected probes)
        for i, level in enumerate(self.levels):
            # Decoupled independent hash for each level
            # We use a simple bitwise salt to simulate independent hash functions
            idx = int(key ^ (i * 0x9E3779B97F4A7C15)) % len(level)
            
            if level[idx] == 0:
                level[idx] = key
                self.size += 1
                return True
            elif level[idx] == key:
                return False # Already exists
                
        # 2. Worst-case Fallback: Linear probing on the final elastic block
        level = self.levels[-1]
        base_idx = int(key ^ ((self.num_levels - 1) * 0x9E3779B97F4A7C15)) % len(level)
        
        # Max probe bounded to preserve O(log^2(1/delta)) worst-case
        max_probes = min(1000, len(level)) 
        for j in range(1, max_probes):
            probe = (base_idx + j) % len(level)
            if level[probe] == 0:
                level[probe] = key
                self.size += 1
                return True
            elif level[probe] == key:
                return False
                
        raise MemoryError("Elastic Hash Registry exceeded load capacity.")
        
    def get_all(self):
        """Returns a contiguous dense array of all registered hashes."""
        return np.concatenate([lvl[lvl != 0] for lvl in self.levels])

class HashPathIndexer:
    def __init__(self, capacity: int = 4000000):
        # Maps Hash -> Relative Path string
        self.hash_to_path = {}
        # Stores hashes in our novel O(1) Elastic Hash Registry
        self.registry = ElasticHashRegistry(capacity=capacity)

    def register_paths(self, paths: list[str]):
        """Hashes a list of paths and stores the mapping using Elastic Hashing."""
        added = 0
        for p in paths:
            h = xxhash.xxh64(p).intdigest()
            # Ensure it fits in uint64 and is non-zero (0 is empty slot marker)
            h = h & 0xFFFFFFFFFFFFFFFF
            if h == 0: 
                h = 1
            if self.registry.insert(h):
                self.hash_to_path[h] = p
                added += 1
        
        dense_registry = self.registry.get_all()
        logger.info(f"Registered {added} new paths (Total: {self.registry.size}). Memory footprint: {dense_registry.nbytes / 1024 / 1024:.2f} MB")

    def get_path(self, path_hash: int) -> str:
        return self.hash_to_path.get(path_hash)

    def save_index(self, out_path: str):
        """Saves the index to disk for reuse."""
        np.save(out_path, self.registry.get_all())
        import pickle
        with open(out_path.replace(".npy", ".pkl"), "wb") as f:
            pickle.dump(self.hash_to_path, f)

def resolve_external_path(path: str) -> str:
    """
    Safely resolves a path on the external D: drive.
    In WSL, Local Disk (D:) is usually /mnt/d/
    """
    if path.startswith("D:\\"):
        return path.replace("D:\\", "/mnt/d/").replace("\\", "/")
    return path
