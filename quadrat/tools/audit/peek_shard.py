import tarfile
import numpy as np
import os

shard_path = "/dev/shm/shards/val_00000.tar"
if not os.path.exists(shard_path):
    shard_path = "/workspace/plantclef/shards/val_00000.tar"

print(f"Peeking into {shard_path}...")
with tarfile.open(shard_path, "r") as tar:
    count = 0
    for member in tar.getmembers():
        if member.name.endswith(".cls"):
            f = tar.extractfile(member)
            data = f.read()
            # If it's binary LE 4 bytes
            val_le = np.frombuffer(data, dtype="<i4")[0]
            print(f"Sample {count} | File: {member.name} | Bytes: {data.hex()} | Int (LE): {val_le}")
            count += 1
            if count >= 5:
                break
