import os
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

@pipeline_def
def test_pipe(tar_file, index_file):
    jpegs, labels = fn.readers.webdataset(
        paths=[tar_file],
        index_paths=[index_file],
        ext=["jpg", "cls"],
        name="Reader"
    )
    return jpegs, labels

def main():
    tar_f = "/dev/shm/shards/train_00000.tar"
    idx_f = "/dev/shm/shards/test_format.idx"
    
    if os.path.exists(idx_f): os.remove(idx_f)
    
    print(f"[*] Generating reference index for {tar_f}...")
    pipe = test_pipe(tar_file=tar_f, index_file=idx_f, batch_size=1, num_threads=1, device_id=0)
    pipe.build()
    
    if os.path.exists(idx_f):
        print(f"[+] Reference index created at {idx_f}")
        with open(idx_f, 'r') as f:
            print("--- FILE START ---")
            for _ in range(3):
                print(f.readline().strip())
            print("--- FILE END ---")
    else:
        print("[!] DALI failed to create the index file.")

if __name__ == "__main__":
    main()
