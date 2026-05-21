import torch
import os

# Essential for Blackwell JIT
torch.cuda.set_device(0)

try:
    from flash_attn.cute import flash_attn_func as fa4_func
    HAS_FA4 = True
except ImportError:
    HAS_FA4 = False

def test_fa4(seqlen, dim, heads, batch=8):
    print(f"\n[*] Testing FA4 with Batch={batch}, SeqLen={seqlen}, Dim={dim}, Heads={heads}")
    device = "cuda"
    q = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16).contiguous()
    k = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16).contiguous()
    v = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16).contiguous()
    
    try:
        # Clear cache to force recompilation
        import shutil
        if os.path.exists("/dev/shm/fa4_cache"):
            shutil.rmtree("/dev/shm/fa4_cache")
        os.makedirs("/dev/shm/fa4_cache", exist_ok=True)
        os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = "/dev/shm/fa4_cache"
        
        out = fa4_func(q, k, v, causal=False)
        print("[+] Success")
    except Exception as e:
        print(f"[-] Failed: {e}")

if __name__ == "__main__":
    test_fa4(256, 64, 16)
