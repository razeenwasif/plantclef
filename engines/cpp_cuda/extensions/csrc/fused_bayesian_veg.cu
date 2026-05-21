#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// --- CUDA KERNEL ---
// This kernel calculates the Excess Green (ExG) index for a tile 
// and immediately weights the species logits by the vegetation ratio.
__global__ void bayesian_veg_fusion_kernel(
    const float* __restrict__ images,      // [N, 3, H, W]
    float* __restrict__ logits,            // [N, C]
    const int N, const int C, const int H, const int W) {
    
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= N) return;

    // 1. Calculate Vegetation Fraction (ExG > 20)
    int veg_pixels = 0;
    int total_pixels = H * W;
    
    for (int i = 0; i < total_pixels; ++i) {
        float r = images[tile_idx * 3 * total_pixels + 0 * total_pixels + i];
        float g = images[tile_idx * 3 * total_pixels + 1 * total_pixels + i];
        float b = images[tile_idx * 3 * total_pixels + 2 * total_pixels + i];
        
        // ExG = 2g - r - b
        if ((2.0f * g - r - b) > 20.0f) {
            veg_pixels++;
        }
    }
    float veg_ratio = (float)veg_pixels / (float)total_pixels;

    // 2. Weight Logits by Vegetation Ratio
    // Species with non-plant characteristics get suppressed in low-vegetation tiles
    for (int c = 0; c < C; ++c) {
        logits[tile_idx * C + c] *= veg_ratio;
    }

}

// --- BINDING ---
void launch_bayesian_veg_fusion(
    torch::Tensor images,
    torch::Tensor logits) {
    
    const int N = images.size(0);
    const int C = logits.size(1);
    const int H = images.size(2);
    const int W = images.size(3);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    bayesian_veg_fusion_kernel<<<blocks, threads>>>(
        images.data_ptr<float>(),
        logits.data_ptr<float>(),
        N, C, H, W
    );
}
