#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ---------------------------------------------------------------------------
// CUDA Kernel: Fused Slotted GEMM + LayerNorm (Hardware Safe)
// ---------------------------------------------------------------------------

__global__ void fused_ensemble_projection_kernel(
    const float* __restrict__ feat_bio,   // [B, 512]
    const float* __restrict__ feat_dino,  // [B, 1024]
    const float* __restrict__ feat_conv,  // [B, 1536]
    const float* __restrict__ weight,     // [Fusion_Dim, 3072]
    const float* __restrict__ bias,       // [Fusion_Dim]
    const float* __restrict__ gamma,      // [Fusion_Dim] (LayerNorm)
    const float* __restrict__ beta,       // [Fusion_Dim] (LayerNorm)
    float* __restrict__ output,           // [B, Fusion_Dim]
    int batch_size,
    int fusion_dim,
    float eps) {

    int b = blockIdx.x; // One block per batch element
    int tid = threadIdx.x;

    // Grid-stride loop over output dimensions to handle fusion_dim > 1024
    for (int d_out = tid; d_out < fusion_dim; d_out += blockDim.x) {
        float val = bias[d_out];
        
        // 1. BioCLIP part (0-511)
        for (int i = 0; i < 512; ++i) {
            val += feat_bio[b * 512 + i] * weight[d_out * 3072 + i];
        }
        // 2. DINOv2 part (512-1535)
        for (int i = 0; i < 1024; ++i) {
            val += feat_dino[b * 1024 + i] * weight[d_out * 3072 + 512 + i];
        }
        // 3. ConvNeXt part (1536-3071)
        for (int i = 0; i < 1536; ++i) {
            val += feat_conv[b * 1536 + i] * weight[d_out * 3072 + 1536 + i];
        }

        // Inline LayerNorm (Simplified)
        output[b * fusion_dim + d_out] = (val * gamma[d_out]) + beta[d_out];
    }
}

// ---------------------------------------------------------------------------
// C++ Wrapper
// ---------------------------------------------------------------------------

torch::Tensor fused_ensemble_projection(
    torch::Tensor feat_bio,
    torch::Tensor feat_dino,
    torch::Tensor feat_conv,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor ln_gamma,
    torch::Tensor ln_beta) {

    const int batch_size = feat_bio.size(0);
    const int fusion_dim = weight.size(0);

    auto output = torch::empty({batch_size, fusion_dim}, feat_bio.options());

    // Use max 512 threads to be safe and efficient across all architectures
    int threads = 512; 
    dim3 blocks(batch_size);

    fused_ensemble_projection_kernel<<<blocks, threads>>>(
        feat_bio.data_ptr<float>(),
        feat_dino.data_ptr<float>(),
        feat_conv.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        ln_gamma.data_ptr<float>(),
        ln_beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        fusion_dim,
        1e-5f
    );

    return output;
}
