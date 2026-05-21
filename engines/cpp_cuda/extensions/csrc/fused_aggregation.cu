#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

/**
 * @file fused_aggregation.cu
 * @brief Fused CUDA Aggregation for SAHI Tile Merging.
 */

__global__ void aggregate_max_kernel(
    const float* __restrict__ prob_matrix,
    float* __restrict__ output,
    int n_tiles,
    int num_classes) {
    
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_classes) return;

    float max_val = 0.0f;
    for (int t = 0; t < n_tiles; ++t) {
        float val = prob_matrix[t * num_classes + c];
        if (val > max_val) max_val = val;
    }
    output[c] = max_val;
}

__global__ void aggregate_bayesian_kernel(
    const float* __restrict__ prob_matrix,
    float* __restrict__ output,
    const float* __restrict__ weights,
    int n_tiles,
    int num_classes) {
    
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_classes) return;

    float weighted_sum = 0.0f;
    for (int t = 0; t < n_tiles; ++t) {
        weighted_sum += prob_matrix[t * num_classes + c] * weights[t];
    }
    output[c] = weighted_sum;
}

torch::Tensor fused_aggregate_cuda(
    torch::Tensor prob_matrix,
    std::string method) {
    
    const int n_tiles = prob_matrix.size(0);
    const int num_classes = prob_matrix.size(1);
    auto options = prob_matrix.options();
    auto output = torch::zeros({num_classes}, options);

    int threads = 256;
    int blocks = (num_classes + threads - 1) / threads;

    if (method == "max") {
        aggregate_max_kernel<<<blocks, threads>>>(
            prob_matrix.data_ptr<float>(),
            output.data_ptr<float>(),
            n_tiles,
            num_classes
        );
    } else if (method == "bayesian") {
        // 1. Calculate entropy-based weights on GPU
        auto p = prob_matrix.clamp(1e-12, 1.0);
        auto entropy = -(p * p.log()).sum(1); // (T,)
        auto weights = (-entropy).exp();
        weights = weights / (weights.sum() + 1e-12);

        aggregate_bayesian_kernel<<<blocks, threads>>>(
            prob_matrix.data_ptr<float>(),
            output.data_ptr<float>(),
            weights.data_ptr<float>(),
            n_tiles,
            num_classes
        );
    } else {
        // Fallback to ATen mean
        return prob_matrix.mean(0);
    }

    return output;
}
