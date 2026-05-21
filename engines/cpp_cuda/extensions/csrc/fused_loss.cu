#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

/**
 * @file fused_loss.cu
 * @brief Stabilized Fused Asymmetric Loss (ASL) for Multi-Label Classification.
 * 
 * Implements a high-throughput multi-label loss kernel optimized for BFloat16 
 * and FP16. Includes safety clamping for soft labels and taxonomic smoothing.
 */

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

/**
 * @brief Forward pass for Fused ASL.
 * 
 * Performs sigmoid, clipping, focusing, and aggregation in a single pass 
 * per thread block to minimize global memory bandwidth.
 */
template <typename scalar_t>
__global__ void fused_asl_forward_kernel(
    const scalar_t* __restrict__ logits,
    const scalar_t* __restrict__ targets,
    const scalar_t* __restrict__ logit_adjustments,
    const int* __restrict__ genus_ids,
    const scalar_t* __restrict__ gamma_neg_tensor,
    scalar_t* __restrict__ losses,
    int batch_size,
    int num_classes,
    float gamma_pos,
    float clip,
    float eps,
    float taxon_smoothing,
    const int* __restrict__ target_genus_ids) {

    int b = blockIdx.x; 
    int tid = threadIdx.x;

    extern __shared__ char shared_mem[];
    float* s_loss = (float*)shared_mem;
    
    // Retrieve genus anchor for taxonomic smoothing
    int true_genus = (taxon_smoothing > 0.0f) ? target_genus_ids[b] : -1;

    float local_loss = 0.0f;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        int idx = b * num_classes + c;
        
        // Convert to FP32 for accumulation and mathematical stability
        float x = static_cast<float>(logits[idx]) + static_cast<float>(logit_adjustments[c]);
        float y = static_cast<float>(targets[idx]);

        // Apply Taxonomic Hierarchy Smoothing (Logical augmentation)
        if (taxon_smoothing > 0.0f && y < 0.01f && true_genus != -1) {
            if (genus_ids[c] == true_genus) {
                y = taxon_smoothing;
            }
        }

        // Fused Sigmoid computation
        float sig = 1.0f / (1.0f + expf(-x));
        float xs_pos = sig;
        float xs_neg = 1.0f - sig;

        // Apply ASL Asymmetric Clipping
        if (clip > 0.0f) {
            xs_neg = fminf(xs_neg + clip, 1.0f);
        }

        float los_pos = y * logf(fmaxf(xs_pos, eps));
        float los_neg = (1.0f - y) * logf(fmaxf(xs_neg, eps));
        float loss = los_pos + los_neg;

        // Probability of ground truth
        float pt = xs_pos * y + xs_neg * (1.0f - y);
        float g_neg = static_cast<float>(gamma_neg_tensor[c]);
        float gamma = y * gamma_pos + (1.0f - y) * g_neg;
        
        // Safety Clamp: Prevents NaNs when (1-pt) is negative due to ASL clipping
        float weight = powf(fmaxf(1.0f - pt, 0.0f), gamma);

        local_loss += -loss * weight;
    }

    // Parallel reduction of loss within the block
    s_loss[tid] = local_loss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_loss[tid] += s_loss[tid + s];
        __syncthreads();
    }

    if (tid == 0) losses[b] = static_cast<scalar_t>(s_loss[0]);
}

/**
 * @brief Backward pass for Fused ASL.
 * 
 * Computes the partial derivative of the ASL loss with respect to logits.
 * Mathematically derived to include the impact of asymmetric focusing.
 */
template <typename scalar_t>
__global__ void fused_asl_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ logits,
    const scalar_t* __restrict__ targets,
    const scalar_t* __restrict__ logit_adjustments,
    const int* __restrict__ genus_ids,
    const scalar_t* __restrict__ gamma_neg_tensor,
    scalar_t* __restrict__ grad_logits,
    int batch_size,
    int num_classes,
    float gamma_pos,
    float clip,
    float eps,
    float taxon_smoothing,
    const int* __restrict__ target_genus_ids) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_classes) return;

    int b = idx / num_classes;
    int c = idx % num_classes;

    // Numerical stability conversion
    float x = static_cast<float>(logits[idx]) + static_cast<float>(logit_adjustments[c]);
    float y = static_cast<float>(targets[idx]);

    if (taxon_smoothing > 0.0f && y < 0.01f) {
        int true_genus = target_genus_ids[b];
        if (true_genus != -1 && genus_ids[c] == true_genus) {
            y = taxon_smoothing;
        }
    }

    float sig = 1.0f / (1.0f + expf(-x));
    float xs_pos = sig;
    float xs_neg = 1.0f - sig;
    bool clipped = false;
    if (clip > 0.0f && (1.0f - sig + clip) > 1.0f) {
        xs_neg = 1.0f; clipped = true;
    } else if (clip > 0.0f) {
        xs_neg = 1.0f - sig + clip;
    }

    float pt = xs_pos * y + xs_neg * (1.0f - y);
    float g_neg = static_cast<float>(gamma_neg_tensor[c]);
    float gamma = y * gamma_pos + (1.0f - y) * g_neg;
    
    float d_xs_pos_dx = sig * (1.0f - sig);
    float d_xs_neg_dx = clipped ? 0.0f : -sig * (1.0f - sig);
    float d_pt_dx = y * d_xs_pos_dx + (1.0f - y) * d_xs_neg_dx;
    
    float base_loss = -(y * logf(fmaxf(xs_pos, eps)) + (1.0f - y) * logf(fmaxf(xs_neg, eps)));
    
    // Safety Clamp: pt can exceed 1.0 with clipping
    float safe_base = fmaxf(1.0f - pt, 1e-8f);
    float weight = powf(safe_base, gamma);
    float d_weight_dx = -gamma * powf(safe_base, fmaxf(gamma - 1.0f, 0.0f)) * d_pt_dx;
    float d_base_loss_dx = -( (y / fmaxf(xs_pos, eps)) * d_xs_pos_dx + ((1.0f - y) / fmaxf(xs_neg, eps)) * d_xs_neg_dx );

    float total_grad = (d_weight_dx * base_loss) + (weight * d_base_loss_dx);
    float grad_out_val = static_cast<float>(grad_output[0]);
    grad_logits[idx] = static_cast<scalar_t>(total_grad * grad_out_val / batch_size);
}

// ---------------------------------------------------------------------------
// C++ Wrappers
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> fused_asl_forward(
    torch::Tensor logits, torch::Tensor targets, torch::Tensor logit_adjustments,
    torch::Tensor genus_ids, torch::Tensor gamma_neg_tensor,
    float gamma_pos, float clip, float eps, float taxon_smoothing,
    torch::Tensor target_genus_ids) {

    const int batch_size = logits.size(0);
    const int num_classes = logits.size(1);
    auto losses = torch::zeros({batch_size}, logits.options());

    int threads = 512;
    size_t shared_mem = threads * sizeof(float);

    // Multi-precision dispatch
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(), "fused_asl_forward", ([&] {
        fused_asl_forward_kernel<scalar_t><<<batch_size, threads, shared_mem>>>(
            logits.data_ptr<scalar_t>(), targets.data_ptr<scalar_t>(), logit_adjustments.data_ptr<scalar_t>(),
            genus_ids.data_ptr<int>(), gamma_neg_tensor.data_ptr<scalar_t>(),
            losses.data_ptr<scalar_t>(), batch_size, num_classes, gamma_pos, clip, eps, taxon_smoothing,
            target_genus_ids.data_ptr<int>()
        );
    }));
    return {losses};
}

torch::Tensor fused_asl_backward(
    torch::Tensor grad_output, torch::Tensor logits, torch::Tensor targets,
    torch::Tensor logit_adjustments, torch::Tensor genus_ids, torch::Tensor gamma_neg_tensor,
    float gamma_pos, float clip, float eps, float taxon_smoothing,
    torch::Tensor target_genus_ids) {

    const int batch_size = logits.size(0);
    const int num_classes = logits.size(1);
    auto grad_logits = torch::empty_like(logits);

    int threads = 256;
    int blocks = (batch_size * num_classes + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(), "fused_asl_backward", ([&] {
        fused_asl_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(), targets.data_ptr<scalar_t>(),
            logit_adjustments.data_ptr<scalar_t>(), genus_ids.data_ptr<int>(), gamma_neg_tensor.data_ptr<scalar_t>(),
            grad_logits.data_ptr<scalar_t>(), batch_size, num_classes, gamma_pos, clip, eps, taxon_smoothing,
            target_genus_ids.data_ptr<int>()
        );
    }));
    return grad_logits;
}
