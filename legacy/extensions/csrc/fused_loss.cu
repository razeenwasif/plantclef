#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

__global__ void fused_asl_forward_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    const float* __restrict__ logit_adjustments,
    float* __restrict__ losses,
    int batch_size,
    int num_classes,
    float gamma_pos,
    float gamma_neg,
    float clip,
    float eps) {

    int b = blockIdx.x; // One block per batch element
    int tid = threadIdx.x;

    extern __shared__ float s_loss[];

    float local_loss = 0.0f;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        int idx = b * num_classes + c;
        
        // 1. Logit Adjustment
        float x = logits[idx] + logit_adjustments[c];
        float y = targets[idx];

        // 2. Sigmoid
        float xs_pos = 1.0f / (1.0f + expf(-x));
        float xs_neg = 1.0f - xs_pos;

        // 3. Asymmetric Clipping
        if (clip > 0.0f) {
            xs_neg = fminf(xs_neg + clip, 1.0f);
        }

        // 4. Loss Components
        float los_pos = y * logf(fmaxf(xs_pos, eps));
        float los_neg = (1.0f - y) * logf(fmaxf(xs_neg, eps));
        float loss = los_pos + los_neg;

        // 5. Asymmetric Weighting
        float pt = xs_pos * y + xs_neg * (1.0f - y);
        float gamma = y * gamma_pos + (1.0f - y) * gamma_neg;
        float weight = powf(1.0f - pt, gamma);

        local_loss += -loss * weight;
    }

    // Parallel reduction in shared memory
    s_loss[tid] = local_loss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_loss[tid] += s_loss[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        losses[b] = s_loss[0];
    }
}

__global__ void fused_asl_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    const float* __restrict__ logit_adjustments,
    float* __restrict__ grad_logits,
    int batch_size,
    int num_classes,
    float gamma_pos,
    float gamma_neg,
    float clip,
    float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_classes) return;

    int b = idx / num_classes;
    int c = idx % num_classes;

    float x = logits[idx] + logit_adjustments[c];
    float y = targets[idx];

    // Sigmoid and its derivative
    float sig = 1.0f / (1.0f + expf(-x));
    
    // We compute the gradient of the ASL loss with respect to x (the adjusted logit)
    // Note: Since Adjustment is added, dLoss/dLogit = dLoss/dx
    
    float xs_pos = sig;
    float xs_neg = 1.0f - sig;
    bool clipped = false;
    if (clip > 0.0f && (1.0f - sig + clip) > 1.0f) {
        xs_neg = 1.0f;
        clipped = true;
    } else if (clip > 0.0f) {
        xs_neg = 1.0f - sig + clip;
    }

    float pt = xs_pos * y + xs_neg * (1.0f - y);
    float gamma = y * gamma_pos + (1.0f - y) * gamma_neg;
    
    // Gradients of weight and base loss
    // Base ASL: - [ y*log(xs_pos) + (1-y)*log(xs_neg) ] * (1-pt)^gamma
    
    // This is complex to derive manually without errors, but effectively:
    // dL/dx = (dL/d_pt * d_pt/dx) + (weight * d_base_loss/dx)
    
    float d_xs_pos_dx = sig * (1.0f - sig);
    float d_xs_neg_dx = -sig * (1.0f - sig); // if not clipped
    if (clipped) d_xs_neg_dx = 0.0f;

    float d_pt_dx = y * d_xs_pos_dx + (1.0f - y) * d_xs_neg_dx;
    
    float base_loss = -(y * logf(fmaxf(xs_pos, eps)) + (1.0f - y) * logf(fmaxf(xs_neg, eps)));
    float weight = powf(1.0f - pt, gamma);

    float d_weight_dx = -gamma * powf(1.0f - pt, fmaxf(gamma - 1.0f, 0.0f)) * d_pt_dx;
    
    float d_base_loss_dx = -( (y / fmaxf(xs_pos, eps)) * d_xs_pos_dx + ((1.0f - y) / fmaxf(xs_neg, eps)) * d_xs_neg_dx );

    float total_grad = (d_weight_dx * base_loss) + (weight * d_base_loss_dx);

    // Normalize by batch size (since we return .mean() in PyTorch)
    grad_logits[idx] = total_grad * grad_output[0] / batch_size;
}

// ---------------------------------------------------------------------------
// C++ Wrappers
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> fused_asl_forward(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor logit_adjustments,
    float gamma_pos,
    float gamma_neg,
    float clip,
    float eps) {

    const int batch_size = logits.size(0);
    const int num_classes = logits.size(1);

    auto losses = torch::zeros({batch_size}, logits.options());

    int threads = 512;
    size_t shared_mem = threads * sizeof(float);

    fused_asl_forward_kernel<<<batch_size, threads, shared_mem>>>(
        logits.data_ptr<float>(),
        targets.data_ptr<float>(),
        logit_adjustments.data_ptr<float>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes,
        gamma_pos,
        gamma_neg,
        clip,
        eps
    );

    return {losses};
}

torch::Tensor fused_asl_backward(
    torch::Tensor grad_output,
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor logit_adjustments,
    float gamma_pos,
    float gamma_neg,
    float clip,
    float eps) {

    const int batch_size = logits.size(0);
    const int num_classes = logits.size(1);
    auto grad_logits = torch::empty_like(logits);

    int threads = 256;
    int blocks = (batch_size * num_classes + threads - 1) / threads;

    fused_asl_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        logits.data_ptr<float>(),
        targets.data_ptr<float>(),
        logit_adjustments.data_ptr<float>(),
        grad_logits.data_ptr<float>(),
        batch_size,
        num_classes,
        gamma_pos,
        gamma_neg,
        clip,
        eps
    );

    return grad_logits;
}
