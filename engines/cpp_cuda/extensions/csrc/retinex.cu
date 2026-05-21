#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>

/**
 * @file retinex.cu
 * @brief High-performance Multi-Scale Retinex (MSR) using cuFFT.
 * 
 * Optimized for ORACLE pipeline to eliminate CPU-GPU data transfers during 
 * illumination normalization.
 */

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void log_transform_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Log transform: log(1 + x * 255.0)
        data[idx] = logf(1.0f + data[idx] * 255.0f);
    }
}

__global__ void apply_gaussian_filter_kernel(
    cufftComplex* data_f, 
    int h, int w, 
    float sigma) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        // Frequency coordinates
        float fx = (x <= w/2) ? (float)x / w : (float)(x - w) / w;
        float fy = (y <= h/2) ? (float)y / h : (float)(y - h) / h;
        
        float freq_sq = fx*fx + fy*fy;
        float kernel_v = expf(-2.0f * (M_PI * sigma) * (M_PI * sigma) * freq_sq);
        
        int idx = y * w + x;
        data_f[idx].x *= kernel_v;
        data_f[idx].y *= kernel_v;
    }
}

__global__ void subtract_and_accumulate_kernel(
    float* original_log, 
    float* blurred, 
    float* msr_acc, 
    int size, 
    float weight) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        msr_acc[idx] += weight * (original_log[idx] - blurred[idx]);
    }
}

__global__ void finalize_retinex_kernel(
    float* msr, 
    float min_v, 
    float max_v, 
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float range = max_v - min_v;
        if (range > 1e-6f) {
            msr[idx] = (msr[idx] - min_v) / range;
        }
    }
}

// ---------------------------------------------------------------------------
// C++ Wrapper
// ---------------------------------------------------------------------------

torch::Tensor retinex_normalize_cuda(
    torch::Tensor image, 
    std::vector<double> sigmas) {
    
    // Input: [C, H, W] float32 on GPU
    auto device = image.device();
    int C = image.size(0);
    int H = image.size(1);
    int W = image.size(2);
    int size_per_channel = H * W;
    int total_size = C * size_per_channel;

    auto log_image = image.clone();
    
    // 1. Log Transform
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    log_transform_kernel<<<blocks, threads>>>(log_image.data_ptr<float>(), total_size);

    auto msr_acc = torch::zeros_like(log_image);
    float weight = 1.0f / sigmas.size();

    // 2. Multi-Scale Blur via cuFFT
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, H, W, CUFFT_R2C));

    // Temporary buffers for FFT
    // R2C FFT produces H x (W/2 + 1) complex numbers
    int W_complex = W / 2 + 1;
    auto complex_spec = torch::empty({C, H, W_complex, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    for (double sigma : sigmas) {
        auto blurred_image = torch::empty_like(log_image);
        
        for (int c = 0; c < C; ++c) {
            float* in_ptr = log_image.data_ptr<float>() + c * size_per_channel;
            cufftComplex* spec_ptr = reinterpret_cast<cufftComplex*>(complex_spec.data_ptr<float>() + c * H * W_complex * 2);
            float* out_ptr = blurred_image.data_ptr<float>() + c * size_per_channel;

            // Forward FFT
            CHECK_CUFFT(cufftExecR2C(plan, in_ptr, spec_ptr));

            // Apply Gaussian in frequency domain
            dim3 threads2d(16, 16);
            dim3 blocks2d((W_complex + threads2d.x - 1) / threads2d.x, (H + threads2d.y - 1) / threads2d.y);
            apply_gaussian_filter_kernel<<<blocks2d, threads2d>>>(spec_ptr, H, W_complex, (float)sigma);

            // Inverse FFT
            CHECK_CUFFT(cufftExecC2R(plan, spec_ptr, out_ptr));
            
            // cuFFT inverse is unnormalized (multiplies by H*W)
            float norm = 1.0f / (H * W);
            auto slice = blurred_image.select(0, c);
            slice.mul_(norm);
        }

        subtract_and_accumulate_kernel<<<blocks, threads>>>(
            log_image.data_ptr<float>(), 
            blurred_image.data_ptr<float>(), 
            msr_acc.data_ptr<float>(), 
            total_size, 
            weight
        );
    }

    cufftDestroy(plan);

    // 3. Finalize and Normalize Range [0, 1]
    float min_v = msr_acc.min().item<float>();
    float max_v = msr_acc.max().item<float>();
    
    finalize_retinex_kernel<<<blocks, threads>>>(msr_acc.data_ptr<float>(), min_v, max_v, total_size);

    return msr_acc;
}
