#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

__global__ void extract_tiles_kernel(
    const float* __restrict__ image,
    float* __restrict__ tiles,
    int channels,
    int img_h,
    int img_w,
    int tile_h,
    int tile_w,
    int y_step,
    int x_step,
    int num_tiles_x,
    int num_tiles_y) {

    // tile_idx = ty * num_tiles_x + tx
    int tile_idx = blockIdx.z;
    int c = blockIdx.y;
    int ty = tile_idx / num_tiles_x;
    int tx = tile_idx % num_tiles_x;

    int y_start = ty * y_step;
    int x_start = tx * x_step;

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = threadIdx.y; // Assume blockDim.y covers some tile height or we loop

    if (tid_x < tile_w && tid_y < tile_h) {
        int img_y = y_start + tid_y;
        int img_x = x_start + tid_x;

        if (img_y < img_h && img_x < img_w) {
            int img_offset = (c * img_h + img_y) * img_w + img_x;
            int tile_offset = ((tile_idx * channels + c) * tile_h + tid_y) * tile_w + tid_x;
            tiles[tile_offset] = image[img_offset];
        }
    }
}

__global__ void fused_max_pool_kernel(
    const float* __restrict__ tile_probs,
    float* __restrict__ final_probs,
    int num_tiles,
    int num_classes) {

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_classes) return;

    float max_val = -1e10f;
    for (int t = 0; t < num_tiles; ++t) {
        max_val = fmaxf(max_val, tile_probs[t * num_classes + c]);
    }
    final_probs[c] = max_val;
}

// ---------------------------------------------------------------------------
// C++ Wrappers
// ---------------------------------------------------------------------------

torch::Tensor extract_tiles_cuda(
    torch::Tensor image,
    int tile_height,
    int tile_width,
    float overlap_ratio) {

    const int channels = image.size(0);
    const int img_h = image.size(1);
    const int img_w = image.size(2);

    int y_step = static_cast<int>(tile_height * (1.0f - overlap_ratio));
    int x_step = static_cast<int>(tile_width * (1.0f - overlap_ratio));

    int num_tiles_y = (img_h - tile_height) / y_step + 1;
    int num_tiles_x = (img_w - tile_width) / x_step + 1;
    int num_tiles = num_tiles_x * num_tiles_y;

    auto tiles = torch::empty({num_tiles, channels, tile_height, tile_width}, image.options());

    dim3 threads(32, 32);
    dim3 blocks((tile_width + threads.x - 1) / threads.x, 1, num_tiles);
    // Note: This kernel assumes image is in NCHW format (C, H, W)
    // We launch 'num_tiles' blocks in Z, and channels in Y if needed, or loop.
    // Let's refine for channels:
    blocks.y = channels;

    extract_tiles_kernel<<<blocks, threads>>>(
        image.data_ptr<float>(),
        tiles.data_ptr<float>(),
        channels, img_h, img_w,
        tile_height, tile_width,
        y_step, x_step,
        num_tiles_x, num_tiles_y
    );

    return tiles;
}

torch::Tensor fused_max_pool(torch::Tensor tile_predictions) {
    const int num_tiles = tile_predictions.size(0);
    const int num_classes = tile_predictions.size(1);

    auto final_probs = torch::empty({num_classes}, tile_predictions.options());

    int threads = 256;
    int blocks = (num_classes + threads - 1) / threads;

    fused_max_pool_kernel<<<blocks, threads>>>(
        tile_predictions.data_ptr<float>(),
        final_probs.data_ptr<float>(),
        num_tiles,
        num_classes
    );

    return final_probs;
}
