#include <torch/script.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <string>

/**
 * @file inference_engine.cpp
 * @brief Unified Native Inference Engine (C++/LibTorch).
 * 
 * Orchestrates the entire inference pipeline:
 * 1. Native Image Decoding/Tiling
 * 2. Multi-Scale Retinex (CUDA)
 * 3. Batch Inference (LibTorch)
 * 4. Fused Aggregation (CUDA)
 */

// External CUDA kernel declarations
torch::Tensor retinex_normalize_cuda(torch::Tensor image, std::vector<double> sigmas);
torch::Tensor sahi_tiling_cuda(torch::Tensor image, torch::Tensor x_starts, torch::Tensor y_starts, int tile_size);
torch::Tensor fused_max_pool(torch::Tensor tile_predictions);

// Rust FFI declarations
extern "C" {
    struct ResizedImage {
        uint8_t* data;
        uint32_t width;
        uint32_t height;
        uint32_t channels;
    };

    ResizedImage rust_resize_image(const char* path, uint32_t target_size);
    void rust_free_image(uint8_t* data, size_t length);
}

class NativeInferenceEngine {
public:
    NativeInferenceEngine(const std::string& model_path) {
        try {
            // Load the TorchScript model
            module_ = torch::jit::load(model_path);
            module_.eval();
            module_.to(at::kCUDA);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.msg() << std::endl;
        }
    }

    /**
     * @brief Run end-to-end inference on a single image file path.
     * 
     * Uses Rust for high-speed I/O and C++/CUDA for processing.
     */
    torch::Tensor predict_from_file(
        const std::string& image_path,
        torch::Tensor x_starts,
        torch::Tensor y_starts,
        int tile_size,
        std::vector<double> retinex_sigmas,
        int batch_size = 16) {

        // 1. Rust I/O: Zero-Copy Load & Resize
        ResizedImage rimg = rust_resize_image(image_path.c_str(), 700); // 700px standard
        if (!rimg.data) {
            throw std::runtime_error("Failed to load image via Rust: " + image_path);
        }

        // 2. Wrap Rust buffer in Torch Tensor
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto cpu_tensor = torch::from_blob(rimg.data, {rimg.height, rimg.width, rimg.channels}, options);
        
        // 3. Push to GPU & Convert to float32
        auto gpu_tensor = cpu_tensor.to(at::kCUDA).to(torch::kFloat32).permute({2, 0, 1}).div(255.0);

        // Free Rust buffer now that it's on GPU
        rust_free_image(rimg.data, rimg.height * rimg.width * rimg.channels);

        // 4. Run existing prediction logic
        return predict(gpu_tensor, x_starts, y_starts, tile_size, retinex_sigmas, batch_size);
    }

    /**
     * @brief Run prediction on an existing GPU tensor.
     */
    torch::Tensor predict(
        torch::Tensor image,
        torch::Tensor x_starts,
        torch::Tensor y_starts,
        int tile_size,
        std::vector<double> retinex_sigmas,
        int batch_size = 16) {

        torch::NoGradGuard no_grad;

        // 1. Preprocessing: Zero-Copy Retinex
        auto processed = retinex_normalize_cuda(image, retinex_sigmas);

        // 2. SAHI: Native Tiling
        auto tiles = sahi_tiling_cuda(processed, x_starts, y_starts, tile_size);

        // 3. Batch Inference
        int num_tiles = tiles.size(0);
        std::vector<torch::Tensor> all_logits;

        for (int i = 0; i < num_tiles; i += batch_size) {
            int current_batch = std::min(batch_size, num_tiles - i);
            auto batch = tiles.narrow(0, i, current_batch);
            
            // Forward pass
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(batch);
            auto output = module_.forward(inputs).toTensor();
            
            all_logits.push_back(output);
        }

        auto concatenated_logits = torch::cat(all_logits, 0);

        // 4. Fused Aggregation
        return fused_max_pool(concatenated_logits);
    }

private:
    torch::jit::script::Module module_;
};

// ---------------------------------------------------------------------------
// Python Bindings for the Engine
// ---------------------------------------------------------------------------

void bind_inference_engine(py::module& m) {
    py::class_<NativeInferenceEngine>(m, "NativeInferenceEngine")
        .def(py::init<const std::string&>())
        .def("predict", &NativeInferenceEngine::predict, 
             py::arg("image"), 
             py::arg("x_starts"), 
             py::arg("y_starts"), 
             py::arg("tile_size"), 
             py::arg("retinex_sigmas"),
             py::arg("batch_size") = 16)
        .def("predict_from_file", &NativeInferenceEngine::predict_from_file,
             py::arg("image_path"),
             py::arg("x_starts"),
             py::arg("y_starts"),
             py::arg("tile_size"),
             py::arg("retinex_sigmas"),
             py::arg("batch_size") = 16);
}
