#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <bitset>

// ---------------------------------------------------------------------------
// Fused ASL Loss (CUDA)
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_asl_forward(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor logit_adjustments,
    float gamma_pos,
    float gamma_neg,
    float clip,
    float eps);

torch::Tensor fused_asl_backward(
    torch::Tensor grad_output,
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor logit_adjustments,
    float gamma_pos,
    float gamma_neg,
    float clip,
    float eps);

// ---------------------------------------------------------------------------
// SAHI Tiling Engine (CUDA)
// ---------------------------------------------------------------------------
torch::Tensor extract_tiles_cuda(
    torch::Tensor image,
    int tile_height,
    int tile_width,
    float overlap_ratio);

torch::Tensor fused_max_pool(torch::Tensor tile_predictions);

// ---------------------------------------------------------------------------
// Taxonomic Filter (C++)
// ---------------------------------------------------------------------------
class TaxonomicFilter {
public:
    static const int MAX_CLASSES = 8192;
    
    TaxonomicFilter(const std::vector<std::vector<int>>& allowed_neighbors) {
        int num_classes = allowed_neighbors.size();
        bitsets_.resize(num_classes);
        for (int i = 0; i < num_classes; ++i) {
            for (int neighbor : allowed_neighbors[i]) {
                if (neighbor >= 0 && neighbor < MAX_CLASSES) bitsets_[i].set(neighbor);
            }
            bitsets_[i].set(i);
        }
    }

    torch::Tensor filter_predictions(torch::Tensor predictions) {
        auto probs_ptr = predictions.data_ptr<float>();
        int num_classes = predictions.size(0);
        int anchor_idx = -1;
        float max_prob = -1.0f;
        for (int i = 0; i < num_classes; ++i) {
            if (probs_ptr[i] > max_prob) {
                max_prob = probs_ptr[i];
                anchor_idx = i;
            }
        }
        if (anchor_idx == -1) return predictions;
        auto filtered = predictions.clone();
        auto filtered_ptr = filtered.data_ptr<float>();
        const auto& allowed = bitsets_[anchor_idx];
        for (int i = 0; i < num_classes; ++i) {
            if (!allowed.test(i)) filtered_ptr[i] = 0.0f;
        }
        return filtered;
    }

private:
    std::vector<std::bitset<MAX_CLASSES>> bitsets_;
};

// ---------------------------------------------------------------------------
// Fused Ensemble Projection (CUDA)
// ---------------------------------------------------------------------------
torch::Tensor fused_ensemble_projection(
    torch::Tensor feat_bio,
    torch::Tensor feat_dino,
    torch::Tensor feat_conv,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor ln_gamma,
    torch::Tensor ln_beta);

// ---------------------------------------------------------------------------
// Parallel Stream Orchestrator
// ---------------------------------------------------------------------------
class StreamOrchestrator {
public:
    StreamOrchestrator() {
        for (int i = 0; i < 3; ++i) {
            cudaStream_t s;
            cudaStreamCreate(&s);
            streams_.push_back(s);
        }
    }
    
    ~StreamOrchestrator() {
        for (auto s : streams_) cudaStreamDestroy(s);
    }

    void synchronize() {
        for (auto s : streams_) cudaStreamSynchronize(s);
    }

    // Returns the raw stream handle for use with torch.cuda.ExternalStream
    uintptr_t get_stream(int id) {
        if (id < 0 || id >= 3) return 0;
        return reinterpret_cast<uintptr_t>(streams_[id]);
    }

private:
    std::vector<cudaStream_t> streams_;
};

// ---------------------------------------------------------------------------
// Module Definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Fused Loss
    m.def("fused_asl_forward", &fused_asl_forward, "Fused ASL Forward");
    m.def("fused_asl_backward", &fused_asl_backward, "Fused ASL Backward");

    // SAHI Tiling Engine
    m.def("extract_tiles", &extract_tiles_cuda, "Extract image tiles directly in CUDA");
    m.def("fused_max_pool", &fused_max_pool, "Fused max pooling over tiles");

    // Ensemble Optimizations
    m.def("fused_projection", &fused_ensemble_projection, "Fused Slotted Ensemble Projection");
    
    py::class_<StreamOrchestrator>(m, "StreamOrchestrator")
        .def(py::init<>())
        .def("synchronize", &StreamOrchestrator::synchronize)
        .def("get_stream", &StreamOrchestrator::get_stream);

    // Taxonomic Filter
    py::class_<TaxonomicFilter>(m, "TaxonomicFilter")
        .def(py::init<const std::vector<std::vector<int>>&>())
        .def("filter_predictions", &TaxonomicFilter::filter_predictions, "Filter impossible combinations");
}
