#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <bitset>

// ---------------------------------------------------------------------------
// Fused ASL Loss (CUDA)
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_asl_forward(
    torch::Tensor logits, torch::Tensor targets, torch::Tensor logit_adjustments,
    torch::Tensor genus_ids, torch::Tensor gamma_neg_tensor,
    float gamma_pos, float clip, float eps, float taxon_smoothing,
    torch::Tensor target_genus_ids);

torch::Tensor fused_asl_backward(
    torch::Tensor grad_output, torch::Tensor logits, torch::Tensor targets,
    torch::Tensor logit_adjustments, torch::Tensor genus_ids, torch::Tensor gamma_neg_tensor,
    float gamma_pos, float clip, float eps, float taxon_smoothing,
    torch::Tensor target_genus_ids);

// ---------------------------------------------------------------------------
// SAHI Tiling Engine (CUDA)
// ---------------------------------------------------------------------------
torch::Tensor sahi_tiling_cuda(
    torch::Tensor image,
    torch::Tensor x_starts,
    torch::Tensor y_starts,
    int tile_size);

torch::Tensor fused_aggregate_cuda(
    torch::Tensor prob_matrix,
    std::string method);

// ---------------------------------------------------------------------------
// Fused Gated Feature Aggregation (GFAM) Projection (CUDA)
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_gfam_projection(
    torch::Tensor feat_bio,
    torch::Tensor feat_dino,
    torch::Tensor feat_conv,
    torch::Tensor gating,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor agg_w,
    torch::Tensor agg_b,
    torch::Tensor ln_gamma,
    torch::Tensor ln_beta);

// ---------------------------------------------------------------------------
// Ecological GCN (CUDA)
// ---------------------------------------------------------------------------
torch::Tensor FusedGcnForward(torch::Tensor adj, torch::Tensor traits, torch::Tensor theta);
torch::Tensor FusedGcnSparseForward(torch::Tensor row_ptr, torch::Tensor col_indices, torch::Tensor values, torch::Tensor traits, torch::Tensor theta);
torch::Tensor FusedGcnBackward(torch::Tensor grad_out, torch::Tensor adj, torch::Tensor traits, int trait_dim, int out_dim);

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
// Fused Bayesian-Vegetation Aggregation (CUDA)
// ---------------------------------------------------------------------------
void launch_bayesian_veg_fusion(torch::Tensor images, torch::Tensor logits);

// ---------------------------------------------------------------------------
// Retinex Illumination Normalization (CUDA)
// ---------------------------------------------------------------------------
torch::Tensor retinex_normalize_cuda(
    torch::Tensor image, 
    std::vector<double> sigmas);

// ---------------------------------------------------------------------------
// Native Inference Engine (LibTorch)
// ---------------------------------------------------------------------------
void bind_inference_engine(py::module& m);

// ---------------------------------------------------------------------------
// Module Definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Inference Engine
    bind_inference_engine(m);

    // Retinex
    m.def("retinex_normalize", &retinex_normalize_cuda, "Multi-Scale Retinex illumination normalization (CUDA)");

    // Fused Loss
    m.def("fused_asl_forward", &fused_asl_forward, "Fused ASL Forward");
    m.def("fused_asl_backward", &fused_asl_backward, "Fused ASL Backward");

    // SAHI Tiling Engine
    m.def("sahi_tiling", &sahi_tiling_cuda, "Extract image tiles directly in CUDA using provided anchors");
    m.def("fused_aggregate_tiles", &fused_aggregate_cuda, "Fused GPU Aggregation across tiles");
    m.def("fused_bayesian_veg", &launch_bayesian_veg_fusion, "Fused Bayesian and ExG Vegetation weighting (CUDA)");

    // Ensemble Optimizations (GFAM)
    m.def("fused_gfam_projection", &fused_gfam_projection, "Fused Gated Feature Aggregation Projection (CUDA)");
    
    // Ecological GCN
    m.def("fused_gcn_forward", &FusedGcnForward, "Dense GCN Forward");
    m.def("fused_gcn_sparse_forward", &FusedGcnSparseForward, "Sparse CSR GCN Forward");
    m.def("fused_gcn_backward", &FusedGcnBackward, "GCN Backward");

    py::class_<StreamOrchestrator>(m, "StreamOrchestrator")
        .def(py::init<>())
        .def("synchronize", &StreamOrchestrator::synchronize)
        .def("get_stream", &StreamOrchestrator::get_stream);
}
