#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

/**
 * @file fused_gcn.cu
 * @brief High-performance Ecological Graph Convolutional Network (GCN) operations.
 * 
 * ORACLE: Hardened for Multi-GPU Stability.
 * Removed L2 Cache Persistence to prevent alignment-driven CUDA errors on clusters.
 */

/**
 * @brief Forward pass for Ecological GCN using cuBLAS.
 */
torch::Tensor FusedGcnForward(
    torch::Tensor adj,
    torch::Tensor traits,
    torch::Tensor theta) {
  
  // Intermediary: H = Traits @ Theta
  // We use the ATen high-level API to ensure proper cuBLAS dispatching
  auto h = at::mm(traits, theta);
  
  // Result: Out = Adj @ H
  return at::mm(adj, h);
}

/**
 * @brief Sparse Forward pass for Ecological GCN using cuSparse.
 */
torch::Tensor FusedGcnSparseForward(
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    torch::Tensor values,
    torch::Tensor traits,
    torch::Tensor theta) {
  
  int num_species = row_ptr.size(0) - 1;
  auto adj_sparse = at::sparse_csr_tensor(
      row_ptr, col_indices, values, 
      {num_species, num_species}, 
      values.options());

  auto h = at::mm(traits, theta);
  return at::_sparse_mm(adj_sparse, h);
}

/**
 * @brief Backward pass for Ecological GCN.
 */
torch::Tensor FusedGcnBackward(
    torch::Tensor grad_out,
    torch::Tensor adj,
    torch::Tensor traits,
    int trait_dim,
    int out_dim) {
  
  // grad_theta = traits.T @ (adj.T @ grad_out)
  auto grad_h = at::mm(adj.t(), grad_out);
  return at::mm(traits.t(), grad_h);
}
