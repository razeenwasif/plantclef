# Engines

This directory contains the high-performance, polyglot components of the Oracle stack that operate outside the Python GIL.

- **`cpp_cuda/`**: Custom C++ and CUDA extensions (like the Fused GFAM kernel, SAHI sliding-window tiling, and Retinex preprocessing) that maximize throughput on Blackwell Tensor Cores.
- **`rust/`**: SIMD-accelerated Rust engines responsible for data plane operations, including ultra-fast image resizing, WebDataset packing, and zero-copy metadata auditing using Polars.
- **`native/`**: Houses native implementations of classical AI algorithms (A* search, Minimax, K-Means) written in C++ and Haskell, serving as a foundation for broader symbolic reasoning tasks.