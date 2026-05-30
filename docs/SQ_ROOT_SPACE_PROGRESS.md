# Extreme Memory-Constrained Computing: Square-Root Space Logic Evaluation
**Project Tracker & Research Roadmap**

This document tracks our implementation of R. Ryan Williams' "Simulating Time With Square-Root Space" (STOC 2025) applied to Neuro-Symbolic Gradient Computation. Our objective is to shatter the memory limitations of backpropagating through deep logic graphs on consumer GPUs like the RTX 4090.

## 🗺️ The Roadmap

### Phase 1: The Theoretical Foundation & Python Proof-of-Concept (PoC)
**Goal:** Prove that we can partition a sequential computation (mock reasoning chain) into blocks, map it to a tree, and recompute gradients selectively without caching the entire forward state.
- [ ] **Task 1.1:** Define a dummy Neuro-Symbolic "reasoning chain" (a sequence of differentiable logic operations).
- [ ] **Task 1.2:** Implement a standard PyTorch `autograd.Function` baseline to measure VRAM consumption of the naive approach.
- [ ] **Task 1.3:** Design the **Block Partitioner**: Split the reasoning chain into chunks.
- [ ] **Task 1.4:** Implement the **Tree Evaluator (Python)**: A custom `autograd.Function` that only saves boundary states and selectively re-executes blocks during the backward pass.
- [ ] **Task 1.5:** Validate gradients match the baseline and measure the theoretical memory reduction.

### Phase 2: C++ / CUDA Architecture Translation
**Goal:** Translate the Python PoC into a high-performance C++ backend. Python overhead for dynamic tree traversal during `backward()` will bottleneck training.
- [ ] **Task 2.1:** Implement the logic block structures as C++ ATen operations.
- [ ] **Task 2.2:** Build the Tree Mapping scheduler in C++ to manage minimal memory buffers for boundary states.
- [ ] **Task 2.3:** Bind the C++ engine to PyTorch via PyBind11.

### Phase 3: The CUDA Tree Evaluator Kernel
**Goal:** Achieve maximum hardware satiation by writing custom CUDA kernels for the tree evaluation.
- [ ] **Task 3.1:** Write a CUDA kernel that parallelizes the recomputation of disjoint tree branches during the backward pass.
- [ ] **Task 3.2:** Implement dynamic shared memory allocation to ensure boundary states fit within the SM's L1 cache, avoiding global memory (HBM) latency.

### Phase 4: Integration with plantclef
**Goal:** Hook the engine into our Probabilistic Logic Programming pipeline to enable incredibly deep ecological constraint reasoning.
- [ ] **Task 4.1:** Integrate the Square-Root Space evaluator into the plantclef AC-3/Loopy BP loss functions.
- [ ] **Task 4.2:** Benchmark the maximum reasoning depth achievable on the 24GB RTX 4090 before and after implementation.

---

## 🧪 Current Focus: Phase 1 (Python PoC)

**Immediate Next Steps:**
We will create a script in `engines/native/classical_ai/python/` (or a similar experimental directory) to build our baseline and our custom `SquareRootSpaceAutograd` class.

**Conceptual Sketch for Task 1.4:**
```python
import torch

class SquareRootSpaceEvaluator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, logic_blocks):
        """
        1. Partition the forward pass into sqrt(T) blocks.
        2. Only save the 'boundary' tensors (inputs to each block) into ctx.
        3. Discard all intermediate activations within the blocks.
        """
        boundaries = [x.detach()]
        current = x
        for block in logic_blocks:
            with torch.no_grad():
                current = block(current)
                boundaries.append(current.detach())
        
        ctx.boundaries = boundaries
        ctx.logic_blocks = logic_blocks
        return current

    @staticmethod
    def backward(ctx, grad_output):
        """
        1. Retrieve the boundary tensors.
        2. Iteratively recompute each block locally (with autograd enabled)
           to get local gradients, preventing massive graph construction.
        3. Chain the gradients backwards.
        """
        boundaries = ctx.boundaries
        blocks = ctx.logic_blocks
        
        grad = grad_output
        for i in reversed(range(len(blocks))):
            block_input = boundaries[i].requires_grad_(True)
            with torch.enable_grad():
                block_output = blocks[i](block_input)
            
            # Recompute backward for just this block
            block_output.backward(grad)
            grad = block_input.grad
            
        return grad, None
```
