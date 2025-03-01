# Day 21: Introduction to Cooperative Groups

CUDA **cooperative groups** enable **flexible thread synchronization** beyond traditional `__syncthreads()`, making them ideal for **complex parallel patterns**.

## 1. Overview & Objectives

### **Objective:**
Understand how to use **cooperative thread groups in CUDA** to manage **synchronization and data sharing** efficiently.

### **Key Learning Outcomes:**
- Learn how to **form cooperative groups** and use **group-level synchronization**.
- Implement **cooperative groups in reduction kernels**.
- Optimize parallel algorithms using **fine-grained synchronization techniques**.

### **Real-World Application:**
Cooperative groups are useful in **distributed computing tasks**, where **multiple threads collaborate** efficiently for **complex operations** like **reduction and merging**.

---

## 2. Key Concepts

### **Cooperative Thread Groups:**
- **Definition:** A cooperative group is a **set of threads that perform collective operations** (e.g., reductions, scans).
- **Flexibility:** Unlike `__syncthreads()`, cooperative groups allow **subgroups (warp-level, block-level, or grid-level)** for **better communication**.

### **Enhanced Synchronization Techniques:**
- Cooperative groups **provide functions like `cg::sync(group)`** for **group-level synchronization**.
- They enable **more scalable synchronization strategies** beyond **block-wide barriers**.

### **Usage and Scope:**
- Ideal for **iterative reductions, merging operations, and parallel prefix sums**.
- **Improves modularity** and **clarity** in parallel algorithms.

---

## 3. Code Example: Reduction Kernel Using Cooperative Groups

This example demonstrates:
- **Creating cooperative groups** for **better synchronization**.
- **Optimizing reduction operations** using cooperative groups.

### **Code: `cudaCooperativeGroups.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Reduction kernel using cooperative groups
__global__ void cooperativeReduction(const float *d_in, float *d_out, int n) {
    extern __shared__ float sdata[];
    cg::thread_block cta = cg::this_thread_block();
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float mySum = 0.0f;
    
    if (i < n) mySum = d_in[i];
    if (i + blockDim.x < n) mySum += d_in[i + blockDim.x];
    sdata[tid] = mySum;
    
    cg::sync(cta);
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        cg::sync(cta);
    }
    
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

int main() {
    int n = 1 << 20;  // 1M elements
    size_t size = n * sizeof(float);
    float *h_in = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;
    
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    int blockSize = 256;
    int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
    cudaMalloc((void**)&d_out, gridSize * sizeof(float));
    
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    cooperativeReduction<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    float *h_out = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    float totalSum = 0.0f;
    for (int i = 0; i < gridSize; i++) totalSum += h_out[i];
    printf("Total sum: %f\n", totalSum);
    
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
```

---

## 4. Conclusion & Next Steps

### **Summary:**
- **Used cooperative groups** to **optimize synchronization** in parallel reductions.
- **Implemented fine-grained thread synchronization** using `cg::sync(cta)`.
- **Achieved better performance** with **more structured and scalable synchronization**.

### **Next Steps:**
- **Experiment with warp-level cooperative groups (`cg::tiled_partition<32>(cta)`)**.
- **Extend cooperative group techniques** to **parallel scan operations**.

### **Action Item:**
- **Rewrite a reduction or scan kernel using cooperative groups**.
- **Compare performance improvements** against **traditional `__syncthreads()` methods**.

---

## Final Thoughts

CUDA **cooperative groups** provide **advanced thread synchronization mechanisms**, making parallel algorithms **more efficient and scalable**.

For better performance, **experiment with different group levels** (warp, block, grid), **profile execution times**, and **optimize cooperative operations**.

Looking forward to **more CUDA performance optimizations next! 🚀**
