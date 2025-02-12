# Day 12: Memory Coalescing Techniques

CUDA **memory coalescing** is an essential optimization technique that ensures **efficient global memory access**, thereby **reducing latency** and **increasing throughput**.

## 1. Overview & Objectives

### **Objective:**
Understand the importance of **memory coalescing** in CUDA and **optimize kernel memory accesses** for better performance.

### **Key Learning Outcomes:**
- Learn how **global memory access patterns** impact performance.
- Implement **coalesced memory accesses** for **efficient memory transactions**.
- Avoid **non-coalesced memory accesses** that slow down execution.

### **Real-World Application:**
Optimized memory accesses are **crucial in big data processing**, where processing **large datasets efficiently** can significantly **improve overall performance**.

---

## 2. Key Concepts

### **Global Memory Access Patterns:**
- **Global memory** has **high latency**, and the way **threads access memory** affects performance.
- **Efficient memory transactions** depend on **properly aligned data access**.

### **Coalesced Memory Accesses:**
- **Definition:** When **consecutive threads in a warp** access **consecutive memory addresses**, the **GPU can combine these accesses** into a **single memory transaction**.
- **Benefits:**
  - **Reduces the number of memory transactions**, minimizing **latency**.
  - **Increases memory throughput**, leading to **faster execution**.

### **Factors Affecting Coalescing:**
- **Alignment:** Data should be **aligned on natural boundaries** (e.g., **32-, 64-, or 128-byte boundaries**) to **maximize memory coalescing**.
- **Access Pattern:** Ensure that **threads within the same warp** access **contiguous locations**. Non-contiguous (strided) accesses **increase memory transactions**, leading to **poor performance**.

---

## 3. Code Example: Non-Coalesced vs. Coalesced Memory Access

This example demonstrates:
- **A non-coalesced kernel** with **strided memory access**.
- **A coalesced kernel** with **contiguous memory access**.

### **Code: `memoryCoalescing.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel with non-coalesced access
__global__ void vectorCopyNonCoalesced(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Non-coalesced: each thread accesses memory with a stride.
        output[idx] = input[idx * 2];
    }
}

// Kernel with coalesced access
__global__ void vectorCopyCoalesced(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Coalesced: each thread accesses contiguous memory addresses.
        output[idx] = input[idx];
    }
}

int main() {
    int n = 1024; // Number of elements
    size_t size = n * sizeof(float);
    size_t sizeNonCoalesced = n * 2 * sizeof(float);

    float *h_input = (float*)malloc(sizeNonCoalesced);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < n * 2; i++) {
        h_input[i] = i * 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeNonCoalesced);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, sizeNonCoalesced, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    vectorCopyNonCoalesced<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Non-Coalesced Kernel Output (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    vectorCopyCoalesced<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("\nCoalesced Kernel Output (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
```

---

## 4. Conclusion & Next Steps

### **Summary:**
- **Implemented coalesced and non-coalesced memory accesses** in CUDA.
- **Analyzed the performance impact** of non-coalesced vs. coalesced memory access.
- **Ensured memory alignment** to maximize **efficient transactions**.

### **Next Steps:**
- **Use CUDA profiling tools** to analyze **memory throughput**.
- **Modify and optimize CUDA kernels** to use **coalesced memory accesses**.

### **Action Item:**
- Implement the **memory coalescing example**, compare execution times, and document improvements.
- Optimize **your own CUDA kernels** to leverage **memory coalescing techniques**.

### **Implementation and Execution Time Comparison:**
- The **non-coalesced kernel** takes **longer execution time** due to inefficient memory transactions.
- The **coalesced kernel** improves memory throughput and **reduces execution time**.

### **Optimizing CUDA Kernels for Memory Coalescing:**
- **Aligned memory access** for improved performance.
- **Thread-block configurations adjusted** to **ensure efficient memory reads**.
- **CUDA profiling tools (`nvprof`, Nsight)** used to verify improved throughput.

``` cpp

```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

// Kernel with non-coalesced access
__global__ void vectorCopyNonCoalesced(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx * 2];
    }
}

// Kernel with coalesced access
__global__ void vectorCopyCoalesced(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);
    size_t sizeNonCoalesced = n * 2 * sizeof(float);

    float *h_input = (float*)malloc(sizeNonCoalesced);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < n * 2; i++) {
        h_input[i] = i * 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeNonCoalesced);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, sizeNonCoalesced, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Timing the non-coalesced kernel
    auto start = std::chrono::high_resolution_clock::now();
    vectorCopyNonCoalesced<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> nonCoalescedTime = end - start;
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Timing the coalesced kernel
    start = std::chrono::high_resolution_clock::now();
    vectorCopyCoalesced<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> coalescedTime = end - start;
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Non-Coalesced Kernel Execution Time: %f ms\n", nonCoalescedTime.count());
    printf("Coalesced Kernel Execution Time: %f ms\n", coalescedTime.count());

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
```



---

## Final Thoughts

Understanding **memory coalescing** is fundamental for **high-performance CUDA programming**. Initially, managing **global memory access** may seem tricky, but **coalescing techniques significantly improve performance**.

By **ensuring contiguous memory access**, we can **reduce memory latency**, **increase memory throughput**, and **improve overall GPU efficiency**. If you're learning CUDA, **practice optimizing memory accesses, experiment with different patterns, and profile performance using CUDA tools**.

Excited to explore **advanced memory optimizations next! 🚀**