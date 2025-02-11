# Day 11: Introduction to Shared Memory

CUDA **shared memory** is a small, fast **on-chip memory** that helps **reduce global memory accesses**, improving performance in **compute-intensive tasks** like **matrix operations**.

## 1. Overview & Objectives

### **Objective:**
Understand how to **leverage shared memory** to boost CUDA performance, avoid **bank conflicts**, and optimize **data access patterns**.

### **Key Learning Outcomes:**
- Learn **why shared memory is faster** than global memory.
- Understand **bank conflicts** and how to **avoid them**.
- Implement **shared memory in CUDA kernels** for **better performance**.

### **Real-World Application:**
Used in **scientific computing, image processing**, and **deep learning**, where **matrix operations** must be computed **efficiently**.

---

## 2. Key Concepts

### **Shared Memory vs. Global Memory:**
- **Global Memory:** **Larger** but **slower** (~400-600 clock cycles per access).
- **Shared Memory:** **Faster (~1-2 clock cycles)** but **smaller** (limited to **48 KB per SM** on most GPUs).
- **Use Case:** Load **frequently used data** into shared memory to **avoid redundant accesses** to global memory.

### **Bank Conflicts:**
- **What are Bank Conflicts?**
  - Shared memory is divided into **banks**, which can be accessed **simultaneously**.
  - If **multiple threads** access **different addresses** in the **same bank**, they must be **serialized**, slowing down execution.
- **Avoiding Bank Conflicts:**
  - Proper **data alignment** and **padding** can **minimize conflicts**.
  - Using a **tiled memory access pattern** improves performance.

### **Usage Strategies:**
- Load **frequently used data** into shared memory (**tiling technique**).
- Ensure **different threads access different banks** whenever possible.
- Use **synchronization (`__syncthreads()`)** to ensure correct computation.

---

## 3. Code Example: Tiled Matrix Multiplication Using Shared Memory

This example demonstrates:
- **How to use shared memory** for **efficient matrix multiplication**.
- **Avoiding bank conflicts** using **proper indexing**.
- **Minimizing global memory accesses** with **tiling**.

### **Code: `matrixMulShared11.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication using shared memory tiling
__global__ void matrixMulShared(float *A, float *B, float *C, int width) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0f;

    for (int t = 0; t < (width + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < width && t * TILE_WIDTH + threadIdx.x < width)
            sA[threadIdx.y][threadIdx.x] = A[row * width + t * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < width && t * TILE_WIDTH + threadIdx.y < width)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * width + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = value;
}

int main() {
    int width = 32;
    size_t size = width * width * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);
    
    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result matrix C:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%0.2f ", h_C[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

---

## 4. Conclusion & Next Steps

### **Summary:**
- **Used shared memory** to **optimize memory access** in **matrix multiplication**.
- **Avoided bank conflicts** with **proper indexing** and **padding techniques**.
- **Improved performance** by **reducing global memory accesses**.

### **Next Steps:**
- **Experiment with different TILE_WIDTH values** to optimize performance.
- **Use CUDA profiling tools** (`nvprof`, `Nsight`) to analyze **shared memory performance**.

### **Action Item:**
- Implement the **matrix multiplication example**, modify **TILE_WIDTH**, and **measure performance gains**.
- Compare execution time **with and without shared memory** to see the impact.

---

## Final Thoughts

Understanding **shared memory** is key to **high-performance CUDA programming**. At first, managing shared memory may seem tricky, but once I started experimenting, I saw **huge speed improvements** over **global memory access**.

Using **tiling strategies** in **matrix multiplication** significantly reduces **redundant memory accesses**. If you're learning CUDA, **try optimizing different workloads with shared memory**, experiment with **memory layouts**, and **use profiling tools** to measure the impact.

Excited to explore **more CUDA memory optimizations next! 🚀**
