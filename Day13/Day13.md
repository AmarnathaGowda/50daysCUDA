# Day 13: Synchronization & Barriers

CUDA **synchronization** is essential for **coordinating threads within a block**, ensuring **correct memory operations** and **preventing race conditions**.

## 1. Overview & Objectives

### **Objective:**
Learn how to **synchronize threads within a CUDA block** using **barrier synchronization (`__syncthreads()`)** to ensure correct memory access ordering.

### **Key Learning Outcomes:**
- Understand how **`__syncthreads()`** works to synchronize **threads within a block**.
- Implement **barrier synchronization** in **parallel reduction kernels**.
- Learn **best practices** for avoiding **deadlocks and race conditions**.

### **Real-World Application:**
Used in **parallel sorting, reductions, and prefix sum computations**, where **synchronized access to shared data is critical**.

---

## 2. Understanding Synchronization in CUDA

### **Barrier Synchronization with `__syncthreads()`**:
- In CUDA, **threads within a block** can be synchronized using `__syncthreads()`.
- When a thread reaches `__syncthreads()`, it **waits until all other threads in the block** reach the barrier before continuing execution.
- Ensures that **shared memory writes are visible** before any thread proceeds with its next phase.

### **Pitfalls & Best Practices:**
- **Avoid Conditional Barriers:** Do **not place `__syncthreads()` inside conditional statements** unless all threads in the block follow the same execution path.
- **Synchronize Before Using Shared Memory:** Ensure that all threads **finish writing to shared memory** before reading from it.

---

## 3. Code Example: Parallel Reduction Kernel Using `__syncthreads()`

This example demonstrates:
- **How to use shared memory** for **parallel reduction**.
- **Synchronizing threads** at each step to avoid **race conditions**.
- **Efficiently computing the sum of an array** using **CUDA reduction**.

### **Code: `reduceSum.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for parallel reduction (sum of an array)
__global__ void reduceSum(float *d_in, float *d_out, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float mySum = 0.0f;
    
    if (i < n)
        mySum = d_in[i];
    if (i + blockDim.x < n)
        mySum += d_in[i + blockDim.x];
    
    sdata[tid] = mySum;
    __syncthreads();  // Ensure all threads have written to shared memory
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Synchronize threads before next reduction step
    }
    
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

int main() {
    const int numElements = 1024;
    size_t size = numElements * sizeof(float);
    
    float *h_in = (float*)malloc(size);
    for (int i = 0; i < numElements; i++) {
        h_in[i] = 1.0f;
    }
    
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    int blockSize = 256;
    int gridSize = (numElements + blockSize * 2 - 1) / (blockSize * 2);
    cudaMalloc((void**)&d_out, gridSize * sizeof(float));
    
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    reduceSum<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, numElements);
    cudaDeviceSynchronize();
    
    float *h_out = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    float totalSum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        totalSum += h_out[i];
    }
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
- **Used `__syncthreads()` for thread synchronization** in CUDA.
- **Implemented parallel reduction** using shared memory and **barrier synchronization**.
- **Ensured proper memory access ordering** to **avoid race conditions**.

### **Next Steps:**
- **Experiment with different block sizes** and **optimizations for reduction algorithms**.
- **Use CUDA profiling tools** to analyze **synchronization overhead**.

### **Action Item:**
- Modify the **parallel reduction kernel** to compute **other operations (e.g., max, min, product)**.
- **Explore warp-level synchronization** techniques like `__syncwarp()` for further optimizations.

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for parallel reduction (sum of an array)
__global__ void reduceSum(float *d_in, float *d_out, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float mySum = 0.0f;
    
    if (i < n)
        mySum = d_in[i];
    if (i + blockDim.x < n)
        mySum += d_in[i + blockDim.x];
    
    sdata[tid] = mySum;
    __syncthreads();  // Ensure all threads have written to shared memory
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Synchronize threads before next reduction step
    }
    
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

// Kernel for parallel maximum reduction
__global__ void reduceMax(float *d_in, float *d_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float myMax = (i < n) ? d_in[i] : -FLT_MAX;
    if (i + blockDim.x < n)
        myMax = max(myMax, d_in[i + blockDim.x]);
    sdata[tid] = myMax;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

int main() {
    const int numElements = 1024;
    size_t size = numElements * sizeof(float);
    
    float *h_in = (float*)malloc(size);
    for (int i = 0; i < numElements; i++) {
        h_in[i] = (float)(rand() % 100);
    }
    
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    int blockSize = 256;
    int gridSize = (numElements + blockSize * 2 - 1) / (blockSize * 2);
    cudaMalloc((void**)&d_out, gridSize * sizeof(float));
    
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    reduceMax<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, numElements);
    cudaDeviceSynchronize();
    
    float *h_out = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(h_out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    float maxVal = -FLT_MAX;
    for (int i = 0; i < gridSize; i++) {
        maxVal = max(maxVal, h_out[i]);
    }
    printf("Max value: %f\n", maxVal);
    
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
```

---

## Final Thoughts

CUDA **synchronization** is crucial for **parallel algorithms**. Using `__syncthreads()` correctly **prevents race conditions** and **ensures correct execution order**.

By **optimizing shared memory access**, we can **improve performance** in **reduction, sorting, and parallel algorithms**. If you're learning CUDA, **experiment with synchronization patterns** and **test performance impacts** using profiling tools.

Looking forward to **advanced CUDA optimizations next! 🚀**
