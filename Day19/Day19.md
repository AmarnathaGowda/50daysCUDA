# Day 19: Concurrent Kernel Execution

CUDA **concurrent kernel execution** allows multiple kernels to run **simultaneously**, maximizing **GPU utilization** and **reducing execution time**.

## 1. Overview & Objectives

### **Objective:**
Learn how to **launch kernels concurrently using CUDA streams** to **overlap execution of independent workloads**.

### **Key Learning Outcomes:**
- Understand how **CUDA streams enable concurrent execution**.
- Learn to **launch multiple kernels** in different streams.
- Use **CUDA synchronization functions** to ensure proper execution.

### **Real-World Application:**
Concurrent kernel execution is ideal for **simulations**, where **different components of a simulation** can be computed **in parallel**.

---

## 2. Key Concepts

### **CUDA Streams:**
- A **stream** is a sequence of operations (kernel launches, memory transfers) executed **in order**.
- **Operations in different streams can execute concurrently**, provided they are **independent**.

### **Overlapping Kernel Execution:**
- **Multiple kernels launched in different streams can execute concurrently**, depending on **GPU resources**.
- **Reduces total execution time** for applications with **independent tasks**.

### **Multi-Stream Management:**
- Requires **proper creation and synchronization**.
- **Use `cudaStreamCreate()` to create** separate execution streams.
- **Use `cudaStreamSynchronize()` to ensure execution completion**.

---

## 3. Code Example: Concurrent Execution with Two Independent Kernels

This example demonstrates:
- **Launching two independent kernels concurrently**.
- **Using separate CUDA streams** to allow **parallel execution**.

### **Code: `cudaConcurrentKernels.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel A: Adds 1.0f to each element in the array
__global__ void kernelA(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000; i++) {
            data[idx] += 1.0f;
        }
    }
}

// Kernel B: Multiplies each element in the array by 1.01f
__global__ void kernelB(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000; i++) {
            data[idx] *= 1.01f;
        }
    }
}

int main() {
    int n = 1 << 20; // 1M elements
    size_t size = n * sizeof(float);

    float *d_dataA, *d_dataB;
    cudaMalloc((void**)&d_dataA, size);
    cudaMalloc((void**)&d_dataB, size);

    cudaMemset(d_dataA, 0, size);
    cudaMemset(d_dataB, 1, size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    kernelA<<<gridSize, blockSize, 0, stream1>>>(d_dataA, n);
    kernelB<<<gridSize, blockSize, 0, stream2>>>(d_dataB, n);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    float *h_dataA = (float*)malloc(size);
    float *h_dataB = (float*)malloc(size);
    cudaMemcpy(h_dataA, d_dataA, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataB, d_dataB, size, cudaMemcpyDeviceToHost);

    printf("Kernel A result sample: %f\n", h_dataA[0]);
    printf("Kernel B result sample: %f\n", h_dataB[0]);

    cudaFree(d_dataA);
    cudaFree(d_dataB);
    free(h_dataA);
    free(h_dataB);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
```

---

## 4. Conclusion & Next Steps

### **Summary:**
- **Used multiple CUDA streams** to launch **independent kernels** concurrently.
- **Overlapped execution to reduce total processing time**.
- **Ensured execution completion using `cudaStreamSynchronize()`**.

### **Next Steps:**
- **Experiment with launching more than two kernels** in **separate streams**.
- **Use CUDA events** to **profile execution time** for each kernel.

### **Action Item:**
- **Modify the example to process three or more kernels in different streams**.
- **Measure performance improvements** using **CUDA profiling tools**.

---

## Final Thoughts

CUDA **concurrent kernel execution** significantly **improves performance** by **overlapping computations** and **maximizing GPU resource usage**.

By **optimizing stream concurrency**, CUDA applications can achieve **higher throughput** and **reduce execution bottlenecks**. If you're learning CUDA, **experiment with streams and analyze execution performance**.

Looking forward to **more CUDA optimizations next! 🚀**
