# Day 18: CUDA Streams & Asynchronous Execution

CUDA **streams** enable **concurrent execution** of memory transfers and kernel computations, allowing **higher throughput and reduced latency** in **real-time applications**.

## 1. Overview & Objectives

### **Objective:**
Learn to use **CUDA streams** to manage **asynchronous execution** of memory transfers and kernel launches.

### **Key Learning Outcomes:**
- Understand **how CUDA streams execute operations concurrently**.
- Use **asynchronous kernel launches and memory copies**.
- Overlap **computation and data transfer** to maximize GPU utilization.

### **Real-World Application:**
CUDA streams are **essential in real-time signal processing**, enabling **data movement and computation to occur concurrently** for **minimum latency**.

---

## 2. Key Concepts

### **CUDA Streams:**
- A **stream** is a sequence of operations (kernel launches, memory copies, etc.) that execute **in order** on the GPU.
- **Different streams can execute concurrently**, depending on **hardware capabilities**.

### **Asynchronous Kernel Launches & Memory Transfers:**
- **cudaMemcpyAsync()** allows **non-blocking memory copies**.
- **Kernel launches in different streams** can execute **simultaneously**.

### **Overlapping Operations:**
- **Concurrent memory transfers and kernel execution** hide **transfer latency**.
- **Ensures maximum GPU utilization** by executing **multiple workloads in parallel**.

---

## 3. Code Example: Overlapping Computation and Data Transfer with Multiple Streams

This example demonstrates:
- **Creating multiple CUDA streams**.
- **Processing independent data chunks in parallel**.
- **Overlapping memory transfers with kernel execution**.

### **Code: `cudaStreamsExample.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20; // 1M elements
    size_t size = n * sizeof(float);
    int half = n / 2;
    size_t halfSize = half * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemcpyAsync(d_a, h_a, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, halfSize, cudaMemcpyHostToDevice, stream1);
    int blockSize = 256;
    int gridSize = (half + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize, 0, stream1>>>(d_a, d_b, d_c, half);
    cudaMemcpyAsync(h_c, d_c, halfSize, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(d_a + half, h_a + half, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_b + half, h_b + half, halfSize, cudaMemcpyHostToDevice, stream2);
    vectorAdd<<<gridSize, blockSize, 0, stream2>>>(d_a + half, d_b + half, d_c + half, half);
    cudaMemcpyAsync(h_c + half, d_c + half, halfSize, cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) { correct = false; break; }
    }
    printf(correct ? "Results are correct!\n" : "Results are incorrect!\n");

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

---

## 4. Conclusion & Next Steps

### **Summary:**
- **Created multiple CUDA streams** for **concurrent execution**.
- **Overlapped memory transfers and kernel execution**.
- **Reduced memory transfer latency** by using **asynchronous operations**.

### **Next Steps:**
- **Use CUDA events** to measure **execution time** for each stream.
- **Increase the number of streams** to further **optimize execution efficiency**.

### **Action Item:**
- **Modify the example to process data in 3 or more streams**.
- **Measure throughput and latency improvements** using **CUDA profiling tools**.

---

## Final Thoughts

CUDA **streams** allow **parallel execution of kernels and memory transfers**, significantly **reducing latency** and **improving performance** in **real-time applications**.

By optimizing **stream concurrency**, CUDA applications can achieve **higher throughput** and **minimize idle time**. If you're learning CUDA, **experiment with streams and profile execution performance**.

Looking forward to **more CUDA optimizations next! 🚀**
