# Day 17: Advanced Memory Management

CUDA **memory management** is crucial for **optimizing data transfers** and **minimizing overhead** in high-speed applications like **real-time data streaming**.

## 1. Overview & Objectives

### **Objective:**
Understand and apply **advanced memory management techniques** in CUDA to improve **data transfer performance**.

### **Key Learning Outcomes:**
- Learn how to use **pinned memory** (`cudaMallocHost()`) for **fast data transfers**.
- Understand **zero-copy memory** to allow **direct GPU access to host memory**.
- Utilize **unified memory** (`cudaMallocManaged()`) to **simplify programming**.

### **Real-World Application:**
These techniques are essential for **high-speed data streaming applications**, where **low-latency memory transfers** are required for **maximum efficiency**.

---

## 2. Key Concepts

### **Pinned (Page-Locked) Memory:**
- **Definition:** Host memory that is **locked in RAM** and **cannot be paged out**.
- **Advantages:**
  - **Faster memory transfers** via **Direct Memory Access (DMA)**.
  - **More predictable performance** as memory is not paged.
- **Usage:**
  - Allocated using `cudaMallocHost()`, freed with `cudaFreeHost()`.

### **Zero-Copy Memory:**
- **Definition:** Allows **direct device access** to **host memory** without explicit `cudaMemcpy()`.
- **Advantages:**
  - Eliminates **data transfer overhead**.
  - Simplifies programming by **providing shared memory between CPU and GPU**.
- **Usage:**
  - Allocated with `cudaHostAlloc()` (using `cudaHostAllocMapped`).
  - Retrieved with `cudaHostGetDevicePointer()`.

### **Unified Memory:**
- **Definition:** Provides a **single memory space** accessible by both **CPU and GPU**.
- **Advantages:**
  - **Simplifies memory management** by automatically migrating data.
  - **Reduces explicit memory copy calls**.
- **Usage:**
  - Allocated using `cudaMallocManaged()`.
  - Requires **fine-tuning** for high-performance applications.

---

## 3. Code Example: Comparing Pinned vs. Pageable Memory Transfers

This example demonstrates:
- **Allocating pinned vs. pageable memory**.
- **Measuring data transfer times**.
- **Showing how pinned memory improves performance**.

### **Code: `memoryManagementExample.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define NUM_ELEMENTS 1<<20  // 1M elements

__global__ void dummyKernel(float* d_out, int n) { }

void measureTransferTime(void* h_ptr, size_t size, const char* label) {
    float *d_data;
    cudaMalloc(&d_data, size);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_data, h_ptr, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    printf("%s transfer time: %f ms\n", label, elapsed);

    cudaFree(d_data);
}

int main() {
    size_t size = NUM_ELEMENTS * sizeof(float);

    float* h_pageable = (float*)malloc(size);
    float* h_pinned;
    cudaMallocHost(&h_pinned, size);

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_pageable[i] = 1.0f;
        h_pinned[i] = 1.0f;
    }

    measureTransferTime(h_pageable, size, "Pageable");
    measureTransferTime(h_pinned, size, "Pinned");

    free(h_pageable);
    cudaFreeHost(h_pinned);
    return 0;
}
```

---

## 4. Conclusion & Next Steps

### **Summary:**
- **Used pinned memory** for **faster data transfers**.
- **Discussed zero-copy and unified memory** for **simplified memory access**.
- **Measured transfer times** to show **performance gains with pinned memory**.

### **Next Steps:**
- **Experiment with zero-copy memory** and **measure performance improvements**.
- **Use CUDA profiling tools** to optimize memory usage.

### **Action Item:**
- **Modify the example to use unified memory** and **compare performance**.
- **Implement a data streaming application** using **pinned memory vs. pageable memory**.

### **Example 1: Comparing Pinned vs. Pageable vs. Unified Memory Transfers**

This example demonstrates:
- **Allocating pinned, pageable, and unified memory**.
- **Measuring data transfer times**.
- **Comparing performance differences**.

#### **Code: `memoryManagementExample.cu`**
```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define NUM_ELEMENTS 1<<20  // 1M elements

void measureTransferTime(void* h_ptr, size_t size, const char* label) {
    float *d_data;
    cudaMalloc(&d_data, size);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_data, h_ptr, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    printf("%s transfer time: %f ms\n", label, elapsed);

    cudaFree(d_data);
}

int main() {
    size_t size = NUM_ELEMENTS * sizeof(float);

    float* h_pageable = (float*)malloc(size);
    float* h_pinned;
    cudaMallocHost(&h_pinned, size);
    float* h_unified;
    cudaMallocManaged(&h_unified, size);

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_pageable[i] = 1.0f;
        h_pinned[i] = 1.0f;
        h_unified[i] = 1.0f;
    }

    measureTransferTime(h_pageable, size, "Pageable");
    measureTransferTime(h_pinned, size, "Pinned");
    measureTransferTime(h_unified, size, "Unified");

    free(h_pageable);
    cudaFreeHost(h_pinned);
    cudaFree(h_unified);
    return 0;
}
```

### **Example 2: Data Streaming Using Pinned vs. Pageable Memory**

#### **Code: `dataStreamingExample.cu`**
```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define NUM_ITERATIONS 100
#define NUM_ELEMENTS 1<<20  // 1M elements

void streamData(void* h_ptr, size_t size, const char* label) {
    float *d_data;
    cudaMalloc(&d_data, size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cudaMemcpyAsync(d_data, h_ptr, size, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    printf("%s streaming time: %f ms\n", label, elapsed);

    cudaFree(d_data);
}

int main() {
    size_t size = NUM_ELEMENTS * sizeof(float);

    float* h_pageable = (float*)malloc(size);
    float* h_pinned;
    cudaMallocHost(&h_pinned, size);

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_pageable[i] = 1.0f;
        h_pinned[i] = 1.0f;
    }

    streamData(h_pageable, size, "Pageable");
    streamData(h_pinned, size, "Pinned");

    free(h_pageable);
    cudaFreeHost(h_pinned);
    return 0;
}
```

---

## Final Thoughts

CUDA **memory management** plays a vital role in **GPU performance**. By using **pinned, zero-copy, and unified memory**, applications **reduce latency and improve efficiency**.

If you're learning CUDA, **experiment with different memory types**, measure performance, and **find the best approach for your workload**.

Looking forward to **more CUDA optimizations next! 🚀**
