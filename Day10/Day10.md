# Day 10: Profiling & Performance Metrics

Understanding how to **profile CUDA applications** is essential for **optimizing performance**. Today, I explored **CUDA profiling tools**, **occupancy metrics**, and **performance analysis** to improve GPU computations.

## 1. Overview & Objectives

### **Objective:**
Learn how to **profile CUDA kernels and applications**, analyze **performance metrics**, and use these insights to optimize CUDA programs.

### **Key Learning Outcomes:**
- Use **NVIDIA profiling tools** to identify performance bottlenecks.
- Understand **occupancy metrics** and their impact on performance.
- Adjust kernel configurations based on profiling insights.

### **Real-World Application:**
Optimizing **financial computations**, where **high throughput and low latency** are crucial for fast decision-making in trading and risk analysis.

---

## 2. Key Concepts

### **Profiling Tools:**
- **NVIDIA Visual Profiler / Nsight:** Graphical tools providing insights into **kernel execution times, memory throughput, and occupancy**.
- **nvprof:** A command-line profiler for quickly displaying **kernel execution times** and other **performance metrics**.

### **Occupancy Metrics:**
- **Definition:** The ratio of **active warps per multiprocessor** to the **maximum possible active warps**.
- **Impact:** Higher occupancy often **improves GPU utilization**, but performance may still be limited by **memory bandwidth or instruction dependencies**.

### **Performance Analysis:**
- **Kernel Execution Time:** Measure how long each kernel runs.
- **Memory Throughput:** Evaluate **data transfer rates** between host and device.
- **Resource Utilization:** Check if **block size and grid configuration** are optimal.

---

## 3. Code Example: Vector Addition Kernel

This example demonstrates:
- **Profiling kernel execution time**.
- **Measuring occupancy metrics**.
- **Analyzing memory throughput**.

### **Code: `vectorAdd.cu`**

```cpp
// vectorAdd.cu: A simple vector addition kernel for profiling
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n = 1 << 20;  // 1 million elements
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define kernel execution configuration
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the first 10 results for verification
    printf("First 10 results of vector addition:\n");
    for (int i = 0; i < 10; i++) {
        printf("h_C[%d] = %f\n", i, h_C[i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```
### **Output:**
```
First 10 results of vector addition:
h_C[0] = 3.000000
h_C[1] = 3.000000
h_C[2] = 3.000000
h_C[3] = 3.000000
h_C[4] = 3.000000
h_C[5] = 3.000000
h_C[6] = 3.000000
h_C[7] = 3.000000
h_C[8] = 3.000000
h_C[9] = 3.000000

```

---

## 4. Profiling the Application

### **Compiling the Code:**
Compile with **NVIDIA CUDA Compiler (nvcc)**:
```bash
nvcc vectorAdd.cu -o vectorAdd
```

### **Profiling Using `nvprof` (Command-Line):**
Run the following command:
```bash
nvprof ./vectorAdd
```
This outputs:
- **Kernel Execution Time:** Time taken by `vectorAdd` kernel.
- **Achieved Occupancy:** Active warps vs. maximum possible warps.
- **Memory Throughput:** Data transfer rates between host and device.

### **Profiling Using NVIDIA Visual Profiler / Nsight:**
- Open **NVIDIA Visual Profiler or Nsight**.
- Load `vectorAdd` executable.
- Run a profiling session to analyze **kernel execution times, memory transfers, and resource utilization**.

---

## 5. Suggested Exercises

### **Exercise 1:**
- Use `nvprof` or **NVIDIA Visual Profiler** to **profile the vector addition kernel**.
- Record **kernel execution time, memory transfer times, and occupancy metrics**.

### **Exercise 2:**
- Experiment with different **block sizes** (e.g., **128, 256, 512**).
- Compare **occupancy and kernel execution time**.
- Identify the **optimal configuration**.

### **Exercise 3:**
- Optimize the **vector addition kernel** based on **profiling data**.
- **Reduce memory access latency** using **coalesced memory access** or **pinned memory**.
- Re-profile to **measure improvements**.

---
---

## Solutions to Exercises

### **Exercise 1: Profile the Kernel Execution Using `nvprof`**
Run the following command:
```bash
nvprof ./vectorAdd
```
Record:
- **Kernel Execution Time:** How long the `vectorAdd` kernel runs.
- **Memory Transfer Times:** Host-to-device and device-to-host transfer times.
- **Occupancy Metrics:** Active warps vs. maximum possible warps.

### **Exercise 2: Experiment with Different Block Sizes**
Modify the `blockSize` parameter in the execution configuration:
```cpp
int blockSize = 128; // Experiment with 128, 256, 512
int gridSize = (n + blockSize - 1) / blockSize;
```
Run the profiler again and compare:
- **Occupancy changes** with different block sizes.
- **Kernel execution time variations**.
- **Identify the optimal block size** for this workload.

### **Exercise 3: Optimize the Kernel Using Coalesced Memory Access**
Optimize memory access by ensuring **contiguous memory access patterns**:
```cpp
__global__ void optimizedVectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx]; // Ensure coalesced access
    }
}
```
Re-profile using `nvprof` to compare:
- **Reduction in memory access latency**.
- **Improved kernel execution time**.

---

---

## 6. Real-World Applications

### **Optimizing Financial Computations:**
- Used in **risk analysis, option pricing models**.
- Even **small performance improvements** can lead to **huge time savings**.

### **Deep Learning & AI Training:**
- **CUDA profiling helps optimize model training speed**.
- Ensures **efficient GPU utilization for AI workloads**.

---

## 7. Conclusion & Next Steps

### **Summary:**
- Learned how to **profile CUDA applications** using `nvprof` and **Visual Profiler**.
- Understood **occupancy metrics, memory throughput, and kernel execution time**.
- Used profiling tools to **identify performance bottlenecks**.

### **Next Steps:**
- **Apply profiling techniques** to optimize personal CUDA projects.
- **Experiment** with different **grid, block sizes, and memory strategies**.

### **Action Item:**
- Run the **vector addition code** and **profile it**.
- Document findings and **fine-tune CUDA applications** for better performance.

---

## Final Thoughts

Profiling is a **powerful tool** for **CUDA optimization**. Initially, it may seem complex, but once I started analyzing **kernel execution time, memory usage, and occupancy**, I realized how **small changes can lead to significant performance improvements**.

Using **NVIDIA profiling tools** helped me understand **how my CUDA code interacts with the GPU**. If you’re new to CUDA, **experiment with profiling, tweak your kernels, and measure the impact of optimizations**.

Excited to explore **advanced CUDA performance optimizations next! 🚀**
