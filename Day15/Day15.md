# Day 15: Using Constant Memory

CUDA **constant memory** is a read-only memory space that is **cached** and **optimized for frequent access** by multiple threads. It is especially useful when **all threads access the same data**, leading to **reduced global memory latency**.

## 1. Overview & Objectives

### **Objective:**
Learn how to **declare and use constant memory** in CUDA to store and access **read-only data efficiently**.

### **Key Learning Outcomes:**
- Understand how to **declare constant memory arrays** and transfer data to **constant memory**.
- Learn how **constant memory caching works** for **faster data access**.
- Implement **constant memory usage in CUDA kernels** for **optimized performance**.

### **Real-World Application:**
Constant memory is widely used in **graphics shader programming**, storing **transformation matrices, color lookup tables**, and other frequently accessed **read-only parameters**.

---

## 2. Key Concepts

### **Declaration of Constant Memory:**
- Declared using the `__constant__` keyword.
- Designed for **data that remains unchanged** during kernel execution.

### **Caching Behavior:**
- **Cached on each Streaming Multiprocessor (SM)** for fast access.
- **High efficiency when multiple threads read the same data simultaneously**.

### **Data Transfer:**
- Data is copied from **host to device** using `cudaMemcpyToSymbol()`.
- **Constant memory is read-only** within device kernels.

---

## 3. Code Example: Using Constant Memory in a Sample Kernel

This example demonstrates:
- **Declaring and using constant memory** in CUDA.
- **Optimizing data access** using **constant memory caching**.

### **Code: `constantMemoryExample.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Declare a constant memory array of 256 floats
__constant__ float constData[256];

// Kernel that uses constant memory to scale each element of an input array
__global__ void scaleWithConstantKernel(const float *d_in, float *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Multiply each element by the first constant factor
        d_out[idx] = d_in[idx] * constData[0];
    }
}

int main() {
    const int n = 1024;
    size_t size = n * sizeof(float);
    
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_in[i] = 1.0f;
    }
    
    float h_const[256];
    for (int i = 0; i < 256; i++) {
        h_const[i] = 2.0f;
    }
    
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constData, h_const, 256 * sizeof(float));
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    scaleWithConstantKernel<<<gridSize, blockSize>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    printf("First 10 output elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_out[i]);
    }
    printf("\n");
    
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    
    return 0;
}
```

---

## 4. Conclusion & Next Steps

### **Summary:**
- **Implemented and used constant memory** in CUDA.
- **Optimized kernel execution** by leveraging **constant memory caching**.
- **Reduced global memory access latency** by storing **frequently accessed data** in constant memory.

### **Next Steps:**
- **Experiment with different constant memory usage patterns** in CUDA applications.
- **Use CUDA profiling tools** to compare performance **with and without constant memory**.

### **Action Item:**
- **Modify the kernel** to use different **constant memory elements** based on **thread indices**.
- **Measure execution times and analyze performance improvements**.

---

## Final Thoughts

CUDA **constant memory** is an excellent tool for **storing frequently accessed read-only data**. By using **constant memory caching**, we can **significantly reduce memory latency and improve performance**.

By applying **constant memory techniques**, CUDA applications **run faster and consume fewer memory resources**. If you're learning CUDA, **experiment with constant memory usage and compare performance gains** using profiling tools.

Looking forward to **more CUDA optimizations next! 🚀**