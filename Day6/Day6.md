# Day 6: Memory Management Basics

CUDA programming involves efficient memory handling to ensure high performance. Today, I explored **memory management in CUDA**, including **allocating, transferring, and freeing memory** on the GPU.

## 1. Overview & Objectives

### Objective:
Understand how to allocate, transfer, and deallocate memory in a CUDA program.

### Key Learning Outcomes:
- Learn about different types of **memory on the GPU** (global, registers, local).
- Use **CUDA API functions** (`cudaMalloc`, `cudaMemcpy`, `cudaFree`) for memory management.
- Gain hands-on experience through a **code example** that transfers data to the GPU, processes it, and retrieves the results.

### Real-World Application:
Proper memory management is **crucial for computational simulations**, where large datasets are processed in parallel on GPUs.

---

## 2. Introduction to GPU Memory Types

### **Global Memory:**
- **Accessible by all threads** and persists for the lifetime of the application.
- Used for **storing large datasets** and exchanging data between the **host (CPU) and device (GPU).**

### **Registers:**
- **Fast, on-chip storage** used for frequently accessed variables.
- Managed **automatically** by the CUDA compiler.

### **Local Memory:**
- **Private to each thread**, but stored in **global memory** (making it slower than registers).
- Used when **register space is insufficient**.

> **Note:** While registers and local memory are important for performance, CUDA programmers explicitly manage **global memory**, which we focus on in this lesson.

---

## 3. Memory Allocation and Data Transfer in CUDA

### **Allocation:**
- Use `cudaMalloc` to allocate memory on the **device (GPU).**

### **Data Transfer:**
- Use `cudaMemcpy` to transfer data between the **host (CPU) and the device (GPU).**

### **Deallocation:**
- Free device memory with `cudaFree` to avoid **memory leaks.**

---

## 4. Code Example: Adding a Constant to an Array

This example helped me understand how to **allocate memory on the GPU, transfer data, perform a computation, and retrieve results.**

### **Code: `memory_management.cu`**

```cpp
// memory_management.cu: Demonstrate memory allocation, data transfer, and deallocation in CUDA

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel that adds a constant value to each element of an array
__global__ void addConstant(float *d_array, float constant, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_array[idx] += constant;
    }
}

int main() {
    int n = 1024;                      // Number of elements
    size_t size = n * sizeof(float);    // Memory required in bytes

    // Allocate host memory
    float *h_array = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        h_array[i] = (float)i;
    }

    // Allocate device (GPU) memory
    float *d_array;
    cudaMalloc((void **)&d_array, size);

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    float constant = 5.0f;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    addConstant<<<gridSize, blockSize>>>(d_array, constant, n);
    cudaDeviceSynchronize();

    // Copy the result back from device to host
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // Print first 10 elements
    printf("First 10 elements after adding %f:\n", constant);
    for (int i = 0; i < 10; i++) {
        printf("Element %d: %f\n", i, h_array[i]);
    }

    // Free memory
    cudaFree(d_array);
    free(h_array);

    return 0;
}
```

---

## 5. Detailed Explanation of the Code

### **Host Memory Allocation:**
- `malloc` is used to allocate memory on the **CPU.**
- The array is initialized with **incremental values.**

### **Device Memory Allocation:**
- `cudaMalloc` allocates memory on the **GPU.**

### **Data Transfer (Host to Device):**
- `cudaMemcpy` transfers data from **CPU to GPU.**

### **Kernel Execution:**
- The kernel `addConstant` is launched with **grid and block configurations**.
- Each thread adds `5.0` to its corresponding element.

### **Data Transfer (Device to Host):**
- `cudaMemcpy` copies the modified data **back to the CPU.**

### **Cleanup:**
- `cudaFree` frees the **device memory.**
- `free(h_array)` releases the **host memory.**

---

## 6. Suggested Exercises

### **Exercise 1:**
- Modify the kernel to **multiply** each element by a constant instead of adding one.

### **Exercise 2:**
- Experiment with **different array sizes** and observe memory transfer times.

### **Exercise 3:**
- Add **error-checking** to each CUDA API call for robust CUDA programming.

---

## 7. Real-World Applications

### **Computational Simulations:**
- Used in **weather forecasting** and **fluid dynamics** to process large datasets efficiently.

### **Data Processing Pipelines:**
- Common in **big data analytics** for transforming, filtering, and statistical computations.

---

## 8. Conclusion & Next Steps

### **Summary:**
- Learned about **GPU memory management.**
- Explored `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.
- Implemented **a simple CUDA program** for memory allocation and data transfer.

### **Preview of the Next Lesson:**
- Next, I will **optimize host-device data transfer** using **asynchronous transfers and pinned memory.**

### **Action Item:**
- Experiment with the **suggested exercises.**
- Try different **grid and block configurations.**

---

## Final Thoughts

Memory management in CUDA is one of the most **important concepts** for performance optimization. Initially, managing **GPU memory** seemed challenging, but after writing and debugging the code, I found it quite **logical and systematic.**

The ability to allocate, transfer, and free memory **efficiently** plays a major role in **large-scale computations**. I now realize why developers focus on **minimizing memory transfers** to improve GPU performance.

If you're starting out, my advice is to **practice a lot**—experiment with different memory sizes, **profile memory usage**, and **observe performance impacts**.

Excited for **Day 7**, where I'll explore **advanced data transfer techniques** to further **optimize performance!** 🚀
