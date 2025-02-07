# Day 7: Host-Device Data Transfer

Understanding **how to efficiently transfer data** between the CPU (host) and GPU (device) is essential for **high-performance CUDA programming**. Today, I explored **synchronous and asynchronous memory transfers**, **pinned memory**, and **CUDA streams** to **optimize data movement**.

## 1. Overview & Objectives

### **Objective:**
Master techniques for moving data between the **host (CPU) and device (GPU)** efficiently. You will learn about **synchronous vs. asynchronous transfers** and how to **leverage pinned memory** for better performance.

### **Key Learning Outcomes:**
- Allocate **pinned memory** on the host.
- Perform **asynchronous data transfers** using **CUDA streams**.
- Ensure **data integrity** after round-trip transfers.

### **Real-World Application:**
These techniques are **critical in real-time video processing**, where fast and **reliable data transfers** are necessary to **maintain high throughput**.

---

## 2. Key Concepts and Functions

### **cudaMemcpy:**
- Standard function to **copy data between host and device**.
- Works **synchronously (blocking)** by default.

### **Pinned Memory:**
- Allocated using `cudaMallocHost` on the **host**.
- **Page-locked**, meaning it **cannot be swapped** out of memory, which improves **transfer speeds**.

### **Asynchronous Transfers:**
- Performed using `cudaMemcpyAsync` **with CUDA streams**.
- Allows **overlapping data transfer with computation**, reducing latency.

### **CUDA Streams:**
- Queues that **manage asynchronous operations**.
- Allow **multiple transfers and kernel launches** to run **concurrently**.

---

## 3. Code Example: Asynchronous Data Transfer Using Pinned Memory

This example demonstrates:
- Allocating **pinned memory** on the host.
- Allocating **device memory**.
- **Asynchronously transferring** an array from host to device.
- Copying data back **asynchronously** to another host array.
- **Verifying data integrity** after transfers.

### **Code: `host_device_transfer.cu`**

```cpp
// host_device_transfer.cu: Demonstrates host-device data transfer using pinned memory and asynchronous transfers

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Macro for error checking
#define cudaCheckError(call)                                                       \
    do {                                                                           \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", __func__, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                            \
            exit(err);                                                             \
        }                                                                          \
    } while (0)

int main() {
    const int n = 1024;                   // Number of elements
    size_t size = n * sizeof(float);      // Memory size in bytes

    float *h_src = NULL;    // Source array (pinned memory)
    float *h_dest = NULL;   // Destination array (pinned memory)
    float *d_array = NULL;  // GPU memory

    // Allocate pinned host memory
    cudaCheckError(cudaMallocHost((void**)&h_src, size));
    cudaCheckError(cudaMallocHost((void**)&h_dest, size));

    // Initialize the source array
    for (int i = 0; i < n; i++) {
        h_src[i] = (float)i;
    }

    // Allocate GPU memory
    cudaCheckError(cudaMalloc((void**)&d_array, size));

    // Create a CUDA stream
    cudaStream_t stream;
    cudaCheckError(cudaStreamCreate(&stream));

    // Asynchronous host-to-device transfer
    cudaCheckError(cudaMemcpyAsync(d_array, h_src, size, cudaMemcpyHostToDevice, stream));

    // (Optional) Insert GPU computation here while the transfer is happening

    // Asynchronous device-to-host transfer
    cudaCheckError(cudaMemcpyAsync(h_dest, d_array, size, cudaMemcpyDeviceToHost, stream));

    // Wait for all operations to complete
    cudaCheckError(cudaStreamSynchronize(stream));

    // Verify data integrity
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_src[i] != h_dest[i]) {
            errors++;
        }
    }

    printf(errors == 0 ? "Data transfer successful!\n" : "Data transfer error: %d mismatches found.\n", errors);

    // Cleanup
    cudaFree(d_array);
    cudaFreeHost(h_src);
    cudaFreeHost(h_dest);
    cudaStreamDestroy(stream);

    return 0;
}
```

```cpp

// host_device_transfer_with_compute.cu: Overlapping computation with asynchronous host-device transfers

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Macro for error checking
#define cudaCheckError(call)                                                   \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", __func__,        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(err);                                                         \
        }                                                                      \
    } while (0)

// Kernel that adds a constant to each element of the array
__global__ void addConstantKernel(float *data, float constant, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += constant;
    }
}

int main() {
    const int n = 1024;                     // Number of elements in the array
    size_t size = n * sizeof(float);          // Total size in bytes

    float *h_src = NULL;    // Host source array (pinned memory)
    float *h_dest = NULL;   // Host destination array (pinned memory)
    float *d_array = NULL;  // Device array

    // Allocate pinned (page-locked) host memory for faster transfers
    cudaCheckError(cudaMallocHost((void**)&h_src, size));
    cudaCheckError(cudaMallocHost((void**)&h_dest, size));

    // Initialize the host source array with values 0, 1, 2, ..., n-1
    for (int i = 0; i < n; i++) {
        h_src[i] = (float)i;
    }

    // Allocate device memory
    cudaCheckError(cudaMalloc((void**)&d_array, size));

    // Create two CUDA streams:
    // - transferStream handles asynchronous memory copies.
    // - computeStream handles kernel execution.
    cudaStream_t transferStream, computeStream;
    cudaCheckError(cudaStreamCreate(&transferStream));
    cudaCheckError(cudaStreamCreate(&computeStream));

    // Asynchronously copy data from host to device using transferStream
    cudaCheckError(cudaMemcpyAsync(d_array, h_src, size, cudaMemcpyHostToDevice, transferStream));

    // Create an event to signal the completion of the host-to-device copy
    cudaEvent_t copyComplete;
    cudaCheckError(cudaEventCreate(&copyComplete));
    cudaCheckError(cudaEventRecord(copyComplete, transferStream));

    // Make computeStream wait until the host-to-device copy is finished
    cudaCheckError(cudaStreamWaitEvent(computeStream, copyComplete, 0));

    // Define kernel launch parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    float constant = 10.0f;

    // Launch the kernel in computeStream to add the constant to each element
    addConstantKernel<<<gridSize, blockSize, 0, computeStream>>>(d_array, constant, n);

    // Create an event to signal the completion of the kernel execution
    cudaEvent_t kernelComplete;
    cudaCheckError(cudaEventCreate(&kernelComplete));
    cudaCheckError(cudaEventRecord(kernelComplete, computeStream));

    // Make transferStream wait until the kernel execution is complete before copying data back
    cudaCheckError(cudaStreamWaitEvent(transferStream, kernelComplete, 0));

    // Asynchronously copy the modified data from device back to host using transferStream
    cudaCheckError(cudaMemcpyAsync(h_dest, d_array, size, cudaMemcpyDeviceToHost, transferStream));

    // Synchronize both streams to ensure all operations have completed
    cudaCheckError(cudaStreamSynchronize(transferStream));
    cudaCheckError(cudaStreamSynchronize(computeStream));

    // Verify that each element has been correctly incremented by the constant
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_dest[i] != h_src[i] + constant) {
            errors++;
        }
    }
    if (errors == 0) {
        printf("Overlapped computation and transfer successful! Data is correct.\n");
    } else {
        printf("Data verification failed with %d errors.\n", errors);
    }

    // Cleanup: Free device memory, pinned host memory, streams, and events
    cudaFree(d_array);
    cudaFreeHost(h_src);
    cudaFreeHost(h_dest);
    cudaStreamDestroy(transferStream);
    cudaStreamDestroy(computeStream);
    cudaEventDestroy(copyComplete);
    cudaEventDestroy(kernelComplete);

    return 0;
}


```

---

## 4. Detailed Explanation of the Code

### **Pinned Host Memory Allocation:**
- `cudaMallocHost` is used to **allocate page-locked memory**, improving transfer speeds.

### **Device Memory Allocation:**
- `cudaMalloc` allocates space on the **GPU** for computation.

### **Asynchronous Data Transfer:**
- `cudaMemcpyAsync` schedules both **host-to-device and device-to-host** transfers in a CUDA stream.
- `cudaStreamSynchronize` ensures **all transfers complete** before proceeding.

### **Data Verification:**
- The host arrays (`h_src` and `h_dest`) are compared to **confirm data integrity**.

### **Cleanup:**
- Device memory, pinned host memory, and CUDA streams are **freed properly** to prevent memory leaks.

---

## 5. Suggested Exercises

### **Exercise 1:**
- Modify the code to **perform a computation** on the GPU before transferring data back.

### **Exercise 2:**
- Compare **synchronous vs. asynchronous** transfers using **CUDA events** for timing.

### **Exercise 3:**
- Extend the code to **transfer a 2D image** (e.g., a grayscale image) and verify pixel data integrity.

---

## 6. Real-World Applications

### **Real-Time Video Processing:**
- Fast **host-to-device transfers** enable **real-time video analysis** and filtering.
- Used in **AI-powered video enhancement, motion tracking, and frame processing**.

### **Big Data Analytics:**
- Efficient data transfers help **train deep learning models** with large datasets.
- Asynchronous processing improves **data ingestion for real-time analytics**.

---

## 7. Conclusion & Next Steps

### **Summary:**
- Learned how to **transfer data efficiently** between CPU and GPU.
- Used **pinned memory** to speed up transfers.
- Implemented **asynchronous transfers** with **CUDA streams**.

### **Preview of Next Lessons:**
- In upcoming lessons, I will explore **memory optimization techniques** and **overlapping computation with data movement**.

### **Action Item:**
- Experiment with **different transfer sizes**.
- Compare **synchronous vs. asynchronous transfers**.
- Implement **data processing operations** between transfers.

---

## Final Thoughts

Optimizing **host-device transfers** is essential for **high-performance GPU applications**. Using **pinned memory and asynchronous transfers** significantly **reduces transfer overhead** and enables **overlapping data movement with computations**.

This lesson helped me **appreciate the power of CUDA streams** in improving performance. If you're learning CUDA, **practice modifying memory transfers**, experiment with **different stream configurations**, and **analyze performance improvements**.

Excited to dive into **further optimization techniques** in the next lesson! 🚀
