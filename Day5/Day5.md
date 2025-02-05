# Day 5: CUDA Thread Hierarchy

CUDA programming enables massive parallel execution, and today, I explored **thread hierarchy** in CUDA. It was fascinating to see how **threads, blocks, and grids** work together to process data in parallel efficiently.

## 1. Overview & Objectives

### Objective:
Understand the CUDA execution configuration by exploring threads, blocks, and grids. Learn how to calculate thread indices and configure kernel launches for optimal performance.

### Key Learning Outcome:
- Be able to write and configure a CUDA kernel using a **1D grid and blocks**.
- Understand how to use **thread indexing** for data processing.

### Real-World Application:
This knowledge is crucial for tasks like **data analytics**, where each data element is processed independently using parallel execution.

---

## 2. Understanding Threads, Blocks, and Grids

### Threads:
- The **smallest unit** of execution in CUDA.
- Each thread executes the same kernel code but can operate on **different data elements**.

### Blocks:
- A **group of threads** that can **share memory** and synchronize execution.
- Threads within a block communicate **efficiently**.

### Grids:
- A **collection of blocks** that together form the complete **execution configuration** for a kernel launch.
- Grids allow large datasets to be processed efficiently by scaling across many GPU cores.

### Thread Indexing:
- Each thread calculates its **global index** using:
  ```cpp
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ```
- This index determines **which part of the data** the thread will process.

---

## 3. Code Example: Using a 1D Grid and Blocks

This example helped me understand how CUDA assigns **unique indices** to each thread using a **1D grid and blocks**.

### Code: `thread_indexing.cu`

```cpp
// thread_indexing.cu: Demonstrate thread indexing using a 1D grid and blocks
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel that assigns each element its global thread index
__global__ void fillWithThreadIndex(int *d_data, int n) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we do not access out-of-bound memory
    if (idx < n) {
        d_data[idx] = idx;  // Each element is set to its unique thread index
    }
}

int main() {
    int n = 1024;                   // Total number of elements
    size_t size = n * sizeof(int);   // Total memory required

    // Allocate host memory
    int *h_data = (int *)malloc(size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    int *d_data;
    cudaMalloc((void **)&d_data, size);

    // Define execution configuration: 1D grid and blocks
    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;  // Ensure all elements are covered

    // Launch the kernel
    fillWithThreadIndex<<<gridSize, blockSize>>>(d_data, n);
    cudaDeviceSynchronize();

    // Copy results from device to host memory
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Display the first few results to verify thread indexing
    printf("First 20 elements (each should match its global thread index):\n");
    for (int i = 0; i < 20; i++) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    // Cleanup
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

---

## 4. Detailed Explanation of the Code

### Kernel Function: `fillWithThreadIndex`
- **Thread Index Calculation:**
  - `idx = blockIdx.x * blockDim.x + threadIdx.x;`
  - This uniquely identifies each thread across the entire grid.
- **Bounds Checking:**
  - Ensures no **out-of-bounds memory access**.

### Execution Configuration:
- **Block Size:** `128` threads per block.
- **Grid Size:** `(n + blockSize - 1) / blockSize` ensures complete array coverage.

### Host-Device Interaction:
- Memory is allocated on both the **host (CPU)** and **device (GPU)**.
- Data is transferred between **CPU and GPU** for processing.

---

## 5. Suggested Exercises

### Exercise 1:
- Modify the kernel to write the **square of each thread index** into the array.

### Exercise 2:
- Experiment with different **block sizes** (e.g., `64`, `256`) and analyze how thread indices change.

### Exercise 3:
- Extend the program to use a **2D grid** and calculate **2D thread indices**.

---

## 6. Real-World Applications

### **Parallel Data Analytics:**
- Each thread can process **individual data records** or elements of a dataset concurrently.
- Forms the basis for **filtering, transformation, and statistical analysis** in **big data applications**.

### **Image Processing:**
- Each thread can handle a **pixel or group of pixels** for **image transformations**.
- CUDA is widely used in **image filtering, edge detection, and object recognition**.

---

## 7. Conclusion & Next Steps

### Summary:
- Explored **CUDA thread hierarchy** and execution configuration.
- Learned how to set up a **1D grid with blocks**.
- Implemented **thread indexing** to ensure **correct parallel execution**.


---

## Final Thoughts

Learning about **CUDA thread hierarchy** was an exciting experience! At first, it seemed a bit tricky, but breaking it down into **threads, blocks, and grids** made it much clearer.

Understanding **thread indexing** felt like unlocking a secret to **efficient parallel execution**. Seeing the **threads map directly to data elements** was fascinating! This knowledge lays a **solid foundation** for writing optimized CUDA programs in the future.

If you’re just starting out, my advice is to **experiment and tweak**—try different **thread configurations**, analyze the **thread indices**, and observe how they map to data. The more you play around, the better you’ll understand CUDA execution!

I’m excited to dive into **CUDA memory management** next—time to optimize performance! 🚀
