# Day 9: Error Handling & Debugging in CUDA

Detecting and handling errors in CUDA applications is crucial for **developing reliable and efficient GPU programs**. Today, I explored **error handling techniques, debugging tools, and best practices** to ensure smooth execution in CUDA programs.

## 1. Overview & Objectives

### **Objective:**
Learn how to **detect and handle runtime errors** in CUDA applications and use debugging tools to diagnose issues.

### **Key Learning Outcomes:**
- Understand **CUDA error codes** and the `cudaError_t` type.
- Learn how to **check and handle errors** after **CUDA API calls and kernel launches**.
- Use debugging tools like **cuda-memcheck** to find memory errors and kernel issues.

### **Real-World Application:**
Reliable error handling and debugging are **crucial in robotics**, where **accurate calculations and predictable execution** are mandatory for safety and performance.

---

## 2. Key Concepts in CUDA Error Handling and Debugging

### **CUDA Error Codes & `cudaError_t`:**
- Every CUDA function **returns** a value of type `cudaError_t`.
- Use `cudaGetErrorString(err)` to **get human-readable error messages**.

### **Error Checking Patterns:**
- **After API Calls:** Check the return value of `cudaMalloc`, `cudaMemcpy`, etc.
- **After Kernel Launches:** Use `cudaGetLastError()` and `cudaDeviceSynchronize()` to **catch asynchronous execution errors**.

### **Debugging Tools:**
- **`cuda-memcheck`**: Detects **out-of-bounds memory accesses, misaligned accesses**, and other errors.
- **Other Tools**: **NVIDIA Nsight** for advanced debugging and performance profiling.

---

## 3. Code Example: Introducing a Deliberate Error

This example demonstrates:
- **Error checking** for CUDA API calls.
- **Kernel execution error detection** using `cudaGetLastError()` and `cudaDeviceSynchronize()`.
- **Deliberate error**: Writing **out-of-bound memory** to demonstrate CUDA error handling.

### **Code: `error_handling_debugging.cu`**

```cpp
// error_handling_debugging.cu: Demonstrates error handling and debugging in CUDA

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel with a deliberate error: accessing out-of-bound memory
__global__ void faultyKernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        d_data[n] = 42;  // Intentional error: writing beyond allocated memory
    }
}

int main() {
    int n = 10;                    // Allocate an array of 10 integers
    size_t size = n * sizeof(int);
    int *d_data = NULL;
    cudaError_t err;

    // Allocate device memory and check for errors
    err = cudaMalloc((void **)&d_data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch the kernel with 1 block of 5 threads
    faultyKernel<<<1, 5>>>(d_data, n);

    // Check for errors after kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to detect runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_data);
    return 0;
}
```
### **Output:**
```
Kernel launch error: the provided PTX was compiled with an unsupported toolchain.
```

---

## 4. Detailed Explanation of the Code

### **Memory Allocation and Error Checking:**
- `cudaMalloc()` allocates **device memory** and immediately checks for errors.

### **Faulty Kernel Implementation:**
- `faultyKernel()` **deliberately accesses out-of-bound memory**, triggering an error.
- Writing to `d_data[n]` exceeds the valid range of **0 to n-1**.

### **Kernel Launch and Error Checks:**
- `cudaGetLastError()` **catches kernel launch errors**.
- `cudaDeviceSynchronize()` **detects runtime errors** (async execution errors).

### **Using Debugging Tools:**
- Run with **cuda-memcheck** to identify memory errors:
  ```bash
  cuda-memcheck ./error_handling_debugging
  ```

---

## 5. Suggested Exercises

### **Exercise 1:**
- Modify the kernel to **trigger a misaligned memory access** and diagnose the issue using `cuda-memcheck`.

### **Exercise 2:**
- Wrap multiple **CUDA API calls** with error checking and **intentionally trigger errors** (e.g., passing incorrect arguments).

### **Exercise 3:**
- Implement a **robust error-checking macro** and integrate it into a CUDA application.

---
## Solutions to Exercises

### **Exercise 1: Trigger a Misaligned Memory Access**
Modify the kernel to **read and write misaligned memory** and diagnose it using `cuda-memcheck`:
```cpp
__global__ void misalignedAccessKernel(int *d_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *misaligned_ptr = (int*)((char*)d_data + 1); // Force misaligned memory access
    *misaligned_ptr = 42;
}
```
Run with `cuda-memcheck`:
```bash
cuda-memcheck ./error_handling_debugging
```

### **Exercise 2: Wrap CUDA API Calls with Error Checking and Trigger Errors**
Modify API calls to **trigger errors** and wrap them with error checking:
```cpp
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err); \
        } \
    } while (0)

int main() {
    int *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, -1)); // Intentional error: invalid size
    cudaFree(d_data);
    return 0;
}
```

### **Exercise 3: Implement a Robust Error-Checking Macro**
Define an **error-checking macro** and use it in a CUDA application:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

__global__ void simpleKernel(int *d_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_data[idx] = idx * 2;
}

int main() {
    int *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, 10 * sizeof(int)));
    simpleKernel<<<1, 10>>>(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_data);
    return 0;
}
```


---

## 6. Real-World Applications

### **Reliable Computation in Robotics:**
- Robotics **requires real-time, error-free execution** for **sensor data processing and decision-making**.
- **Robust error handling** ensures systems **operate safely and efficiently**.

### **High-Performance Computing (HPC):**
- Ensures **large-scale simulations** do not crash due to **memory issues or kernel failures**.

---

## 7. Conclusion & Next Steps

### **Summary:**
- Explored **CUDA error handling** and debugging.
- Used `cudaError_t`, `cudaGetLastError()`, and `cudaDeviceSynchronize()` for **error detection**.
- Learned how **cuda-memcheck helps diagnose CUDA memory issues**.

### **Next Steps:**
- Practice by **integrating error-checking routines** in your CUDA programs.
- **Experiment** with debugging tools and **deliberately introduce errors** to strengthen debugging skills.

### **Action Item:**
- Use `cuda-memcheck` to analyze **kernel execution errors**.
- Implement **error-handling macros** for cleaner CUDA code.

---

## Final Thoughts

Debugging CUDA programs **requires discipline and structured error handling**. At first, it may seem complex, but once I understood **error checking mechanisms**, finding and fixing errors became **much easier**.

Using **cuda-memcheck** was an eye-opener, as it **quickly pinpointed memory issues**. If you're learning CUDA, **practice error handling**, experiment with **debugging tools**, and integrate **proper error checks** to make your code more **robust and reliable**.

Excited to explore **more CUDA performance optimizations in future lessons! 🚀**
