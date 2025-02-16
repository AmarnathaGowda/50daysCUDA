# Day 16: Using Texture Memory

CUDA **texture memory** is a **specialized read-only memory** optimized for **spatial locality** and **cached accesses**. It is widely used in **image and video processing**, where **efficient data retrieval** is critical.

## 1. Overview & Objectives

### **Objective:**
Understand and apply **texture memory in CUDA** to enhance **performance in image processing tasks**.

### **Key Learning Outcomes:**
- Learn how to **declare texture references** and **bind data arrays to textures**.
- Use **texture memory caching** to improve **data access efficiency**.
- Implement **texture memory in CUDA kernels** for **fast read operations**.

### **Real-World Application:**
Texture memory is **heavily used in graphics and video processing**, such as in **real-time video enhancement** and **shader programming**, where **efficient memory access** boosts performance.

---

## 2. Key Concepts

### **Texture Memory Basics:**
- **Specialized read-only memory** with **its own caching mechanism**.
- **Optimized for spatial locality**, making it ideal for accessing **image data**.

### **Binding Textures:**
- **Before using texture memory**, a **device array must be bound** to a **texture reference**.
- Texture parameters include **addressing mode** (wrap, clamp) and **filtering mode** (point, linear).

### **Caching Advantages:**
- **Optimized texture caches** reduce **global memory traffic**.
- When **threads in a warp access neighboring pixels**, **texture caches improve efficiency**.

---

## 3. Code Example: Binding and Fetching Texture Memory

This example demonstrates:
- **Declaring and binding texture memory** in CUDA.
- **Using texture memory fetching** inside a CUDA kernel.
- **Improving memory access efficiency** for **image data processing**.

### **Code: `textureMemoryExample.cu`**

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Declare a texture reference for 1D float data
texture<float, cudaTextureType1D, cudaReadModeElementType> texRef;

// Kernel that fetches data from texture memory and inverts image intensities
__global__ void processTextureKernel(float *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = tex1Dfetch(texRef, idx);  // Fetch from texture memory
        d_out[idx] = 1.0f - val;  // Invert the value (simple example)
    }
}

int main() {
    const int n = 1024;
    size_t size = n * sizeof(float);
    
    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = i / (float)n;
    }
    
    float *d_out, *d_data;
    cudaMalloc((void**)&d_out, size);
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;
    texRef.normalized = 0;  // Use unnormalized coordinates
    
    cudaBindTexture(NULL, texRef, d_data, size);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    processTextureKernel<<<gridSize, blockSize>>>(d_out, n);
    cudaDeviceSynchronize();
    
    float *h_out = (float*)malloc(size);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    printf("First 10 output elements (inverted values):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_out[i]);
    }
    printf("\n");
    
    cudaUnbindTexture(texRef);
    cudaFree(d_data);
    cudaFree(d_out);
    free(h_data);
    free(h_out);
    
    return 0;
}
```

---

## 4. Conclusion & Next Steps

### **Summary:**
- **Implemented texture memory fetching** in CUDA.
- **Optimized memory access** using **texture caching advantages**.
- **Reduced global memory accesses** by leveraging **spatial locality**.

### **Next Steps:**
- **Experiment with 2D textures** for **image processing applications**.
- **Compare texture memory vs. global memory performance** using CUDA **profiling tools**.

### **Action Item:**
- **Modify the kernel to process 2D textures** (e.g., brightness adjustment).
- **Analyze the performance improvement** using **texture memory vs. direct global memory access**.

---

## Final Thoughts

CUDA **texture memory** provides a powerful way to **optimize read operations** for **image and video processing**. By leveraging **texture caching**, we can **reduce memory latency and improve performance**.

By applying **texture memory techniques**, CUDA applications **gain significant efficiency in handling image data**. If you're learning CUDA, **experiment with texture memory usage and compare performance gains** using profiling tools.

Looking forward to **more CUDA optimizations next! 🚀**
