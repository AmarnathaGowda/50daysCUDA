# Day 4: Writing Your First CUDA Program

CUDA programming is something I always wanted to explore, and today, I finally got my hands dirty with writing my first CUDA program! Initially, it felt a bit overwhelming, but as I went step by step, things started making sense. If you’re also curious about how GPUs can process data in parallel, this is the perfect starting point.

## 1. Overview & Objectives

### What I Learned Today:
- How to write and run a basic CUDA kernel that processes data.
- The interaction between the CPU (host) and GPU (device) in CUDA programming.
- Memory transfers between host and device.
- How to compile and execute a CUDA program.

### Real-World Application:
Think of image processing—applying filters, adjusting brightness, or enhancing an image. These operations are done on millions of pixels **at the same time**, making GPUs the perfect tool for the job. That’s exactly what CUDA enables!

---

## 2. Understanding CUDA Kernels and Host-Device Interaction

### What is a CUDA Kernel?
A **CUDA kernel** is just a function, but unlike CPU functions, it runs on the **GPU in parallel**. The magic happens because multiple GPU threads execute this function at the same time.

### Host-Device Interaction:
- **Host (CPU):** Handles memory allocation, data preparation, and kernel launching.
- **Device (GPU):** Runs the kernel function across multiple threads.
- **Data Transfer:** Since CPU and GPU have separate memory, data must be copied between them before and after execution.

At first, this looked confusing, but once I saw the actual process, it clicked! 

---

## 3. My First CUDA Program: Squaring an Array

This simple program helped me understand how CUDA works. Here’s what it does:
1. Allocates memory on both the CPU and GPU.
2. Copies data from the CPU to the GPU.
3. Runs a CUDA kernel where **each GPU thread squares an element** of an array.
4. Transfers the results back to the CPU.
5. Prints the final squared values.

### Code Example:

```cpp
// square.cu: My first CUDA program to square each element in an array

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel: Each thread squares one element of the array
__global__ void squareArray(float *d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread index
    if (idx < n) {
        d_array[idx] = d_array[idx] * d_array[idx];
    }
}

int main() {
    int n = 10; // Number of elements
    size_t size = n * sizeof(float); // Total memory size

    // Host array (CPU memory)
    float h_array[n];
    for (int i = 0; i < n; i++) {
        h_array[i] = (float)i;
    }

    // Device array (GPU memory)
    float *d_array;
    cudaMalloc((void **)&d_array, size);

    // Copy data from CPU to GPU
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // Kernel launch setup
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    squareArray<<<gridSize, blockSize>>>(d_array, n);
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // Print the squared values
    printf("Squared Array:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_array[i]);
    }
    printf("\n");

    // Free GPU memory
    cudaFree(d_array);

    return 0;
}
```

---

## 4. Compiling & Running My CUDA Code

### Compilation Steps:
1. Open a terminal and navigate to the directory where `square.cu` is saved.
2. Compile the code using the NVIDIA CUDA Compiler:
   ```bash
   nvcc square.cu -o square
   ```

### Running the Program:
Run the compiled program:
   ```bash
   ./square
   ```

### Expected Output:
```
Squared Array:
0.000000 1.000000 4.000000 9.000000 16.000000 25.000000 36.000000 49.000000 64.000000 81.000000 
```

I was thrilled when I saw the correct output! It meant my first CUDA program worked successfully!

---

## 5. Things I Experimented With

### 1. Modifying the Computation:
I changed the kernel function to **cube** each element instead of squaring.

### 2. Playing with Thread Configurations:
I adjusted `blockSize` and `gridSize` to see how it affects performance for **larger arrays**.

### 3. Adding Error Handling:
To make my code more robust, I added error checks after CUDA API calls like `cudaMalloc()` and `cudaMemcpy()`.

---

## 6. Real-World Applications

### How is This Used in the Real World?
- **Image Processing:** Each pixel is like an array element, modified in parallel by CUDA threads.
- **Deep Learning:** Neural networks rely on matrix operations that are processed in parallel on GPUs.
- **Simulations & Physics Engines:** Compute-heavy simulations become feasible because of CUDA’s parallelism.

This made me realize that CUDA isn’t just theoretical—it’s powering things I use every day!

---

## 7. Wrapping Up & Next Steps

### What I Learned:
- The basics of writing **a CUDA program**.
- How **parallel execution** happens on the GPU.
    - GPUs process multiple tasks **simultaneously** using **many small cores**. Unlike CPUs (which have few powerful cores), GPUs handle **massive parallel computing**, making them ideal for AI, gaming, and simulations.

    ## 1️⃣ GPU Architecture  
    - **Thousands of small cores** enable parallel execution.  
    - **Optimized for high-performance computing** in AI, graphics, and simulations.  

    ## 2️⃣ Execution Model  
    - **Thread →** Smallest unit of execution.  
    - **Block →** Group of threads sharing resources.  
    - **Grid →** Collection of blocks.  
    - **SIMD Execution →** Same instruction applied to multiple data points.  

    ## 3️⃣ Parallel Processing Steps  
    1. **Divide Work** → Split large tasks into smaller ones.  
    2. **Assign Threads** → Each thread gets a task.  
    3. **Form Blocks** → Threads are grouped into blocks.  
    4. **Create Grid** → Blocks are structured for execution.  
    5. **Run in Parallel** → Multiple threads execute **together** efficiently.  

    ## 4️⃣ Memory Hierarchy  
    - **Shared Memory** → Fast, used within a block.  
    - **Global Memory** → Slower, accessible by all threads.  

    ## 5️⃣ Performance Boosts  
    - **Warps (32 threads)** execute together.  
    - **Streaming Multiprocessors (SMs)** handle multiple warps.  
    - **Pipelining** speeds up execution.  
    - **Asynchronous execution** enables simultaneous computation & data transfer.  

    ## 6️⃣ Why GPUs?  
    - **Faster than CPUs** for parallel tasks.  
    - **Used in AI, gaming, simulations, and scientific research.**  

    🔥 **GPUs = High-speed computing power!**
- How to **transfer data** between the CPU and GPU.
    ```cpp
    // Allocate host and device memory
        float *h_data, *d_data;
        size_t size = N * sizeof(float);

        cudaMallocHost(&h_data, size);  // Pinned host memory
        cudaMalloc(&d_data, size);      // Device memory

    // Transfer data from host to device
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel
        myKernel<<<gridSize, blockSize>>>(d_data);

    // Transfer results back to host
        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Free memory
        cudaFreeHost(h_data);
        cudaFree(d_data);

    ```

### What’s Next (Day 5 Preview):
- Learning more about **CUDA threading models**.
- Understanding **threads, blocks, and grids** for optimizing GPU performance.

### What You Can Try:
- Modify the kernel to perform a different computation.
- Try using **larger arrays** and optimize execution speed.
- Explore **error handling** in CUDA.

I am excited to dive deeper into CUDA threading tomorrow! If you’ve followed along, congratulations on writing your first CUDA program! 🚀

---

## Final Thoughts

Writing my first CUDA program was a fantastic learning experience! At first, it seemed complex, but as I broke it down into steps, it became much easier. The power of parallel computing is truly fascinating, and seeing my program process data across multiple threads in parallel was mind-blowing!

If you’re starting out, my advice would be to **experiment**—play around with different computations, try handling larger datasets, and observe how different configurations affect performance. The more you explore, the deeper your understanding will be.

I can’t wait to explore more about CUDA threading in the next lesson. Onward to more CUDA adventures! 🚀
