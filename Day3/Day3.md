# Day 3: CUDA Basics & GPU Architecture

Understanding CUDA and GPU architecture is essential for utilizing the power of parallel computing. Today, we will explore how GPUs work, how CUDA cores and streaming multiprocessors (SMs) help in parallel execution, and how GPU architecture differs from CPUs.

## 1. Overview & Objectives

### Objective:
Develop a clear understanding of GPU hardware fundamentals and the CUDA architecture. Learn how CUDA cores and streaming multiprocessors (SMs) enable massive parallelism.

### Key Learning Outcomes:
- Understand the basic hardware structure of GPUs.
- Identify the role of CUDA cores in parallel processing.
- Comprehend how SMs organize and manage groups of CUDA cores.
- Appreciate the design differences between CPUs and GPUs in terms of processing and memory management.

---

## 2. GPU Hardware Overview

### What is a GPU?
A **GPU (Graphics Processing Unit)** is a specialized processor originally designed for rendering graphics. Unlike CPUs, which handle general-purpose tasks, GPUs contain thousands of simple, efficient cores that can perform many operations **simultaneously**.

### GPU vs. CPU:
| Feature | CPU (Central Processing Unit) | GPU (Graphics Processing Unit) |
|---------|-------------------------------|--------------------------------|
| Cores | Few, high-performance cores | Thousands of simpler cores |
| Task Execution | Optimized for sequential tasks | Optimized for parallel workloads |
| Best For | General-purpose computing | Graphics, AI, and scientific computing |

### Why It Matters:
Understanding these differences explains why GPUs excel at tasks like **graphics rendering, scientific simulations, and deep learning**, which require vast amounts of computation.

---

## 3. CUDA Cores

### Definition & Role:
- **CUDA cores** are the basic computational units inside an NVIDIA GPU.
- Each core executes a single thread’s instruction at a time, enabling **massive parallelism**.

### Comparison to CPU Cores:
- CPUs have fewer but **more powerful** cores that handle complex tasks sequentially.
- GPUs have **hundreds or thousands** of CUDA cores that execute many operations at once.

**Key Point:**
The large number of CUDA cores allows GPUs to process large datasets simultaneously, making them perfect for tasks like deep learning and real-time physics simulations.

---

## 4. Streaming Multiprocessors (SMs)

### What are SMs?
- SMs are **groups of CUDA cores** that work together to execute threads efficiently.
- Each SM manages resources like **registers, shared memory, and scheduling** of CUDA cores.

### How SMs Enhance Performance:
- SMs help organize execution so that multiple threads run **in parallel**.
- They improve memory access efficiency by using **shared memory**.

### Architectural Details:
- A GPU consists of multiple **SMs**, each containing several CUDA cores.
- The total number of SMs and CUDA cores varies by GPU model.

---

## 5. How GPU Architecture Enables Parallel Computing

### Parallel Execution:
- GPUs follow the **Single Instruction, Multiple Threads (SIMT)** model, meaning multiple threads execute the **same instruction** simultaneously.

### Memory Hierarchy:
- **Global Memory**: Accessible by all threads but slower.
- **Shared Memory**: Faster memory shared within an SM.
- **Registers**: The fastest memory, used by individual threads.

### Scalability:
- More **SMs and CUDA cores** enable higher performance.
- The architecture is ideal for **scientific simulations, AI training, and graphics rendering**.

---

## 6. Hands-On Exercise: Querying Your GPU's Properties

A practical way to understand your GPU's architecture is to check its properties using CUDA APIs.

### Sample Code: Querying GPU Properties

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }

    printf("Detected %d CUDA-capable GPU(s):\n", deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        printf("\nDevice %d: %s\n", device, prop.name);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  CUDA Capability: %d.%d\n", prop.major, prop.minor);
    }
    return 0;
}
```
Output : 
![output](output.png)

---

## 7. Suggested Exercises & Self-Study

### Identify GPU Components:
- Use the sample code or the **nvidia-smi** command to check your GPU specifications.

    - To check your GPU specifications using nvidia-smi, you can use the following commands:
      ```
      !nvidia-smi -L
      ```
    - Display a summary of GPU information:
      ```
      !nvidia-smi
      ```
    - Query specific GPU details:
      ```
      nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version,driver_version,temperature.gpu,utilization.gpu,memory.total,memory.free,memory.used --format=csv

      ```
    

### Study Hardware Diagrams:

- Search for **GPU architecture diagrams** and **white papers**.

---

## 🔍 Latest GPU Architecture Details  

### **🟢 NVIDIA Blackwell Architecture**  
The **NVIDIA Blackwell architecture**, announced in **January 2025**, is the latest GPU architecture from NVIDIA.  

#### **Key Features:**  
- Fabricated on **TSMC's custom 4NP process**  
- **GB100 die** with **104 billion transistors**  
- **Dual GB100 dies** connected via **10 TB/s NV-High Bandwidth Interface (NV-HBI)**  
- **Total of 208 billion transistors** across the dual-die package  
- New features for **neural rendering** and **AI-powered computing**  

#### **The Blackwell Architecture White Paper Includes:**  
- **Detailed block diagrams** of the full GB202 GPU  
- **Streaming Multiprocessor (SM) architecture** breakdown  
- **Memory subsystem** and **cache hierarchy details**  
- New **AI and HPC acceleration features**  

---

### **🟠 NVIDIA Ada Lovelace Architecture**  
The **Ada Lovelace architecture**, released in **2022**, includes:  

#### **Key Features:**  
- Fabricated on **TSMC's custom 4N process**  
- **AD102 GPU** with **76.3 billion transistors**  
- **18,432 CUDA Cores, 144 RT Cores, 576 Tensor Cores**  
- New features for **ray tracing** and **neural graphics**  

#### **The Ada Architecture White Paper Provides:**  
- **Full AD102 GPU block diagram**  
- **Detailed Streaming Multiprocessor (SM) layout**  
- **Ray tracing and Tensor Core** improvements  
- **Memory subsystem architecture**  

---

### **🔵 NVIDIA Ampere Architecture**  
The **Ampere architecture**, launched in **2020**, features:  

#### **Key Features:**  
- Fabricated on **Samsung's 8nm process** for GeForce GPUs  
- **GA102 GPU** with **28.3 billion transistors**  
- **10,752 CUDA Cores, 84 RT Cores, 336 Tensor Cores**  
- Improvements in **ray tracing** and **AI performance**  

#### **The Ampere Architecture White Paper Includes:**  
- **GA102 full chip block diagram**  
- **Graphics Processing Cluster (GPC)** and **Streaming Multiprocessor (SM) layouts**  
- **Second-generation RT Core** and **third-generation Tensor Core** details  
- **Memory and cache subsystem architecture**  

---

### 📄 **White Paper Reference:**  
🔗 [White Paper Link](https://www.amax.com/nvidia-technical-whitepapers/)  

---
  

### Reflective Questions:

#### 1. How do CUDA cores and SMs complement each other?

🚀 **CUDA Cores and SMs: The Dynamic Duo of GPU Processing!**  

### Imagine a Bustling Tech Factory 🏭  
Think of a GPU like a massive technological factory, where:  
- **Streaming Multiprocessors (SMs)** are like production floors  
- **CUDA cores** are the hardworking workers 💪  

### How They Work Together 🤝  

#### **SMs: The Smart Managers**  
- Coordinate and schedule work  
- Distribute computational tasks  
- Manage resources like shared memory  
- Control thread block execution  

#### **CUDA Cores: The Computational Powerhouses**  
- Execute actual mathematical operations  
- Process instructions super-fast  
- Work in synchronized teams (warps)  
- Perform floating-point and integer calculations  

### **Real-World Analogy 🌟**  
Imagine a call center:  
- **SM** = Call center manager  
- **CUDA cores** = Individual customer support executives  
- **Thread blocks** = Different customer queries  
- **Warp** = Team handling similar queries  

### **Performance Magic ✨**  
- More **SMs** = More simultaneous processing  
- More **CUDA cores per SM** = Higher computational speed  
- Efficient coordination = Blazing-fast computations  

### **Typical Configuration Example**  
**RTX 3060 Ti:**  
- 38 SMs  
- 4,864 CUDA cores  
- **Massive parallel processing capability 🚀**  

Bas, that's how CUDA cores and SMs rock the GPU world! 💻🇮🇳  

---

#### 2. Why is it better for a GPU to have many simple cores rather than a few powerful ones?

GPUs are designed with many simple cores rather than a few powerful ones for several reasons:  

- **Parallel processing:** GPUs excel at handling tasks that can be broken down into smaller, independent parts. Having numerous simple cores allows for **massive parallelism**, where thousands of threads can be executed simultaneously.  

- **Specialized tasks:** GPU cores are optimized for specific operations, particularly **floating-point calculations and graphics rendering**. This specialization allows for efficient execution of parallel tasks common in **graphics processing and machine learning**.  

- **Heat management:** Running many cores at slower speeds **generates less heat** compared to fewer, more powerful cores. This allows for **better thermal management and higher overall throughput**.  

- **Memory latency tolerance:** GPU architecture is designed to handle **memory latency** more effectively by focusing on **keeping the cores busy** with parallel computations rather than relying heavily on caching.  

- **Cost-effectiveness:** Manufacturing many simple cores is often **more cost-effective** than producing a few complex ones, allowing for **higher computational power at a lower cost**.  

---

## 8. Real-World Applications

### Graphics Rendering:
- GPUs process **millions of pixels** in parallel for real-time rendering.
- Used in **games, animation, and visual effects**.

### Scientific Simulations:
- Ideal for **physics, chemistry, and engineering simulations**.
- Performs **large-scale calculations efficiently**.

---

## 9. Conclusion & Next Steps

### Summary:
- Today, we explored **GPU architecture, CUDA cores, and SMs**.
- This knowledge is essential for writing optimized CUDA programs.

### Preparation for Day 4:
- Tomorrow, we will **write our first CUDA kernel** to perform computations on the GPU.
- This hands-on exercise will build on today’s architectural insights.

### Action Item:
- Ensure you **run the GPU properties program** successfully.
- Review any GPU diagrams to **better understand SMs and CUDA cores**.

---

## Final Thoughts

Learning about GPU architecture and CUDA cores might seem a bit overwhelming at first, but it lays the foundation for efficient parallel computing. Today’s session has helped us understand how GPUs are structured and why they excel at handling massive computational workloads.

If you’ve successfully completed today’s tasks, congratulations! You now have a better idea of **how your GPU operates and how CUDA enables parallelism**. If any part was confusing, take some time to review the concepts and diagrams—understanding the architecture will make future CUDA programming much easier.

Tomorrow, we’ll dive into writing **our first CUDA program** and running computations on the GPU. Exciting times ahead! 🚀
