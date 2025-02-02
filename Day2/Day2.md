# Day 2: Setting Up CUDA Environment

Yesterday, we explored some of the theoretical concepts of GPUs. Today, it's time to get our hands dirty and set up our CUDA development environment! This is an exciting step because once everything is properly configured, we will be able to run CUDA programs and truly harness the power of parallel computing.

## 1. Overview & Objectives

### Objective:
Learn how to install and configure the CUDA development environment by setting up NVIDIA drivers, the CUDA Toolkit, and an Integrated Development Environment (IDE).

### Key Learning Outcome:
Successfully install and configure the CUDA toolkit and drivers, ensuring your system is ready to compile and run CUDA applications.

---

## 2. Prerequisites

### Hardware Check:
- Verify that your computer has an **NVIDIA GPU** that supports CUDA.
- Check your GPU model using the system’s device manager or by running:
  ```bash
  nvidia-smi
  ```
  (if NVIDIA drivers are already installed).

### Operating System Compatibility:
- Ensure your **OS** (Windows, Linux, or macOS) is compatible with the CUDA Toolkit version you plan to install.

---

## 3. Installing NVIDIA Drivers

### For Windows:
1. Visit the **[NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx)** page.
2. Enter your **GPU model** and download the latest driver.
3. Follow the installation wizard to complete the setup.

### For Linux:
- Install drivers via your **distribution’s package manager** or download them from NVIDIA.
- On **Ubuntu**, run:
  ```bash
  sudo apt update
  sudo apt install nvidia-driver-<version>
  ```
- Reboot your system to ensure the drivers are correctly loaded.

---

## 4. Installing the CUDA Toolkit

### Download & Installation:
1. Visit the **[CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)** page.
2. Select the appropriate version for your OS.
3. Follow the guided installation steps:
   - **Windows**: Installer sets up the toolkit.
   - **Linux**: Use package manager or runfile installation.

### Environment Variables:
- Ensure CUDA Toolkit’s `bin` and `lib` directories are added to your system’s **PATH**:
  - **Windows**: Typically handled by the installer.
  - **Linux**: Verify with:
    ```bash
    echo $PATH
    ```

---

## 5. IDE Configuration

### Choosing an IDE:
- Recommended options:
  - **Visual Studio** (Windows)
  - **Visual Studio Code** (cross-platform)
  - **Command-line editors**

### Visual Studio Code Setup (Recommended for Beginners):
1. Install **[Visual Studio Code](https://code.visualstudio.com/)**.
2. Install the **C/C++ extension** for syntax highlighting and IntelliSense.
3. Configure a **build task** to compile CUDA code:

   
   {
  "version": "2.0.0",
  "tasks": [
    {
      "label": "build cuda",  
      "type": "shell",        
      "command": "nvcc",      
      "args": ["hello.cu", "-o", "hello"],  
      "group": { 
        "kind": "build",      
        "isDefault": true     
      },
      "problemMatcher": "$gcc"  
    }
  ]
}
   

---

## 6. Running a Sample “Hello World” Kernel

### Sample Code:
Create a file **hello.cu** and add the following CUDA program:

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello, World from thread %d!\n", threadIdx.x);
}

int main() {
    helloKernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### Compilation & Execution:
1. Open a **terminal** or use the **IDE’s build task**.
2. Compile with **nvcc**:
   ```bash
   nvcc hello.cu -o hello
   ```
3. Run the executable:
   ```bash
   ./hello
   ```
4. Expected Output:
   ```
   Hello, World from thread 0!
   Hello, World from thread 1!
   ...
   Hello, World from thread 9!
   ```

---

## 7. Suggested Exercises & Self-Study

### Installation Verification:
- Confirm GPU recognition:
  ```bash
  nvidia-smi
  ```

### IDE Practice:
- Experiment by creating a **simple CUDA project** and configuring build tasks.

### Read & Research:
- Explore **CUDA installation guides** and initial configuration tutorials.
- Review **Massively Parallel Processors: A Hands-on Approach** for system setup details.

---

## 8. Next Steps

### Preview of **Day 3:**
Tomorrow, I will explore **CUDA programming fundamentals**, including GPU architecture and writing CUDA kernels.

### Action Item:
- Ensure your **CUDA environment is functional** by running the **Hello World** example.
- Troubleshoot any issues **before moving forward**.

---

## Final Thoughts
Setting up the CUDA environment might seem like a tedious task, but it is one of the most important steps before diving into CUDA programming. A properly configured environment ensures that you can write, compile, and execute CUDA programs without any issues. 

During this setup, I realized how crucial GPU drivers and toolkit versions are—if they don’t match correctly, things can break! Debugging installation errors can be frustrating, but once it's set up, the possibilities are endless. 

If you’ve completed today’s setup successfully, **congratulations!** You’re now one step closer to unlocking the power of GPU acceleration. If you ran into issues, take some time to troubleshoot and revisit the installation guides. 

Tomorrow, we’ll get into **writing actual CUDA code** and understanding how kernels work. Exciting times ahead! 🚀
