# Parallel Neural Network for Image Classification

This project is part of the **High-Performance Computing Engineer Module** course at Prince of Songkla University, Hat Yai Campus. It demonstrates the development and acceleration of training a **Multilayer Perceptron (MLP)** model for image classification. We applied five different Parallel Computing techniques across various architectures to benchmark performance and resolve architecture-specific bottlenecks.

## Dataset
This project utilizes the **Fashion-MNIST** dataset, which consists of grayscale images of clothing items categorized into 10 distinct classes.
* **Data Format:** 28x28 pixel images.
* **Dataset Size:** 60,000 training examples and 10,000 testing examples.
* **Source:** [Fashion-MNIST on Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
  
## Model Architecture
* **Input Layer:** 784 nodes (Supports 28x28 pixel images flattened into a 1D vector).
* **Hidden Layer:** 256 nodes (Activation Function: ReLU).
* **Output Layer:** 10 nodes (Activation Function: Softmax for 10 classes).

## Parallel Techniques & Optimizations
This project implements and deeply optimizes the neural network using the following frameworks:
1. **C++ Thread:** Utilized `std::thread` with a custom Thread Pool pattern. Prevented False Sharing on CPU cache lines by applying 64-byte Array Padding.
2. **OpenMP:** Optimized for Shared Memory CPUs using Kernel Fusion to minimize Fork-Join barrier overheads, along with `#pragma omp simd` for hardware-level vectorization.
3. **MPI:** Scaled for Distributed Memory systems. Used `MPI_Scatter` for efficient data partitioning and minimized network latency via Message Coalescing before calling `MPI_Allreduce`.
4. **CUDA C++:** Achieved massive acceleration on GPUs using Extreme Upfront Memory Management to eliminate PCIe bus bottlenecks, combined with Tiled Matrix Multiplication (16x16) utilizing `__shared__` memory.
5. **PySpark:** Deployed on a Big Data framework using Local SGD via `mapPartitions` and Synchronous Model Averaging via `treeReduce` to minimize serialization overheads.

## Performance Benchmark
*Scope: Benchmarked strictly on the mathematical training phase for 100 Epochs (excluding File I/O and initial memory allocation).*

| Technique | Training Time (Seconds) | Throughput (GFLOPS) |
| :--- | :---: | :---: |
| **CUDA C++** | 3.3988 | 1201.695 |
| **OpenMP** | 45.3439 | 90.075 |
| **C++ Thread** | 49.2501 | 82.930 |
| **MPI** | 64.3576 | 63.463 |
| **PySpark** | 102.2778 | 40.139 |

> **Conclusion:** The GPU architecture (CUDA C++) drastically outperformed others due to its massively parallel computing capabilities suitable for dense matrix multiplications. On the CPU side, OpenMP delivered the highest throughput on a single-node shared memory architecture.

## Contributors
* Keetasin Kongsee (Student ID: 6610110425)
* Alongkorn Jongyingyos (Student ID: 6610110693)
