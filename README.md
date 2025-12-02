# Rust CUDA Brownian Motion 演示

本项目演示了如何使用 Rust 结合 CUDA (GPU) 进行高性能的布朗运动 (Brownian Motion, BM) 模拟。项目通过 [cudarc](https://github.com/coreylowman/cudarc) 调用自定义 CUDA kernel，并使用 [Criterion](https://github.com/bheisler/criterion.rs) 将其与 CPU 实现 (`diffusionx` crate) 进行性能对比。

## 🚀 功能特性

本项目实现了以下布朗运动统计量的 GPU 加速计算：

*   **均值 (Mean)**
*   **均方位移 (Mean Squared Displacement, MSD)**
*   **原点矩与中心矩 (Raw & Central Moments)**

## 📋 前置要求

*   **Rust Toolchain**: 建议使用最新的 Stable 版本。
*   **CUDA Toolkit**: 必须安装 CUDA 工具包，确保 `nvcc` 编译器位于系统的 `PATH` 环境变量中，以便编译 CUDA kernels。

## 🛠️ 项目结构

*   `src/bm.rs`: Rust 代码，负责初始化 CUDA 上下文、加载 PTX 模块以及启动 GPU kernel。
*   `src/kernels/bm.cu`: CUDA C++ 源代码，包含布朗运动模拟的核心算法。
*   `benches/bm.rs`: 基准测试套件，用于对比 CPU (`diffusionx`, f64) 与 GPU (本项目, f32) 的性能。
*   `build.rs`: 构建脚本，用于自动编译 CUDA 代码为 PTX 格式。

## ▶️ 如何运行

确保已正确配置 CUDA 环境，然后运行以下命令进行基准测试：

```bash
cargo bench
```

这将自动编译 CUDA kernel 并运行 Criterion 测试，输出 CPU 与 GPU 实现的性能对比数据。

## �️ 测试环境

*   **CPU**: Intel Core i7-13·700HX
*   **GPU**: NVIDIA GeForce RTX 5060 Laptop (8GB VRAM)
*   **RAM**: 16GB

## �📊 性能测试结果 (N=10,000)

以下结果基于上述硬件环境运行 `cargo bench` 获得 (时间越短越好)：

| 任务 (Task) | CPU (f64) | GPU (f32) | 加速比 (Speedup) |
| :--- | :--- | :--- | :--- |
| **Mean** | 40.49 ms | 0.82 ms | **~49x** |
| **MSD** | 41.88 ms | 0.83 ms | **~50x** |
| **2nd Central Moment** | 83.38 ms | 1.65 ms | **~50x** |

> 注：CPU 实现采用 `f64` 双精度，GPU 实现采用 `f32` 单精度。对于大规模粒子模拟，GPU 版本展现出了显著的性能优势。
