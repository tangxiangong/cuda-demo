# Rust CUDA 性能测试演示

本项目演示了 Rust (CPU) 与 CUDA (GPU) 实现之间的高性能计算对比。项目特意选择了 **矩阵乘法** (Matrix Multiplication, MatMul) 作为基准测试用例，以展示不同执行模型的性能特征。

## 🚀 实现方式

本项目提供了三种 MatMul 的实现：

1.  **串行 CPU (`seq`)**: 一个朴素的、单线程 Rust 实现。
2.  **并行 CPU (`par`)**: 一个利用 [rayon](https://github.com/rayon-rs/rayon) crate 进行数据并行的多线程实现。
3.  **CUDA GPU (`cuda`)**: 一个通过 [cudarc](https://github.com/coreylowman/cudarc) crate 调用自定义 CUDA kernel 的硬件加速实现。

## 📋 前置要求

-   **Rust Toolchain**: 建议使用最新的 Stable 版本。
-   **CUDA Toolkit**: 必须安装 CUDA 工具包，并且确保 `nvcc` 编译器在系统的 `PATH` 环境变量中，以便编译 CUDA kernels。

## 🛠️ 项目结构

-   `benches/bench.rs`: [Criterion](https://github.com/bheisler/criterion.rs) 基准测试套件，用于对比上述三种实现的性能。
-   `src/kernels/*.cu`: CUDA C++ kernel 源代码文件 (例如 `matmul.cu`)。
-   `src/cuda.rs`: Rust 对 CUDA kernel 的封装，负责数据传输和 kernel 启动。
-   `src/par.rs` & `src/seq.rs`: CPU 版本的实现代码。

## ▶️ 如何运行

确保已安装 CUDA 环境，然后运行以下命令进行基准测试：

```bash
cargo bench
```

这将自动编译 CUDA kernel (通过 `build.rs`) 并运行 Criterion 测试。

## 📊 预期结果

在矩阵乘法任务中，随着矩阵维度的增加（例如 512x512 或更大），预期的性能排序通常为：

**CUDA GPU** > **并行 CPU** > **串行 CPU**

> 注意：对于非常小的数据规模，由于 PCIe 数据传输和 Kernel 启动的开销，GPU 版本可能会慢于 CPU 版本。
