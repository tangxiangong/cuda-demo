use cudarc::{
    driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::sync::{Arc, LazyLock};

const MATMUL_KERNEL_PTX: &str = include_str!(env!("MATMUL_KERNEL_PTX"));

struct CudaKernels {
    ctx: Arc<CudaContext>,
    matmul: CudaFunction,
}

static CUDA: LazyLock<anyhow::Result<CudaKernels>> = LazyLock::new(|| {
    let ctx = CudaContext::new(0)?;
    let module = ctx.load_module(Ptx::from(MATMUL_KERNEL_PTX))?;
    let matmul = module.load_function("matmul_kernel")?;
    Ok(CudaKernels { ctx, matmul })
});

pub fn matmul(a: &[f32], b: &[f32], n: usize) -> anyhow::Result<Vec<f32>> {
    let kernels = CUDA
        .as_ref()
        .map_err(|e| anyhow::anyhow!("CUDA init failed: {}", e))?;

    let stream = kernels.ctx.default_stream();
    let device_a = stream.clone_htod(a)?;
    let device_b = stream.clone_htod(b)?;
    let mut device_out = stream.alloc_zeros::<f32>(n * n)?;

    let block_size = 16;
    let grid_size = (n as u32).div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size, grid_size, 1),
        block_dim: (block_size, block_size, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&kernels.matmul);
    builder.arg(&mut device_out);
    builder.arg(&device_a);
    builder.arg(&device_b);
    let n_i32 = n as i32;
    let n_i32_ref = &n_i32;
    builder.arg(n_i32_ref);

    unsafe {
        builder.launch(cfg)?;
    }

    let out_host = stream.clone_dtoh(&device_out)?;
    Ok(out_host)
}
