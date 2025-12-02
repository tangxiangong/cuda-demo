use cudarc::{
    driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::sync::{Arc, LazyLock};

const BM_KERNEL_PTX: &str = include_str!(env!("BM_KERNEL_PTX"));

struct CudaKernels {
    ctx: Arc<CudaContext>,
    mean: CudaFunction,
    msd: CudaFunction,
    central_moment: CudaFunction,
    raw_moment: CudaFunction,
}

static CUDA: LazyLock<anyhow::Result<CudaKernels>> = LazyLock::new(|| {
    let ctx = CudaContext::new(0)?;
    let module = ctx.load_module(Ptx::from(BM_KERNEL_PTX))?;
    let mean = module.load_function("bm_mean")?;
    let msd = module.load_function("bm_msd")?;
    let central_moment = module.load_function("bm_central_moment")?;
    let raw_moment = module.load_function("bm_raw_moment")?;
    Ok(CudaKernels {
        ctx,
        mean,
        msd,
        central_moment,
        raw_moment,
    })
});

pub fn bm_mean(
    start_position: f32,
    diffusivity: f32,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> anyhow::Result<f32> {
    let seed = std::time::SystemTime::now().elapsed()?.as_secs();
    let kernels = CUDA
        .as_ref()
        .map_err(|e| anyhow::anyhow!("CUDA init failed: {}", e))?;

    let stream = kernels.ctx.default_stream();

    let mut device_out = stream.alloc_zeros::<f32>(1)?;

    let block_size = 256;
    let grid_size = particles.div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&kernels.mean);
    builder.arg(&mut device_out);
    builder.arg(&start_position);
    builder.arg(&diffusivity);
    builder.arg(&duration);
    builder.arg(&time_step);
    builder.arg(&particles);
    builder.arg(&seed);

    unsafe {
        builder.launch(cfg)?;
    }

    let out_host = stream.clone_dtoh(&device_out)?;
    let sum = out_host[0];
    Ok(sum / particles as f32)
}

pub fn bm_msd(
    diffusivity: f32,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> anyhow::Result<f32> {
    let seed = std::time::SystemTime::now().elapsed()?.as_secs();
    let kernels = CUDA
        .as_ref()
        .map_err(|e| anyhow::anyhow!("CUDA init failed: {}", e))?;

    let stream = kernels.ctx.default_stream();

    let mut device_out = stream.alloc_zeros::<f32>(1)?;

    let block_size = 256;
    let grid_size = particles.div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&kernels.msd);
    builder.arg(&mut device_out);
    builder.arg(&diffusivity);
    builder.arg(&duration);
    builder.arg(&time_step);
    builder.arg(&particles);
    builder.arg(&seed);

    unsafe {
        builder.launch(cfg)?;
    }

    let out_host = stream.clone_dtoh(&device_out)?;
    let sum = out_host[0];
    Ok(sum / particles as f32)
}

pub fn bm_moment(
    start_position: f32,
    diffusivity: f32,
    order: i32,
    central: bool,
    duration: f32,
    time_step: f32,
    particles: usize,
) -> anyhow::Result<f32> {
    let seed = std::time::SystemTime::now().elapsed()?.as_secs();
    let mean = bm_mean(start_position, diffusivity, duration, time_step, particles)?;
    let kernels = CUDA
        .as_ref()
        .map_err(|e| anyhow::anyhow!("CUDA init failed: {}", e))?;

    let stream = kernels.ctx.default_stream();

    let mut device_out = stream.alloc_zeros::<f32>(1)?;

    let block_size = 256;
    let grid_size = particles.div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = if central {
        stream.launch_builder(&kernels.central_moment)
    } else {
        stream.launch_builder(&kernels.raw_moment)
    };
    builder.arg(&mut device_out);
    if central {
        builder.arg(&mean);
    }
    builder.arg(&start_position);
    builder.arg(&diffusivity);
    builder.arg(&order);
    builder.arg(&duration);
    builder.arg(&time_step);
    builder.arg(&particles);
    builder.arg(&seed);

    unsafe {
        builder.launch(cfg)?;
    }

    let out_host = stream.clone_dtoh(&device_out)?;
    let sum = out_host[0];
    Ok(sum / particles as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brownian() {
        let mean = bm_mean(0.0, 0.5, 10.0, 0.01, 10_000).unwrap();
        println!("Mean: {mean}");
    }
}
