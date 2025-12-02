use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/bm.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    let bm_ptx_path = out_dir.join("bm.ptx");
    let status = Command::new("nvcc")
        .args([
            "-ptx",
            "src/kernels/bm.cu",
            "-o",
            bm_ptx_path.to_str().expect("Invalid PTX path"),
        ])
        .status()
        .expect("Failed to execute nvcc. Is CUDA installed and nvcc in PATH?");

    if !status.success() {
        panic!("nvcc failed with status: {status}");
    }

    println!("cargo:rustc-env=BM_KERNEL_PTX={}", bm_ptx_path.display());
}
