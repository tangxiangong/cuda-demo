use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/matmul.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let matmul_ptx_path = out_dir.join("matmul.ptx");

    let status = Command::new("nvcc")
        .args([
            "-ptx",
            "src/kernels/matmul.cu",
            "-o",
            matmul_ptx_path.to_str().expect("Invalid PTX path"),
        ])
        .status()
        .expect("Failed to execute nvcc. Is CUDA installed and nvcc in PATH?");

    if !status.success() {
        panic!("nvcc failed with status: {status}");
    }

    println!(
        "cargo:rustc-env=MATMUL_KERNEL_PTX={}",
        matmul_ptx_path.display()
    );
}
