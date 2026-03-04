use std::env;
use std::path::PathBuf;

fn main() {
    // Only build CUDA code if the "cuda" feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2".to_string());
        
        println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");

        // Compile the wrapper.cu file
        cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_61") // Target Pascal (Quadro P620)
            .flag("-use_fast_math")
            .flag("--expt-relaxed-constexpr")
            .define("ENABLE_FP32", None) // Force FP32 for now to match Rust CPU code
            .include("../llmc") // Include path to llm.c/llmc headers
            .include("../") // Include path for dev/unistd.h (referenced as dev/unistd.h)
            .file("src/cuda/wrapper.cu")
            .compile("llm_cuda_kernels");

        println!("cargo:rerun-if-changed=src/cuda/wrapper.cu");
    }
}
