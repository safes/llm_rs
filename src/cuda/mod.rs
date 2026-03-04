use std::ffi::c_void;

#[repr(C)]
pub struct cudaDeviceProp {
    // We don't need the full struct definition if we only use it opaquely or
    // if the C++ side handles it. But if we need to access properties, we'd need it.
    // For now, let's keep it opaque or just trust the C++ init.
    _opaque: [u8; 1024], // Placeholder size
}

extern "C" {
    pub fn init_cuda_c();
    pub fn cleanup_cuda_c();

    // Memory management
    pub fn malloc_c(ptr: *mut *mut c_void, size: usize);
    pub fn free_c(ptr: *mut c_void);
    pub fn memcpy_htod_c(dest: *mut c_void, src: *const c_void, size: usize);
    pub fn memcpy_dtoh_c(dest: *mut c_void, src: *const c_void, size: usize);
    pub fn memset_c(dest: *mut c_void, value: i32, size: usize);
    pub fn device_synchronize_c();

    // Kernels
    pub fn encoder_forward_c(
        out: *mut f32,
        inp: *const i32,
        wte: *mut f32,
        wpe: *mut f32,
        B: i32,
        T: i32,
        C: i32,
    );
    pub fn encoder_backward_c(
        dwte: *mut f32,
        dwpe: *mut f32,
        dout: *mut f32,
        inp: *const i32,
        inputs_cpu: *const i32,
        scratch: *mut f32,
        workload_indices: *mut i32,
        bucket_info: *mut i32,
        B: i32,
        T: i32,
        C: i32,
        seed: u32,
    );

    pub fn layernorm_forward_c(
        out: *mut f32,
        mean: *mut f32,
        rstd: *mut f32,
        inp: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        B: i32,
        T: i32,
        C: i32,
    );

    pub fn layernorm_backward_c(
        dinp: *mut f32,
        dweight: *mut f32,
        dbias: *mut f32,
        scratch: *mut f32,
        dout: *mut f32,
        inp: *mut f32,
        weight: *mut f32,
        mean: *const f32,
        rstd: *const f32,
        B: i32,
        T: i32,
        C: i32,
    );

    pub fn matmul_forward_c(
        out: *mut f32,
        inp: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        B: i32,
        T: i32,
        C: i32,
        OC: i32,
    );

    pub fn matmul_backward_c(
        dinp: *mut f32,
        dweight: *mut f32,
        dbias: *mut f32,
        dout: *mut f32,
        inp: *mut f32,
        weight: *mut f32,
        dbias_buffer: *mut f32,
        B: i32,
        T: i32,
        C: i32,
        OC: i32,
    );

    pub fn attention_forward_c(
        out: *mut f32,
        qkvr: *mut f32,
        att: *mut f32,
        scratch: *mut f32,
        B: i32,
        T: i32,
        C: i32,
        NH: i32,
    );

    pub fn attention_backward_c(
        dinp: *mut f32,
        dqkvr: *mut f32,
        datt: *mut f32,
        scratch: *mut f32,
        dout: *mut f32,
        qkvr: *mut f32,
        att: *mut f32,
        B: i32,
        T: i32,
        C: i32,
        NH: i32,
    );

    pub fn fused_classifier_c(
        dlogits: *mut f32,
        losses: *mut f32,
        logits: *mut f32,
        targets: *const i32,
        B: i32,
        T: i32,
        V: i32,
        Vp: i32,
    );

    pub fn fused_classifier_backward_c(
        dlogits: *mut f32,
        losses: *mut f32,
        targets: *const i32,
        B: i32,
        T: i32,
        V: i32,
        Vp: i32,
    );

    pub fn fused_residual_forward5_c(
        residual_out: *mut f32,
        ln_out: *mut f32,
        mean: *mut f32,
        rstd: *mut f32,
        residual_in: *mut f32,
        ln_in: *mut f32,
        weight: *mut f32,
        bias: *mut f32,
        N: i32,
        C: i32,
    );

    pub fn residual_forward_c(out: *mut f32, inp1: *const f32, inp2: *const f32, N: i32);
    pub fn residual_backward_c(dinp1: *mut f32, dinp2: *mut f32, dout: *const f32, N: i32);

    pub fn gelu_forward_c(out: *mut f32, inp: *const f32, N: usize);
    pub fn gelu_backward_c(dinp: *mut f32, inp: *const f32, dout: *const f32, N: usize);
    pub fn accumulate_c(a: *mut f32, b: *const f32, n: usize);

    pub fn adamw_step_c(
        params: *mut f32,
        grads: *mut f32,
        m: *mut f32,
        v: *mut f32,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
        num_parameters: usize,
    );
    pub fn global_norm_c(out: *mut f32, values: *const f32, count: usize);
    pub fn scale_grads_c(grads: *mut f32, scale: f32, count: usize);
}

/// Initialize CUDA context and libraries (cuBLAS, cuBLASLt)
pub fn init() {
    unsafe {
        init_cuda_c();
    }
}

/// Cleanup CUDA resources
pub fn cleanup() {
    unsafe {
        cleanup_cuda_c();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_init() {
        // This test requires a generic lock to avoid race conditions if run in parallel
        // with other CUDA tests, but for now we only have one.
        // It also requires a GPU.
        if std::env::var("SKIP_CUDA_TESTS").is_err() {
            init();
            cleanup();
        }
    }
}
