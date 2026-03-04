//! GELU activation function
//! Converts from train_gpt2.c lines 407-434

use std::f32::consts::PI;

const GELU_SCALING_FACTOR: f32 = 0.7978845608; // sqrt(2/PI)

/// Forward pass for GELU activation
///
/// # Arguments
/// * `out` - Output array
/// * `inp` - Input array
pub fn gelu_forward(out: &mut [f32], inp: &[f32]) {
    assert_eq!(out.len(), inp.len());

    for (o, &x) in out.iter_mut().zip(inp.iter()) {
        let cube = 0.044715 * x * x * x;
        *o = 0.5 * x * (1.0 + (GELU_SCALING_FACTOR * (x + cube)).tanh());
    }
}

/// Backward pass for GELU activation
///
/// # Arguments
/// * `dinp` - Gradient for input
/// * `inp` - Input array
/// * `dout` - Gradient from upstream
pub fn gelu_backward(dinp: &mut [f32], inp: &[f32], dout: &[f32]) {
    assert_eq!(dinp.len(), inp.len());
    assert_eq!(dinp.len(), dout.len());

    for i in 0..inp.len() {
        let x = inp[i];
        let cube = 0.044715 * x * x * x;
        let tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        let tanh_out = tanh_arg.tanh();
        let cosh_out = tanh_arg.cosh();
        let sech_out = 1.0 / (cosh_out * cosh_out);
        let local_grad = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x);
        dinp[i] += local_grad * dout[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_forward() {
        let inp = vec![0.0, 1.0, -1.0, 2.0];
        let mut out = vec![0.0; 4];

        gelu_forward(&mut out, &inp);

        // GELU(0) ≈ 0
        assert!(out[0].abs() < 1e-5);
        // GELU(1) ≈ 0.841
        assert!((out[1] - 0.841).abs() < 0.01);
    }
}
