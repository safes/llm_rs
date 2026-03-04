//! Residual connections
//! Converts from train_gpt2.c lines 436-447

/// Forward pass for residual connection
pub fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32]) {
    assert_eq!(out.len(), inp1.len());
    assert_eq!(out.len(), inp2.len());

    for i in 0..out.len() {
        out[i] = inp1[i] + inp2[i];
    }
}

/// Backward pass for residual connection
pub fn residual_backward(dinp1: &mut [f32], dinp2: &mut [f32], dout: &[f32]) {
    assert_eq!(dinp1.len(), dout.len());
    assert_eq!(dinp2.len(), dout.len());

    for i in 0..dout.len() {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}
