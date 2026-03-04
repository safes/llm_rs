//! Multi-head attention layer
//! Converts from train_gpt2.c lines 271-405

use ndarray::{Array3, Array4, ArrayViewMut3, ArrayViewMut4};

/// Forward pass for multi-head attention
///
/// # Arguments
/// * `out` - Output tensor (B, T, C)
/// * `preatt` - Pre-attention scores (B, NH, T, T)
/// * `att` - Attention weights (B, NH, T, T)
/// * `inp` - Input QKV tensor (B, T, 3*C)
/// * `nh` - Number of heads
pub fn attention_forward(
    mut out: ArrayViewMut3<f32>,
    mut preatt: ArrayViewMut4<f32>,
    mut att: ArrayViewMut4<f32>,
    inp: &Array3<f32>,
    nh: usize,
) {
    let (b, t, c3) = inp.dim();
    let c = c3 / 3;
    let hs = c / nh; // head size
    let scale = 1.0 / (hs as f32).sqrt();

    // Sequential iteration (parallel version has borrow checker issues)
    for batch in 0..b {
        for pos in 0..t {
            for h in 0..nh {
                // Pass 1: Calculate query dot key and maxval
                let mut maxval = f32::NEG_INFINITY;
                for t2 in 0..=pos {
                    // Dot product
                    let mut val = 0.0f32;
                    for i in 0..hs {
                        val += inp[[batch, pos, h * hs + i]]
                            * inp[[batch, t2, h * hs + c + i]];
                    }
                    val *= scale;

                    if val > maxval {
                        maxval = val;
                    }

                    preatt[[batch, h, pos, t2]] = val;
                }

                // Pass 2: Calculate exp and sum
                let mut expsum = 0.0f32;
                for t2 in 0..=pos {
                    let expv = (preatt[[batch, h, pos, t2]] - maxval).exp();
                    expsum += expv;
                    att[[batch, h, pos, t2]] = expv;
                }

                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // Pass 3: Normalize to get softmax
                for t2 in 0..t {
                    if t2 <= pos {
                        att[[batch, h, pos, t2]] *= expsum_inv;
                    } else {
                        // Causal mask
                        att[[batch, h, pos, t2]] = 0.0;
                    }
                }

                // Pass 4: Accumulate weighted values
                for i in 0..hs {
                    out[[batch, pos, h * hs + i]] = 0.0;
                }

                for t2 in 0..=pos {
                    let att_val = att[[batch, h, pos, t2]];
                    for i in 0..hs {
                        out[[batch, pos, h * hs + i]] +=
                            att_val * inp[[batch, t2, h * hs + c * 2 + i]];
                    }
                }
            }
        }
    }
}

/// Backward pass for multi-head attention
///
/// # Arguments
/// * `dinp` - Gradient for input (B, T, 3*C)
/// * `dpreatt` - Gradient for pre-attention (B, NH, T, T)
/// * `datt` - Gradient for attention (B, NH, T, T)
/// * `dout` - Gradient from upstream (B, T, C)
/// * `inp` - Input from forward pass (B, T, 3*C)
/// * `att` - Attention weights from forward pass (B, NH, T, T)
/// * `nh` - Number of heads
pub fn attention_backward(
    mut dinp: ArrayViewMut3<f32>,
    mut dpreatt: ArrayViewMut4<f32>,
    mut datt: ArrayViewMut4<f32>,
    dout: &Array3<f32>,
    inp: &Array3<f32>,
    att: &Array4<f32>,
    nh: usize,
) {
    let (b, t, c3) = inp.dim();
    let c = c3 / 3;
    let hs = c / nh;
    let scale = 1.0 / (hs as f32).sqrt();

    for batch in 0..b {
        for pos in 0..t {
            for h in 0..nh {
                // Backward pass 4: value accumulation
                for t2 in 0..=pos {
                    for i in 0..hs {
                        datt[[batch, h, pos, t2]] +=
                            inp[[batch, t2, h * hs + c * 2 + i]] * dout[[batch, pos, h * hs + i]];
                        dinp[[batch, t2, h * hs + c * 2 + i]] +=
                            att[[batch, h, pos, t2]] * dout[[batch, pos, h * hs + i]];
                    }
                }

                // Backward pass 2 & 3: softmax
                for t2 in 0..=pos {
                    for t3 in 0..=pos {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative =
                            att[[batch, h, pos, t2]] * (indicator - att[[batch, h, pos, t3]]);
                        dpreatt[[batch, h, pos, t3]] +=
                            local_derivative * datt[[batch, h, pos, t2]];
                    }
                }

                // Backward pass 1: query @ key
                for t2 in 0..=pos {
                    for i in 0..hs {
                        dinp[[batch, pos, h * hs + i]] +=
                            inp[[batch, t2, h * hs + c + i]]
                                * dpreatt[[batch, h, pos, t2]]
                                * scale;
                        dinp[[batch, t2, h * hs + c + i]] +=
                            inp[[batch, pos, h * hs + i]] * dpreatt[[batch, h, pos, t2]] * scale;
                    }
                }
            }
        }
    }
}
