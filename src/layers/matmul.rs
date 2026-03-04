//! Matrix multiplication layer
//! Converts from train_gpt2.c lines 163-269

use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2, ArrayViewMut3, Zip};
use rayon::prelude::*;

/// Forward pass for matrix multiplication with optional bias
///
/// # Arguments
/// * `out` - Output tensor (B, T, OC)
/// * `inp` - Input tensor (B, T, C)
/// * `weight` - Weight matrix (OC, C)
/// * `bias` - Optional bias vector (OC,)
pub fn matmul_forward(
    mut out: ArrayViewMut3<f32>,
    inp: &Array3<f32>,
    weight: &Array2<f32>,
    bias: Option<&[f32]>,
) {
    let (b, t, c) = inp.dim();
    let oc = out.shape()[2];

    // Parallel over batch and time dimensions
    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(batch, mut out_b)| {
            for pos in 0..t {
                for o in 0..oc {
                    let mut val = if let Some(b) = bias { b[o] } else { 0.0 };

                    for i in 0..c {
                        val += inp[[batch, pos, i]] * weight[[o, i]];
                    }

                    out_b[[pos, o]] = val;
                }
            }
        });
}

/// Backward pass for matrix multiplication
///
/// # Arguments
/// * `dinp` - Gradient for input (B, T, C)
/// * `dweight` - Gradient for weight (OC, C)
/// * `dbias` - Optional gradient for bias (OC,)
/// * `dout` - Gradient from upstream (B, T, OC)
/// * `inp` - Input from forward pass (B, T, C)
/// * `weight` - Weight matrix (OC, C)
pub fn matmul_backward(
    mut dinp: ArrayViewMut3<f32>,
    mut dweight: ArrayViewMut2<f32>,
    mut dbias: Option<&mut [f32]>,
    dout: &Array3<f32>,
    inp: &Array3<f32>,
    weight: &Array2<f32>,
) {
    let (b, t, c) = inp.dim();
    let oc = dout.shape()[2];

    // Backward into input (parallel over B, T)
    dinp.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(batch, mut dinp_b)| {
            for pos in 0..t {
                for o in 0..oc {
                    let d = dout[[batch, pos, o]];
                    for i in 0..c {
                        dinp_b[[pos, i]] += weight[[o, i]] * d;
                    }
                }
            }
        });

    // Backward into weight and bias (sequential to avoid borrow checker issues)
    for o in 0..oc {
        for batch in 0..b {
            for pos in 0..t {
                let d = dout[[batch, pos, o]];

                if let Some(db) = dbias.as_mut() {
                    db[o] += d;
                }

                for i in 0..c {
                    dweight[[o, i]] += inp[[batch, pos, i]] * d;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_matmul_forward() {
        let b = 1;
        let t = 2;
        let c = 3;
        let oc = 2;

        let inp = Array3::ones((b, t, c));
        let weight = Array2::ones((oc, c));
        let bias = vec![0.5; oc];

        let mut out = Array3::zeros((b, t, oc));

        matmul_forward(out.view_mut(), &inp, &weight, Some(&bias));

        // Each output should be sum of inputs (3.0) + bias (0.5) = 3.5
        assert!((out[[0, 0, 0]] - 3.5).abs() < 1e-5);
    }
}
