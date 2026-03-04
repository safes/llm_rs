//! Layer Normalization
//! Converts from train_gpt2.c lines 78-161

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3};

const EPS: f32 = 1e-5;

/// Forward pass for layer normalization
///
/// # Arguments
/// * `out` - Output tensor (B, T, C)
/// * `mean` - Mean values (B, T)
/// * `rstd` - Reciprocal standard deviation (B, T)
/// * `inp` - Input tensor (B, T, C)
/// * `weight` - Scale parameters (C,)
/// * `bias` - Shift parameters (C,)
pub fn layernorm_forward(
    mut out: ArrayViewMut3<f32>,
    mut mean: ArrayViewMut2<f32>,
    mut rstd: ArrayViewMut2<f32>,
    inp: &Array3<f32>,
    weight: &Array1<f32>,
    bias: &Array1<f32>,
) {
    let (b, t, c) = inp.dim();

    for batch in 0..b {
        for pos in 0..t {
            // Calculate mean
            let mut m = 0.0f32;
            for i in 0..c {
                m += inp[[batch, pos, i]];
            }
            m /= c as f32;

            // Calculate variance
            let mut v = 0.0f32;
            for i in 0..c {
                let xshift = inp[[batch, pos, i]] - m;
                v += xshift * xshift;
            }
            v /= c as f32;

            // Calculate reciprocal standard deviation
            let s = 1.0 / (v + EPS).sqrt();

            // Normalize, scale, and shift
            for i in 0..c {
                let n = s * (inp[[batch, pos, i]] - m);
                out[[batch, pos, i]] = n * weight[i] + bias[i];
            }

            // Cache mean and rstd for backward pass
            mean[[batch, pos]] = m;
            rstd[[batch, pos]] = s;
        }
    }
}

/// Backward pass for layer normalization
///
/// # Arguments
/// * `dinp` - Gradient for input (B, T, C)
/// * `dweight` - Gradient for weight (C,)
/// * `dbias` - Gradient for bias (C,)
/// * `dout` - Gradient from upstream (B, T, C)
/// * `inp` - Input tensor (B, T, C)
/// * `weight` - Scale parameters (C,)
/// * `mean` - Cached mean values (B, T)
/// * `rstd` - Cached reciprocal std (B, T)
pub fn layernorm_backward(
    mut dinp: ArrayViewMut3<f32>,
    mut dweight: ArrayViewMut1<f32>,
    mut dbias: ArrayViewMut1<f32>,
    dout: &Array3<f32>,
    inp: &Array3<f32>,
    weight: &Array1<f32>,
    mean: &Array2<f32>,
    rstd: &Array2<f32>,
) {
    let (b, t, c) = dout.dim();

    for batch in 0..b {
        for pos in 0..t {
            let mean_bt = mean[[batch, pos]];
            let rstd_bt = rstd[[batch, pos]];

            // First: two reduce operations
            let mut dnorm_mean = 0.0f32;
            let mut dnorm_norm_mean = 0.0f32;

            for i in 0..c {
                let norm_bti = (inp[[batch, pos, i]] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout[[batch, pos, i]];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }

            dnorm_mean /= c as f32;
            dnorm_norm_mean /= c as f32;

            // Second: accumulate gradients
            for i in 0..c {
                let norm_bti = (inp[[batch, pos, i]] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout[[batch, pos, i]];

                // Gradient for bias
                dbias[i] += dout[[batch, pos, i]];

                // Gradient for weight
                dweight[i] += norm_bti * dout[[batch, pos, i]];

                // Gradient for input
                let mut dval = dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp[[batch, pos, i]] += dval;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_layernorm_forward() {
        let b = 1;
        let t = 1;
        let c = 4;

        let inp = Array3::from_shape_vec((b, t, c), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = Array1::ones(c);
        let bias = Array1::zeros(c);

        let mut out = Array3::zeros((b, t, c));
        let mut mean = Array2::zeros((b, t));
        let mut rstd = Array2::zeros((b, t));

        layernorm_forward(
            out.view_mut(),
            mean.view_mut(),
            rstd.view_mut(),
            &inp,
            &weight,
            &bias,
        );

        // Mean should be 2.5
        assert!((mean[[0, 0]] - 2.5).abs() < 1e-5);

        // Output should be normalized (mean=0, std=1)
        let out_mean: f32 = out.iter().sum::<f32>() / c as f32;
        assert!(out_mean.abs() < 1e-5);
    }
}
