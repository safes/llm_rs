//! Softmax and Cross-Entropy layers
//! Converts from train_gpt2.c lines 449-521

use ndarray::{Array2, Array3, ArrayViewMut2, ArrayViewMut3, s};

/// Forward pass for softmax
///
/// # Arguments
/// * `probs` - Output probabilities (B, T, Vp)
/// * `logits` - Input logits (B, T, Vp)
/// * `v` - Real vocab size
/// * `vp` - Padded vocab size
pub fn softmax_forward(
    mut probs: ArrayViewMut3<f32>,
    logits: &Array3<f32>,
    v: usize,
    vp: usize,
) {
    let (b, t, _) = logits.dim();

    for batch in 0..b {
        for pos in 0..t {
            // Find max for numerical stability
            let mut maxval = f32::NEG_INFINITY;
            for i in 0..v {
                if logits[[batch, pos, i]] > maxval {
                    maxval = logits[[batch, pos, i]];
                }
            }

            // Compute exp and sum
            let mut sum = 0.0f32;
            for i in 0..v {
                let exp_val = (logits[[batch, pos, i]] - maxval).exp();
                probs[[batch, pos, i]] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for i in 0..v {
                probs[[batch, pos, i]] /= sum;
            }

            // Zero out padding
            for i in v..vp {
                probs[[batch, pos, i]] = 0.0;
            }
        }
    }
}

/// Forward pass for cross-entropy loss
///
/// # Arguments
/// * `losses` - Output losses (B, T)
/// * `probs` - Input probabilities (B, T, Vp)
/// * `targets` - Target token indices (B, T)
pub fn crossentropy_forward(
    mut losses: ArrayViewMut2<f32>,
    probs: &Array3<f32>,
    targets: &Array2<i32>,
) {
    let (b, t) = targets.dim();

    for batch in 0..b {
        for pos in 0..t {
            let target_idx = targets[[batch, pos]] as usize;
            losses[[batch, pos]] = -probs[[batch, pos, target_idx]].ln();
        }
    }
}

/// Backward pass for combined softmax and cross-entropy
///
/// # Arguments
/// * `dlogits` - Gradient for logits (B, T, Vp)
/// * `dlosses` - Gradient from upstream (B, T)
/// * `probs` - Probabilities from forward pass (B, T, Vp)
/// * `targets` - Target token indices (B, T)
/// * `v` - Real vocab size
pub fn crossentropy_softmax_backward(
    mut dlogits: ArrayViewMut3<f32>,
    dlosses: &Array2<f32>,
    probs: &Array3<f32>,
    targets: &Array2<i32>,
    v: usize,
) {
    let (b, t) = targets.dim();

    for batch in 0..b {
        for pos in 0..t {
            let dloss = dlosses[[batch, pos]];
            let target_idx = targets[[batch, pos]] as usize;

            for i in 0..v {
                let p = probs[[batch, pos, i]];
                let indicator = if i == target_idx { 1.0 } else { 0.0 };
                dlogits[[batch, pos, i]] += (p - indicator) * dloss;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_softmax_forward() {
        let b = 1;
        let t = 1;
        let v = 3;
        let vp = 4;

        let logits = Array3::from_shape_vec((b, t, vp), vec![1.0, 2.0, 3.0, 0.0]).unwrap();
        let mut probs = Array3::zeros((b, t, vp));

        softmax_forward(probs.view_mut(), &logits, v, vp);

        // Sum of probabilities should be 1.0
        let sum: f32 = probs.slice(s![0, 0, 0..v]).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Padding should be zero
        assert_eq!(probs[[0, 0, 3]], 0.0);
    }
}
