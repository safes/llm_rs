//! Encoder layer: Token + Position embeddings
//! Converts from train_gpt2.c lines 35-76

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};

/// Forward pass for encoder (token + position embeddings)
///
/// # Arguments
/// * `out` - Output tensor (B, T, C)
/// * `inp` - Input token indices (B, T)
/// * `wte` - Token embeddings (V, C)
/// * `wpe` - Position embeddings (maxT, C)
pub fn encoder_forward(
    mut out: ArrayViewMut3<f32>,
    inp: &Array2<i32>,
    wte: &Array2<f32>,
    wpe: &Array2<f32>,
) {
    let (b, t) = inp.dim();
    let c = out.shape()[2];

    for batch in 0..b {
        for pos in 0..t {
            let token_idx = inp[[batch, pos]] as usize;
            let token_emb = wte.row(token_idx);
            let pos_emb = wpe.row(pos);

            for i in 0..c {
                out[[batch, pos, i]] = token_emb[i] + pos_emb[i];
            }
        }
    }
}

/// Backward pass for encoder
///
/// # Arguments
/// * `dwte` - Gradient for token embeddings (V, C)
/// * `dwpe` - Gradient for position embeddings (maxT, C)
/// * `dout` - Gradient from upstream (B, T, C)
/// * `inp` - Input token indices (B, T)
pub fn encoder_backward(
    mut dwte: ArrayViewMut2<f32>,
    mut dwpe: ArrayViewMut2<f32>,
    dout: &Array3<f32>,
    inp: &Array2<i32>,
) {
    let (b, t, c) = dout.dim();

    for batch in 0..b {
        for pos in 0..t {
            let token_idx = inp[[batch, pos]] as usize;

            for i in 0..c {
                let grad = dout[[batch, pos, i]];
                dwte[[token_idx, i]] += grad;
                dwpe[[pos, i]] += grad;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_encoder_forward() {
        let b = 2;
        let t = 3;
        let c = 4;
        let v = 10;

        let mut out = Array3::<f32>::zeros((b, t, c));
        let inp = Array2::from_shape_vec((b, t), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let wte = Array2::<f32>::ones((v, c));
        let wpe = Array2::<f32>::ones((t, c)) * 0.5;

        encoder_forward(out.view_mut(), &inp, &wte, &wpe);

        // Each output should be token_emb (1.0) + pos_emb (0.5) = 1.5
        assert!((out[[0, 0, 0]] - 1.5).abs() < 1e-6);
    }
}
