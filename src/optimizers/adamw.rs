//! AdamW optimizer implementation

use ndarray::{Array, Dimension};

pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize, // timestep
}

impl AdamW {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            t: 0,
        }
    }

    pub fn with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
        }
    }

    /// Perform a single optimization step
    pub fn step<D: Dimension>(
        &mut self,
        params: &mut Array<f32, D>,
        grads: &Array<f32, D>,
        m: &mut Array<f32, D>,
        v: &mut Array<f32, D>,
    ) {
        self.t += 1;

        let lr = self.learning_rate;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        // Bias correction
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);

        // Update parameters using iterator
        use ndarray::Zip;
        Zip::from(params)
            .and(grads)
            .and(m)
            .and(v)
            .for_each(|p, &g, m_val, v_val| {
                // Update biased first moment estimate
                *m_val = beta1 * *m_val + (1.0 - beta1) * g;

                // Update biased second raw moment estimate
                *v_val = beta2 * *v_val + (1.0 - beta2) * g * g;

                // Compute bias-corrected estimates
                let m_hat = *m_val / bias_correction1;
                let v_hat = *v_val / bias_correction2;

                // Update parameters with weight decay
                *p = *p - lr * (m_hat / (v_hat.sqrt() + eps) + wd * *p);
            });
    }

    pub fn zero_grad<D: Dimension>(&self, grads: &mut Array<f32, D>) {
        grads.fill(0.0);
    }
}
