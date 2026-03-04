//! Logging utilities for training metrics and progress tracking

use log::{info, debug};
use std::time::Instant;

pub struct TrainingLogger {
    start_time: Instant,
    step: usize,
}

impl TrainingLogger {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            step: 0,
        }
    }

    pub fn log_step(&mut self, step: usize, loss: f32, elapsed_ms: f64) {
        self.step = step;
        info!("step {}: train loss {:.6} (took {:.3} ms)", step, loss, elapsed_ms);
    }

    pub fn log_validation(&self, val_loss: f32) {
        info!("val loss {:.6}", val_loss);
    }

    pub fn log_model_info(&self, config: &crate::model::GPT2Config, num_params: usize) {
        info!("[GPT-2]");
        info!("max_seq_len: {}", config.max_seq_len);
        info!("vocab_size: {}", config.vocab_size);
        info!("num_layers: {}", config.num_layers);
        info!("num_heads: {}", config.num_heads);
        info!("channels: {}", config.channels);
        info!("num_parameters: {}", num_params);
    }

    pub fn log_generation(&self, text: &str) {
        info!("generating:");
        info!("---");
        info!("{}", text);
        info!("---");
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

impl Default for TrainingLogger {
    fn default() -> Self {
        Self::new()
    }
}
