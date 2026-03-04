//! GPT-2 model structure and configuration
//! Converts from train_gpt2.c lines 526-763

use crate::Result;
use ndarray::{Array1, Array2, Array3, Array4, Array5};
use std::path::Path;

/// GPT-2 model configuration
#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub padded_vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub channels: usize,
}

impl GPT2Config {
    pub fn gpt2_124m() -> Self {
        Self {
            max_seq_len: 1024,
            vocab_size: 50257,
            padded_vocab_size: 50304,
            num_layers: 12,
            num_heads: 12,
            channels: 768,
        }
    }
}

/// Parameter tensors for GPT-2
pub struct ParameterTensors {
    pub wte: Array2<f32>,      // (V, C) token embeddings
    pub wpe: Array2<f32>,      // (maxT, C) position embeddings
    pub ln1w: Array2<f32>,     // (L, C) layernorm 1 weight
    pub ln1b: Array2<f32>,     // (L, C) layernorm 1 bias
    pub qkvw: Array3<f32>,     // (L, 3*C, C) QKV weight
    pub qkvb: Array2<f32>,     // (L, 3*C) QKV bias
    pub attprojw: Array3<f32>, // (L, C, C) attention projection weight
    pub attprojb: Array2<f32>, // (L, C) attention projection bias
    pub ln2w: Array2<f32>,     // (L, C) layernorm 2 weight
    pub ln2b: Array2<f32>,     // (L, C) layernorm 2 bias
    pub fcw: Array3<f32>,      // (L, 4*C, C) feedforward weight
    pub fcb: Array2<f32>,      // (L, 4*C) feedforward bias
    pub fcprojw: Array3<f32>,  // (L, C, 4*C) feedforward projection weight
    pub fcprojb: Array2<f32>,  // (L, C) feedforward projection bias
    pub lnfw: Array1<f32>,     // (C,) final layernorm weight
    pub lnfb: Array1<f32>,     // (C,) final layernorm bias
}

/// Activation tensors for GPT-2
pub struct ActivationTensors {
    pub encoded: Array3<f32>,   // (B, T, C)
    pub ln1: Array4<f32>,       // (L, B, T, C)
    pub ln1_mean: Array3<f32>,  // (L, B, T)
    pub ln1_rstd: Array3<f32>,  // (L, B, T)
    pub qkv: Array4<f32>,       // (L, B, T, 3*C)
    pub atty: Array4<f32>,      // (L, B, T, C)
    pub preatt: Array5<f32>,    // (L, B, NH, T, T)
    pub att: Array5<f32>,       // (L, B, NH, T, T)
    pub attproj: Array4<f32>,   // (L, B, T, C)
    pub residual2: Array4<f32>, // (L, B, T, C)
    pub ln2: Array4<f32>,       // (L, B, T, C)
    pub ln2_mean: Array3<f32>,  // (L, B, T)
    pub ln2_rstd: Array3<f32>,  // (L, B, T)
    pub fch: Array4<f32>,       // (L, B, T, 4*C)
    pub fch_gelu: Array4<f32>,  // (L, B, T, 4*C)
    pub fcproj: Array4<f32>,    // (L, B, T, C)
    pub residual3: Array4<f32>, // (L, B, T, C)
    pub lnf: Array3<f32>,       // (B, T, C)
    pub lnf_mean: Array2<f32>,  // (B, T)
    pub lnf_rstd: Array2<f32>,  // (B, T)
    pub logits: Array3<f32>,    // (B, T, Vp)
    pub probs: Array3<f32>,     // (B, T, Vp)
    pub losses: Array2<f32>,    // (B, T)
}

/// Main GPT-2 model structure
pub struct GPT2 {
    pub config: GPT2Config,
    pub params: ParameterTensors,
    pub grads: ParameterTensors,
    pub acts: Option<ActivationTensors>,
    pub grads_acts: Option<ActivationTensors>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub mean_loss: f32,
}

impl GPT2 {
    /// Create a new GPT-2 model from a checkpoint file
    pub fn from_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::fs::File;

        let mut file = File::open(path)?;

        // Read header
        let mut header = [0i32; 256];
        for i in 0..256 {
            header[i] = file.read_i32::<LittleEndian>()?;
        }

        // Validate magic number and version
        if header[0] != 20240326 {
            anyhow::bail!("Bad magic model file");
        }
        if header[1] != 3 {
            anyhow::bail!("Bad version in model file");
        }

        // Read hyperparameters
        let config = GPT2Config {
            max_seq_len: header[2] as usize,
            vocab_size: header[3] as usize,
            num_layers: header[4] as usize,
            num_heads: header[5] as usize,
            channels: header[6] as usize,
            padded_vocab_size: header[7] as usize,
        };

        // Read parameters
        // TODO: Implement parameter loading from binary file
        // For now, create zero-initialized parameters
        let params = Self::create_zero_params(&config);
        let grads = Self::create_zero_params(&config);

        Ok(Self {
            config,
            params,
            grads,
            acts: None,
            grads_acts: None,
            batch_size: 0,
            seq_len: 0,
            mean_loss: -1.0,
        })
    }

    fn create_zero_params(config: &GPT2Config) -> ParameterTensors {
        let v = config.padded_vocab_size;
        let c = config.channels;
        let l = config.num_layers;
        let max_t = config.max_seq_len;

        ParameterTensors {
            wte: Array2::zeros((v, c)),
            wpe: Array2::zeros((max_t, c)),
            ln1w: Array2::zeros((l, c)),
            ln1b: Array2::zeros((l, c)),
            qkvw: Array3::zeros((l, 3 * c, c)),
            qkvb: Array2::zeros((l, 3 * c)),
            attprojw: Array3::zeros((l, c, c)),
            attprojb: Array2::zeros((l, c)),
            ln2w: Array2::zeros((l, c)),
            ln2b: Array2::zeros((l, c)),
            fcw: Array3::zeros((l, 4 * c, c)),
            fcb: Array2::zeros((l, 4 * c)),
            fcprojw: Array3::zeros((l, c, 4 * c)),
            fcprojb: Array2::zeros((l, c)),
            lnfw: Array1::zeros(c),
            lnfb: Array1::zeros(c),
        }
    }

    /// Allocate memory for activations and gradients if not already present
    fn ensure_memory(&mut self) {
        if self.acts.is_some() {
            return;
        }

        let B = self.batch_size;
        let T = self.seq_len;
        let C = self.config.channels;
        let L = self.config.num_layers;
        let V = self.config.padded_vocab_size;
        let NH = self.config.num_heads;

        let acts = ActivationTensors {
            encoded: Array3::zeros((B, T, C)),
            ln1: Array4::zeros((L, B, T, C)),
            ln1_mean: Array3::zeros((L, B, T)),
            ln1_rstd: Array3::zeros((L, B, T)),
            qkv: Array4::zeros((L, B, T, 3 * C)),
            atty: Array4::zeros((L, B, T, C)),
            preatt: Array5::zeros((L, B, NH, T, T)),
            att: Array5::zeros((L, B, NH, T, T)),
            attproj: Array4::zeros((L, B, T, C)),
            residual2: Array4::zeros((L, B, T, C)),
            ln2: Array4::zeros((L, B, T, C)),
            ln2_mean: Array3::zeros((L, B, T)),
            ln2_rstd: Array3::zeros((L, B, T)),
            fch: Array4::zeros((L, B, T, 4 * C)),
            fch_gelu: Array4::zeros((L, B, T, 4 * C)),
            fcproj: Array4::zeros((L, B, T, C)),
            residual3: Array4::zeros((L, B, T, C)),
            lnf: Array3::zeros((B, T, C)),
            lnf_mean: Array2::zeros((B, T)),
            lnf_rstd: Array2::zeros((B, T)),
            logits: Array3::zeros((B, T, V)),
            probs: Array3::zeros((B, T, V)),
            losses: Array2::zeros((B, T)),
        };

        // Gradients for activations (needed for backward pass)
        let grads_acts = ActivationTensors {
            encoded: Array3::zeros((B, T, C)),
            ln1: Array4::zeros((L, B, T, C)),
            ln1_mean: Array3::zeros((L, B, T)), // Not strictly needed for gradients but keeping structure
            ln1_rstd: Array3::zeros((L, B, T)),
            qkv: Array4::zeros((L, B, T, 3 * C)),
            atty: Array4::zeros((L, B, T, C)),
            preatt: Array5::zeros((L, B, NH, T, T)),
            att: Array5::zeros((L, B, NH, T, T)),
            attproj: Array4::zeros((L, B, T, C)),
            residual2: Array4::zeros((L, B, T, C)),
            ln2: Array4::zeros((L, B, T, C)),
            ln2_mean: Array3::zeros((L, B, T)),
            ln2_rstd: Array3::zeros((L, B, T)),
            fch: Array4::zeros((L, B, T, 4 * C)),
            fch_gelu: Array4::zeros((L, B, T, 4 * C)),
            fcproj: Array4::zeros((L, B, T, C)),
            residual3: Array4::zeros((L, B, T, C)),
            lnf: Array3::zeros((B, T, C)),
            lnf_mean: Array2::zeros((B, T)),
            lnf_rstd: Array2::zeros((B, T)),
            logits: Array3::zeros((B, T, V)),
            probs: Array3::zeros((B, T, V)),
            losses: Array2::zeros((B, T)),
        };

        self.acts = Some(acts);
        self.grads_acts = Some(grads_acts);
    }

    /// Forward pass
    pub fn forward(&mut self, inputs: &Array2<i32>, targets: Option<&Array2<i32>>) -> Result<()> {
        // TODO: Implement forward pass using layer functions
        Ok(())
    }

    /// Backward pass
    pub fn backward(&mut self) -> Result<()> {
        // TODO: Implement backward pass
        Ok(())
    }

    /// Count total number of parameters
    pub fn num_parameters(&self) -> usize {
        let v = self.config.padded_vocab_size;
        let c = self.config.channels;
        let l = self.config.num_layers;
        let max_t = self.config.max_seq_len;

        v * c
            + max_t * c
            + l * c
            + l * c
            + l * 3 * c * c
            + l * 3 * c
            + l * c * c
            + l * c
            + l * c
            + l * c
            + l * 4 * c * c
            + l * 4 * c
            + l * c * 4 * c
            + l * c
            + c
            + c
    }
}

#[cfg(feature = "cuda")]
pub mod gpt2_cuda;
#[cfg(feature = "cuda")]
pub use gpt2_cuda::GPT2Cuda;
