use crate::cuda;
use crate::model::GPT2Config;
use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use std::ffi::c_void;
use std::fs::File;
use std::io::Read;
use std::ptr;

/// Parameter pointers into the allocated memory block
struct ParameterPointers {
    wte: *mut f32,
    wpe: *mut f32,
    ln1w: *mut f32,
    ln1b: *mut f32,
    qkvw: *mut f32,
    qkvb: *mut f32,
    attprojw: *mut f32,
    attprojb: *mut f32,
    ln2w: *mut f32,
    ln2b: *mut f32,
    fcw: *mut f32,
    fcb: *mut f32,
    fcprojw: *mut f32,
    fcprojb: *mut f32,
    lnfw: *mut f32,
    lnfb: *mut f32,
}

/// Activation pointers into the allocated memory block
struct ActivationPointers {
    encoded: *mut f32,
    ln1: *mut f32,
    ln1_mean: *mut f32,
    ln1_rstd: *mut f32,
    atty: *mut f32,
    att: *mut f32,
    residual2: *mut f32,
    ln2: *mut f32,
    ln2_mean: *mut f32,
    ln2_rstd: *mut f32,
    fch: *mut f32,
    fch_gelu: *mut f32,
    residual3: *mut f32,
    lnf: *mut f32,
    lnf_mean: *mut f32,
    lnf_rstd: *mut f32,
    losses: *mut f32,
    qkvr: *mut f32,
    output: *mut f32,
    scratch_bt4c: *mut f32,
    scratch_btc: *mut f32,
}

pub struct GPT2Cuda {
    pub config: GPT2Config,
    params_memory: *mut c_void,
    pub grads_memory: *mut f32,
    pub m_memory: *mut f32,
    pub v_memory: *mut f32,
    acts_memory: *mut c_void,
    pub inputs_device: *mut i32,
    pub targets_device: *mut i32,
    pub mean_loss_device: *mut f32, // reuse for global norm
    pub inputs_cpu: Vec<i32>,
    workload_indices: Vec<i32>,
    bucket_info: Vec<i32>,
    seed: u32,
    pub num_parameters: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub mean_loss: f32,
}

impl GPT2Cuda {
    pub fn from_checkpoint(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;

        let mut header = [0i32; 256];
        for i in 0..256 {
            header[i] = file.read_i32::<LittleEndian>()?;
        }

        if header[0] != 20240326 {
            anyhow::bail!("Bad magic model file");
        }

        let config = GPT2Config {
            max_seq_len: header[2] as usize,
            vocab_size: header[3] as usize,
            num_layers: header[4] as usize,
            num_heads: header[5] as usize,
            channels: header[6] as usize,
            padded_vocab_size: header[7] as usize,
        };

        cuda::init();

        let mut model = Self {
            config,
            params_memory: ptr::null_mut(),
            grads_memory: ptr::null_mut(),
            m_memory: ptr::null_mut(),
            v_memory: ptr::null_mut(),
            acts_memory: ptr::null_mut(),
            inputs_device: ptr::null_mut(),
            targets_device: ptr::null_mut(),
            mean_loss_device: ptr::null_mut(),
            inputs_cpu: Vec::new(),
            workload_indices: Vec::new(),
            bucket_info: Vec::new(),
            seed: 1337,
            num_parameters: 0,
            batch_size: 0,
            seq_len: 0,
            mean_loss: -1.0,
        };

        model.allocate_weights();

        let num_params = model.num_parameters;
        let param_bytes = num_params * std::mem::size_of::<f32>();

        // Skip the header (256 integers = 1024 bytes)
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(1024))?;

        let mut params_cpu = vec![0u8; param_bytes];
        file.read_exact(&mut params_cpu)?;

        unsafe {
            cuda::memcpy_htod_c(
                model.params_memory,
                params_cpu.as_ptr() as *const c_void,
                param_bytes,
            );
        }

        Ok(model)
    }

    fn get_num_parameters(&self) -> usize {
        let c = self.config.channels;
        let l = self.config.num_layers;
        let v = self.config.padded_vocab_size;
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

    fn allocate_weights(&mut self) {
        let num_params = self.get_num_parameters();
        self.num_parameters = num_params;
        let bytes = num_params * std::mem::size_of::<f32>();

        unsafe {
            cuda::malloc_c(&mut self.params_memory, bytes);
            cuda::malloc_c(&mut self.grads_memory as *mut _ as *mut *mut c_void, bytes);
            cuda::malloc_c(&mut self.m_memory as *mut _ as *mut *mut c_void, bytes);
            cuda::malloc_c(&mut self.v_memory as *mut _ as *mut *mut c_void, bytes);

            cuda::memset_c(self.grads_memory as *mut c_void, 0, bytes);
            cuda::memset_c(self.m_memory as *mut c_void, 0, bytes);
            cuda::memset_c(self.v_memory as *mut c_void, 0, bytes);
        }
    }

    pub fn allocate_state(&mut self, B: usize, T: usize) {
        self.batch_size = B;
        self.seq_len = T;

        let c = self.config.channels;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let vp = self.config.padded_vocab_size;

        let act_size = (
            B * T * c + // encoded
            l * B * T * c + // ln1
            l * B * T + // ln1_mean
            l * B * T + // ln1_rstd
            l * B * T * c + // atty
            l * B * nh * T * T + // att
            l * B * T * c + // residual2
            l * B * T * c + // ln2
            l * B * T + // ln2_mean
            l * B * T + // ln2_rstd
            l * B * T * 4 * c + // fch
            B * T * 4 * c + // fch_gelu (recompute mode)
            l * B * T * c + // residual3
            B * T * c + // lnf
            B * T + // lnf_mean
            B * T + // lnf_rstd
            B * T + // losses
            l * B * T * 3 * c + // qkvr
            B * T * vp.max(3 * c).max(nh * T) + // output
            B * T * 4 * c + // scratch_bt4c
            B * T * c
            // scratch_btc
        ) * std::mem::size_of::<f32>();

        unsafe {
            cuda::malloc_c(&mut self.acts_memory, act_size);
            cuda::memset_c(self.acts_memory, 0, act_size);

            cuda::malloc_c(
                &mut self.inputs_device as *mut _ as *mut *mut c_void,
                B * T * std::mem::size_of::<i32>(),
            );
            cuda::malloc_c(
                &mut self.targets_device as *mut _ as *mut *mut c_void,
                B * T * std::mem::size_of::<i32>(),
            );
            cuda::malloc_c(
                &mut self.mean_loss_device as *mut _ as *mut *mut c_void,
                1024 * std::mem::size_of::<f32>(),
            );
        }

        self.inputs_cpu = vec![0i32; B * T];
        let x128_size = 4; // FP32 size (16 bytes / 4 bytes)
        let warp_size = 32;
        let num_c_groups = (c + (x128_size * warp_size) - 1) / (x128_size * warp_size);
        self.workload_indices = vec![0i32; B * T * num_c_groups];
        self.bucket_info = vec![0i32; B * T * num_c_groups * 4]; // int4 is 4 ints
    }

    fn get_param_ptrs(&self) -> ParameterPointers {
        let c = self.config.channels;
        let l = self.config.num_layers;
        let v = self.config.padded_vocab_size;
        let max_t = self.config.max_seq_len;

        let mut offset = 0usize;
        let base = self.params_memory as *mut f32;

        unsafe {
            let wte = base.add(offset);
            offset += v * c;
            let wpe = base.add(offset);
            offset += max_t * c;
            let ln1w = base.add(offset);
            offset += l * c;
            let ln1b = base.add(offset);
            offset += l * c;
            let qkvw = base.add(offset);
            offset += l * 3 * c * c;
            let qkvb = base.add(offset);
            offset += l * 3 * c;
            let attprojw = base.add(offset);
            offset += l * c * c;
            let attprojb = base.add(offset);
            offset += l * c;
            let ln2w = base.add(offset);
            offset += l * c;
            let ln2b = base.add(offset);
            offset += l * c;
            let fcw = base.add(offset);
            offset += l * 4 * c * c;
            let fcb = base.add(offset);
            offset += l * 4 * c;
            let fcprojw = base.add(offset);
            offset += l * c * 4 * c;
            let fcprojb = base.add(offset);
            offset += l * c;
            let lnfw = base.add(offset);
            offset += c;
            let lnfb = base.add(offset);

            ParameterPointers {
                wte,
                wpe,
                ln1w,
                ln1b,
                qkvw,
                qkvb,
                attprojw,
                attprojb,
                ln2w,
                ln2b,
                fcw,
                fcb,
                fcprojw,
                fcprojb,
                lnfw,
                lnfb,
            }
        }
    }

    fn get_grad_ptrs(&self) -> ParameterPointers {
        let c = self.config.channels;
        let l = self.config.num_layers;
        let v = self.config.padded_vocab_size;
        let max_t = self.config.max_seq_len;

        let mut offset = 0usize;
        let base = self.grads_memory as *mut f32;

        unsafe {
            let wte = base.add(offset);
            offset += v * c;
            let wpe = base.add(offset);
            offset += max_t * c;
            let ln1w = base.add(offset);
            offset += l * c;
            let ln1b = base.add(offset);
            offset += l * c;
            let qkvw = base.add(offset);
            offset += l * 3 * c * c;
            let qkvb = base.add(offset);
            offset += l * 3 * c;
            let attprojw = base.add(offset);
            offset += l * c * c;
            let attprojb = base.add(offset);
            offset += l * c;
            let ln2w = base.add(offset);
            offset += l * c;
            let ln2b = base.add(offset);
            offset += l * c;
            let fcw = base.add(offset);
            offset += l * 4 * c * c;
            let fcb = base.add(offset);
            offset += l * 4 * c;
            let fcprojw = base.add(offset);
            offset += l * c * 4 * c;
            let fcprojb = base.add(offset);
            offset += l * c;
            let lnfw = base.add(offset);
            offset += c;
            let lnfb = base.add(offset);

            ParameterPointers {
                wte,
                wpe,
                ln1w,
                ln1b,
                qkvw,
                qkvb,
                attprojw,
                attprojb,
                ln2w,
                ln2b,
                fcw,
                fcb,
                fcprojw,
                fcprojb,
                lnfw,
                lnfb,
            }
        }
    }

    fn get_act_ptrs(&self) -> ActivationPointers {
        let b = self.batch_size;
        let t = self.seq_len;
        let c = self.config.channels;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let vp = self.config.padded_vocab_size;

        let mut offset = 0usize;
        let base = self.acts_memory as *mut f32;

        unsafe {
            let encoded = base.add(offset);
            offset += b * t * c;
            let ln1 = base.add(offset);
            offset += l * b * t * c;
            let ln1_mean = base.add(offset);
            offset += l * b * t;
            let ln1_rstd = base.add(offset);
            offset += l * b * t;
            let atty = base.add(offset);
            offset += l * b * t * c;
            let att = base.add(offset);
            offset += l * b * nh * t * t;
            let residual2 = base.add(offset);
            offset += l * b * t * c;
            let ln2 = base.add(offset);
            offset += l * b * t * c;
            let ln2_mean = base.add(offset);
            offset += l * b * t;
            let ln2_rstd = base.add(offset);
            offset += l * b * t;
            let fch = base.add(offset);
            offset += l * b * t * 4 * c;
            let fch_gelu = base.add(offset);
            offset += b * t * 4 * c;
            let residual3 = base.add(offset);
            offset += l * b * t * c;
            let lnf = base.add(offset);
            offset += b * t * c;
            let lnf_mean = base.add(offset);
            offset += b * t;
            let lnf_rstd = base.add(offset);
            offset += b * t;
            let losses = base.add(offset);
            offset += b * t;
            let qkvr = base.add(offset);
            offset += l * b * t * 3 * c;
            let output = base.add(offset);
            offset += b * t * vp.max(3 * c).max(nh * t);
            let scratch_bt4c = base.add(offset);
            offset += b * t * 4 * c;
            let scratch_btc = base.add(offset);

            ActivationPointers {
                encoded,
                ln1,
                ln1_mean,
                ln1_rstd,
                atty,
                att,
                residual2,
                ln2,
                ln2_mean,
                ln2_rstd,
                fch,
                fch_gelu,
                residual3,
                lnf,
                lnf_mean,
                lnf_rstd,
                losses,
                qkvr,
                output,
                scratch_bt4c,
                scratch_btc,
            }
        }
    }

    pub fn forward(&mut self, inputs: &[i32], targets: Option<&[i32]>) -> Result<f32> {
        let b = self.batch_size;
        let t = self.seq_len;
        let c = self.config.channels;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let v = self.config.vocab_size;
        let vp = self.config.padded_vocab_size;

        unsafe {
            cuda::memcpy_htod_c(
                self.inputs_device as *mut c_void,
                inputs.as_ptr() as *const c_void,
                inputs.len() * 4,
            );
            self.inputs_cpu.copy_from_slice(inputs);

            let params = self.get_param_ptrs();

            // Simplified forward - encoder
            let acts_base = self.acts_memory as *mut f32;
            let encoded = acts_base;

            cuda::encoder_forward_c(
                encoded,
                self.inputs_device,
                params.wte,
                params.wpe,
                b as i32,
                t as i32,
                c as i32,
            );

            let mut residual = encoded;

            for l in 0..l {
                let l_ln1w = params.ln1w.add(l * c as usize);
                let l_ln1b = params.ln1b.add(l * c as usize);
                let l_qkvw = params.qkvw.add(l * 3 * c as usize * c as usize);
                let l_qkvb = params.qkvb.add(l * 3 * c as usize);
                let l_attprojw = params.attprojw.add(l * c as usize * c as usize);
                let l_attprojb = params.attprojb.add(l * c as usize);
                let l_ln2w = params.ln2w.add(l * c as usize);
                let l_ln2b = params.ln2b.add(l * c as usize);
                let l_fcw = params.fcw.add(l * 4 * c as usize * c as usize);
                let l_fcb = params.fcb.add(l * 4 * c as usize);
                let l_fcprojw = params.fcprojw.add(l * c as usize * 4 * c as usize);
                let l_fcprojb = params.fcprojb.add(l * c as usize);

                let acts = self.get_act_ptrs();
                let l_ln1 = acts.ln1.add(l * b * t * c as usize);
                let l_ln1_mean = acts.ln1_mean.add(l * b * t);
                let l_ln1_rstd = acts.ln1_rstd.add(l * b * t);
                let l_qkv = acts.qkvr.add(l * b * t * 3 * c as usize); // llmc names it qkvr
                let l_atty = acts.atty.add(l * b * t * c as usize);
                let l_att = acts.att.add(l * b * nh as usize * t * t);
                // Note: acts.attproj in train_gpt2.c is scratchpad memory here. We use acts.scratch_bt4c.
                let l_attproj = acts.scratch_bt4c;
                let l_residual2 = acts.residual2.add(l * b * t * c as usize);
                let l_ln2 = acts.ln2.add(l * b * t * c as usize);
                let l_ln2_mean = acts.ln2_mean.add(l * b * t);
                let l_ln2_rstd = acts.ln2_rstd.add(l * b * t);
                let l_fch = acts.fch.add(l * b * t * 4 * c as usize);
                let l_fch_gelu = acts.fch_gelu; // assuming recompute < 1 for now (buffer shared or per-layer if needed)
                let l_fcproj = acts.scratch_bt4c; // scratch buffer
                let l_residual3 = acts.residual3.add(l * b * t * c as usize);

                cuda::layernorm_forward_c(
                    l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, b as i32, t as i32,
                    c as i32,
                );
                cuda::matmul_forward_c(
                    l_qkv,
                    l_ln1,
                    l_qkvw,
                    l_qkvb,
                    b as i32,
                    t as i32,
                    c as i32,
                    (3 * c) as i32,
                );
                cuda::attention_forward_c(
                    l_atty,
                    l_qkv,
                    l_att,
                    acts.scratch_btc,
                    b as i32,
                    t as i32,
                    c as i32,
                    nh as i32,
                );
                cuda::matmul_forward_c(
                    l_attproj, l_atty, l_attprojw, l_attprojb, b as i32, t as i32, c as i32,
                    c as i32,
                );
                cuda::fused_residual_forward5_c(
                    l_residual2,
                    l_ln2,
                    l_ln2_mean,
                    l_ln2_rstd,
                    residual,
                    l_attproj,
                    l_ln2w,
                    l_ln2b,
                    (b * t) as i32,
                    c as i32,
                );

                cuda::matmul_forward_c(
                    l_fch,
                    l_ln2,
                    l_fcw,
                    l_fcb,
                    b as i32,
                    t as i32,
                    c as i32,
                    (4 * c) as i32,
                );
                cuda::gelu_forward_c(
                    acts.fch_gelu,
                    acts.fch.add(l * b * t * 4 * c as usize),
                    b * t * 4 * c,
                );
                cuda::matmul_forward_c(
                    l_fcproj,
                    l_fch_gelu,
                    l_fcprojw,
                    l_fcprojb,
                    b as i32,
                    t as i32,
                    (4 * c) as i32,
                    c as i32,
                );

                // wait, the final layer outputs to residual3.
                residual = l_residual3;
                let l_next_ln_out = if l == self.config.num_layers - 1 {
                    acts.lnf
                } else {
                    acts.ln1.add((l + 1) * b * t * c as usize)
                };
                let l_next_ln_mean = if l == self.config.num_layers - 1 {
                    acts.lnf_mean
                } else {
                    acts.ln1_mean.add((l + 1) * b * t)
                };
                let l_next_ln_rstd = if l == self.config.num_layers - 1 {
                    acts.lnf_rstd
                } else {
                    acts.ln1_rstd.add((l + 1) * b * t)
                };
                let l_next_ln_weight = if l == self.config.num_layers - 1 {
                    params.lnfw
                } else {
                    params.ln1w.add((l + 1) * c as usize)
                };
                let l_next_ln_bias = if l == self.config.num_layers - 1 {
                    params.lnfb
                } else {
                    params.ln1b.add((l + 1) * c as usize)
                };

                cuda::fused_residual_forward5_c(
                    l_residual3,
                    l_next_ln_out,
                    l_next_ln_mean,
                    l_next_ln_rstd,
                    l_residual2,
                    l_fcproj,
                    l_next_ln_weight,
                    l_next_ln_bias,
                    (b * t) as i32,
                    c as i32,
                );
            }

            let acts = self.get_act_ptrs();
            cuda::matmul_forward_c(
                acts.output,
                acts.lnf,
                params.wte,
                std::ptr::null_mut(),
                b as i32,
                t as i32,
                c as i32,
                vp as i32,
            );

            // Compute loss if targets provided
            // Zero out losses for each forward pass
            cuda::memset_c(
                acts.losses as *mut c_void,
                0,
                b * t * std::mem::size_of::<f32>(),
            );

            if let Some(tgt) = targets {
                cuda::memcpy_htod_c(
                    self.targets_device as *mut c_void,
                    tgt.as_ptr() as *const c_void,
                    tgt.len() * 4,
                );

                cuda::fused_classifier_c(
                    acts.output,
                    acts.losses,
                    acts.output,
                    self.targets_device,
                    b as i32,
                    t as i32,
                    v as i32,
                    vp as i32,
                );

                let mut cpu_losses = vec![0f32; b * t];
                cuda::memcpy_dtoh_c(
                    cpu_losses.as_mut_ptr() as *mut c_void,
                    acts.losses as *const c_void,
                    b * t * 4,
                );

                self.mean_loss = cpu_losses.iter().sum::<f32>() / (b * t) as f32;
            }

            cuda::device_synchronize_c();
        }

        Ok(self.mean_loss)
    }

    pub fn backward(&mut self) -> Result<()> {
        let b = self.batch_size;
        let t = self.seq_len;
        let c = self.config.channels;
        let v = self.config.vocab_size;
        let vp = self.config.padded_vocab_size;
        let nh = self.config.num_heads;

        unsafe {
            let params = self.get_param_ptrs();
            let grads = self.get_grad_ptrs();
            let acts = self.get_act_ptrs();

            // Zero out gradients for the backward pass
            let num_param_bytes = self.num_parameters * std::mem::size_of::<f32>();
            cuda::memset_c(self.grads_memory as *mut c_void, 0, num_param_bytes);

            // Zero out losses (effectively the start of backward in llm.c)
            cuda::memset_c(
                acts.losses as *mut c_void,
                0,
                b * t * std::mem::size_of::<f32>(),
            );

            let dresidual = acts.scratch_btc; // The main cumulative gradient buffer
                                              // IMPORTANT: Must zero out dresidual at the start of backward pass
            cuda::memset_c(
                dresidual as *mut c_void,
                0,
                b * t * c * std::mem::size_of::<f32>(),
            );

            // 1. fused_classifier_backward
            // dlogits is 'output' in acts (aliased).
            cuda::fused_classifier_backward_c(
                acts.output,
                acts.losses,
                self.targets_device,
                b as i32,
                t as i32,
                v as i32,
                vp as i32,
            );

            // 2. Classifier matmul backward: calculate dresidual and dwte
            // dresidual: main BTC-sized residual gradient, lives in scratch_btc
            // acts.output: dlogits (B, T, Vp)
            // acts.lnf: input to classifier matmul
            // params.wte: weight matrix (Vp, C)
            let dresidual = acts.scratch_btc;
            cuda::memset_c(
                dresidual as *mut c_void,
                0,
                b * t * c * std::mem::size_of::<f32>(),
            );
            let dl_bt4c = acts.scratch_bt4c;
            let scratch_f = acts.output; // Scratchpad for layernorm bias reduction

            cuda::matmul_backward_c(
                dl_bt4c,         // dinp: output of classifier matmul in llm.c is scratch_bt4c (B, T, C)
                grads.wte,       // dweight: accumulates into dwte
                ptr::null_mut(), // dbias
                acts.output,     // dout: dlogits (B, T, Vp)
                acts.lnf,        // inp: lnf output (B, T, C)
                params.wte,      // weight (Vp, C)
                ptr::null_mut(), // dbias_buffer
                b as i32,
                t as i32,
                c as i32,
                vp as i32,
            );

            // 3. Final LNF layernorm backward
            // dresidual accumulates into scratch_btc
            let last_residual = acts.residual3.add((self.config.num_layers - 1) * b * t * c);
            cuda::layernorm_backward_c(
                dresidual,     // dinp: accumulates INTO dresidual (acts.scratch_btc)
                grads.lnfw,    // dweight
                grads.lnfb,    // dbias
                scratch_f,     // scratch
                dl_bt4c,       // dout: from classifier matmul above
                last_residual, // inp: input to lnf in forward
                params.lnfw,
                acts.lnf_mean,
                acts.lnf_rstd,
                b as i32,
                t as i32,
                c as i32,
            );

            // 4. Layers backward
            for l in (0..self.config.num_layers).rev() {
                let residual_inp = if l == 0 {
                    acts.encoded
                } else {
                    acts.residual3.add((l - 1) * b * t * c)
                };

                // Layer parameters and gradients
                let l_ln1w = params.ln1w.add(l * c);
                let gl_ln1w = grads.ln1w.add(l * c);
                let gl_ln1b = grads.ln1b.add(l * c);

                let l_qkvw = params.qkvw.add(l * 3 * c * c);
                let gl_qkvw = grads.qkvw.add(l * 3 * c * c);
                let gl_qkvb = grads.qkvb.add(l * 3 * c);

                let l_attprojw = params.attprojw.add(l * c * c);
                let gl_attprojw = grads.attprojw.add(l * c * c);
                let gl_attprojb = grads.attprojb.add(l * c);

                let l_ln2w = params.ln2w.add(l * c);
                let gl_ln2w = grads.ln2w.add(l * c);
                let gl_ln2b = grads.ln2b.add(l * c);

                let l_fcw = params.fcw.add(l * 4 * c * c);
                let gl_fcw = grads.fcw.add(l * 4 * c * c);
                let gl_fcb = grads.fcb.add(l * 4 * c);

                let l_fcprojw = params.fcprojw.add(l * c * 4 * c);
                let gl_fcprojw = grads.fcprojw.add(l * c * 4 * c);
                let gl_fcprojb = grads.fcprojb.add(l * c);

                // Layer activations
                let l_ln1 = acts.ln1.add(l * b * t * c);
                let l_ln1_mean = acts.ln1_mean.add(l * b * t);
                let l_ln1_rstd = acts.ln1_rstd.add(l * b * t);
                let l_qkv = acts.qkvr.add(l * b * t * 3 * c);
                let l_atty = acts.atty.add(l * b * t * c);
                let l_att = acts.att.add(l * b * nh * t * t);
                let l_residual2 = acts.residual2.add(l * b * t * c);
                let l_ln2 = acts.ln2.add(l * b * t * c);
                let l_ln2_mean = acts.ln2_mean.add(l * b * t);
                let l_ln2_rstd = acts.ln2_rstd.add(l * b * t);
                let l_fch = acts.fch.add(l * b * t * 4 * c);
                let l_fch_gelu = acts.fch_gelu; // Shared scratch buffer

                // Recompute GELU for the backward pass
                cuda::gelu_forward_c(l_fch_gelu, l_fch, b * t * 4 * c);

                // IMPORTANT: In llm.c, dl_btc reuses the residual3 memory for the current layer
                // so it doesn't alias with dresidual (scratch_btc)
                let dl_btc = acts.residual3.add(l * b * t * c);

                // 4.1 MLP Block Backward
                // Matmul backward for FC projection
                cuda::matmul_backward_c(
                    dl_bt4c,    // dinp
                    gl_fcprojw, // dweight
                    gl_fcprojb, // dbias
                    dresidual,  // dout
                    l_fch_gelu, // inp
                    l_fcprojw,  // weight
                    scratch_f,  // scratchpad
                    b as i32,
                    t as i32,
                    (4 * c) as i32,
                    c as i32,
                );

                // GELU backward
                cuda::gelu_backward_c(dl_bt4c, l_fch, dl_bt4c, b * t * 4 * c);

                // Matmul backward for FC
                cuda::matmul_backward_c(
                    dl_btc,    // dinp
                    gl_fcw,    // dweight
                    gl_fcb,    // dbias
                    dl_bt4c,   // dout
                    l_ln2,     // inp
                    l_fcw,     // weight
                    scratch_f, // scratchpad
                    b as i32,
                    t as i32,
                    c as i32,
                    (4 * c) as i32,
                );

                // Layernorm 2 backward
                cuda::layernorm_backward_c(
                    dresidual,   // dinp: accumulates into dresidual
                    gl_ln2w,     // dweight
                    gl_ln2b,     // dbias
                    scratch_f,   // scratch
                    dl_btc,      // dout
                    l_residual2, // inp
                    l_ln2w,
                    l_ln2_mean,
                    l_ln2_rstd,
                    b as i32,
                    t as i32,
                    c as i32,
                );

                // 4.2 Attention Block Backward
                // Matmul backward for attention projection
                cuda::matmul_backward_c(
                    dl_btc,      // dinp
                    gl_attprojw, // dweight
                    gl_attprojb, // dbias
                    dresidual,   // dout
                    l_atty,      // inp
                    l_attprojw,  // weight
                    scratch_f,   // scratchpad
                    b as i32,
                    t as i32,
                    c as i32,
                    c as i32,
                );

                // Attention backward
                cuda::attention_backward_c(
                    dl_bt4c,   // dinp: BxTx3C
                    l_fch,     // scratch buffer
                    scratch_f, // scratch buffer (reusing acts.output)
                    l_atty,    // scratch buffer
                    dl_btc,    // dout
                    l_qkv, l_att, b as i32, t as i32, c as i32, nh as i32,
                );

                // Matmul backward for QKV
                cuda::matmul_backward_c(
                    dl_btc,    // dinp
                    gl_qkvw,   // dweight
                    gl_qkvb,   // dbias
                    dl_bt4c,   // dout
                    l_ln1,     // inp
                    l_qkvw,    // weight
                    scratch_f, // scratchpad
                    b as i32,
                    t as i32,
                    c as i32,
                    (3 * c) as i32,
                );

                // Layernorm 1 backward
                cuda::layernorm_backward_c(
                    dresidual,    // dinp: accumulates into dresidual
                    gl_ln1w,      // dweight
                    gl_ln1b,      // dbias
                    scratch_f,    // scratch
                    dl_btc,       // dout
                    residual_inp, // inp
                    l_ln1w,
                    l_ln1_mean,
                    l_ln1_rstd,
                    b as i32,
                    t as i32,
                    c as i32,
                );
            }

            // 5. Encoder backward
            cuda::encoder_backward_c(
                grads.wte,
                grads.wpe,
                dresidual, // dout
                self.inputs_device,
                self.inputs_cpu.as_ptr(),
                acts.output, // Using output acts as scratchpad
                self.workload_indices.as_mut_ptr(),
                self.bucket_info.as_mut_ptr(),
                b as i32,
                t as i32,
                c as i32,
                self.seed,
            );

            self.seed += 1;
        }
        Ok(())
    }

    pub fn clip_gradients(&mut self, max_norm: f32) -> Result<f32> {
        unsafe {
            cuda::global_norm_c(
                self.mean_loss_device,
                self.grads_memory,
                self.num_parameters,
            );
            let mut norm_sq = 0.0f32;
            cuda::memcpy_dtoh_c(
                &mut norm_sq as *mut f32 as *mut c_void,
                self.mean_loss_device as *const c_void,
                4,
            );
            let norm = norm_sq.sqrt();
            if norm > max_norm {
                let scale = max_norm / (norm + 1e-6);
                cuda::scale_grads_c(self.grads_memory, scale, self.num_parameters);
            }
            Ok(norm)
        }
    }

    pub fn step(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
    ) -> Result<()> {
        unsafe {
            let mut cpu_buf = vec![0f32; self.num_parameters];
            cuda::memcpy_dtoh_c(
                cpu_buf.as_mut_ptr() as *mut c_void,
                self.grads_memory as *const c_void,
                self.num_parameters * 4,
            );
            let has_nan = cpu_buf.iter().any(|&x| x.is_nan() || x.is_infinite());
            if has_nan {
                println!(
                    "NaN/Inf detected in GRADS right before AdamW at step {}",
                    step
                );
            }

            cuda::adamw_step_c(
                self.params_memory as *mut f32,
                self.grads_memory as *mut f32,
                self.m_memory as *mut f32,
                self.v_memory as *mut f32,
                learning_rate,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                self.num_parameters,
            );
            cuda::device_synchronize_c();

            let mut cpu_buf = vec![0f32; self.num_parameters];
            cuda::memcpy_dtoh_c(
                cpu_buf.as_mut_ptr() as *mut c_void,
                self.params_memory as *const c_void,
                self.num_parameters * 4,
            );
            let has_nan = cpu_buf.iter().any(|&x| x.is_nan() || x.is_infinite());
            if has_nan {
                println!("NaN/Inf detected in params after step {}", step);
            }
        }
        Ok(())
    }

    pub fn check_diagnostics(&self, step: i32) {
        unsafe {
            let num_params = self.num_parameters;
            let mut grads_cpu = vec![0.0f32; num_params];
            let mut params_cpu = vec![0.0f32; num_params];

            cuda::memcpy_dtoh_c(
                grads_cpu.as_mut_ptr() as *mut f32 as *mut std::ffi::c_void,
                self.grads_memory as *const std::ffi::c_void,
                num_params * 4,
            );
            cuda::memcpy_dtoh_c(
                params_cpu.as_mut_ptr() as *mut f32 as *mut std::ffi::c_void,
                self.params_memory as *const std::ffi::c_void,
                num_params * 4,
            );

            let mut grad_norm_sq = 0.0f64;
            let mut param_norm_sq = 0.0f64;
            let mut has_nan_grad = false;
            let mut has_nan_param = false;

            for i in 0..num_params {
                let g = grads_cpu[i] as f64;
                let p = params_cpu[i] as f64;

                if g.is_nan() || g.is_infinite() {
                    has_nan_grad = true;
                } else {
                    grad_norm_sq += g * g;
                }

                if p.is_nan() || p.is_infinite() {
                    has_nan_param = true;
                } else {
                    param_norm_sq += p * p;
                }
            }

            let grad_norm = grad_norm_sq.sqrt();
            let param_norm = param_norm_sq.sqrt();

            println!(
                "[Step {}] Grad Norm: {:.6}, Param Norm: {:.6}{}{}",
                step,
                grad_norm,
                param_norm,
                if has_nan_grad { " (NaN in Grads!)" } else { "" },
                if has_nan_param {
                    " (NaN in Params!)"
                } else {
                    ""
                }
            );
        }
    }
}

impl Drop for GPT2Cuda {
    fn drop(&mut self) {
        unsafe {
            if !self.params_memory.is_null() {
                cuda::free_c(self.params_memory);
            }
            if !self.grads_memory.is_null() {
                cuda::free_c(self.grads_memory as *mut c_void);
            }
            if !self.m_memory.is_null() {
                cuda::free_c(self.m_memory as *mut c_void);
            }
            if !self.v_memory.is_null() {
                cuda::free_c(self.v_memory as *mut c_void);
            }
            if !self.acts_memory.is_null() {
                cuda::free_c(self.acts_memory);
            }
            if !self.mean_loss_device.is_null() {
                cuda::free_c(self.mean_loss_device as *mut c_void);
            }
            if !self.inputs_device.is_null() {
                cuda::free_c(self.inputs_device as *mut c_void);
            }
            if !self.targets_device.is_null() {
                cuda::free_c(self.targets_device as *mut c_void);
            }
            cuda::cleanup();
        }
    }
}
