//! Main training binary for GPT-2 with CUDA support

use anyhow::Result;
use clap::Parser;
use env_logger;
use log::info;

#[cfg(feature = "cuda")]
use llm_rs::GPT2Cuda;

use llm_rs::DataLoader;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model checkpoint
    #[arg(short, long, default_value = "../gpt2_124M.bin")]
    checkpoint: String,

    /// Path to training data
    #[arg(
        short = 'i',
        long,
        default_value = "../dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
    )]
    train_data: String,

    /// Path to validation data
    #[arg(
        short = 'v',
        long,
        default_value = "../dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
    )]
    val_data: String,

    /// Batch size
    #[arg(short, long, default_value_t = 4)]
    batch_size: usize,

    /// Sequence length
    #[arg(short = 't', long, default_value_t = 64)]
    seq_len: usize,

    /// Number of training steps
    #[arg(short = 'n', long, default_value_t = 400)]
    num_steps: usize,

    /// Learning rate
    #[arg(short, long, default_value_t = 0.0001)]
    learning_rate: f32,

    /// Weight decay
    #[arg(short, long, default_value_t = 0.1)]
    weight_decay: f32,

    /// Gradient clipping threshold
    #[arg(short, long, default_value_t = 1.0)]
    grad_clip: f32,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    info!("Starting llm_rs training");
    info!("Checkpoint: {}", args.checkpoint);
    info!(
        "Batch size: {}, Sequence length: {}",
        args.batch_size, args.seq_len
    );

    #[cfg(feature = "cuda")]
    {
        info!("Using CUDA backend");
        let mut model = GPT2Cuda::from_checkpoint(&args.checkpoint)?;

        info!("[GPT-2 CUDA]");
        info!("max_seq_len: {}", model.config.max_seq_len);
        info!("vocab_size: {}", model.config.vocab_size);
        info!("num_layers: {}", model.config.num_layers);
        info!("num_heads: {}", model.config.num_heads);
        info!("channels: {}", model.config.channels);
        info!("num_parameters: {}", model.num_parameters);

        model.allocate_state(args.batch_size, args.seq_len);

        let mut train_loader = DataLoader::new(&args.train_data, args.batch_size, args.seq_len)?;
        let mut val_loader = DataLoader::new(&args.val_data, args.batch_size, args.seq_len)?;

        info!("train dataset num_batches: {}", train_loader.num_batches());
        info!("val dataset num_batches: {}", val_loader.num_batches());

        // Initial validation
        info!("Computing initial validation loss...");
        let (val_inputs, val_targets) = val_loader.next_batch()?;
        model.forward(
            val_inputs.as_slice().unwrap(),
            Some(val_targets.as_slice().unwrap()),
        )?;
        info!("val loss {:.6}", model.mean_loss);

        // Training loop
        info!("Starting training...");
        for step in 0..args.num_steps {
            let start = std::time::Instant::now();

            let (inputs, targets) = train_loader.next_batch()?;
            model.forward(
                inputs.as_slice().unwrap(),
                Some(targets.as_slice().unwrap()),
            )?;
            let loss = model.mean_loss;

            model.backward()?;
            let grad_norm = model.clip_gradients(args.grad_clip)?;
            model.step(
                args.learning_rate,
                0.9,
                0.999,
                1e-8,
                args.weight_decay,
                step as i32 + 1,
            )?;

            if (step + 1) % 10 == 0 || step == 0 {
                info!("[Step {}] Grad Norm: {:.6}", step + 1, grad_norm);
                model.check_diagnostics(step as i32 + 1);
            }

            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            info!(
                "step {}: train loss {:.6} (took {:.3} ms)",
                step, loss, elapsed
            );
        }

        // Final validation
        let (val_inputs, val_targets) = val_loader.next_batch()?;
        model.forward(
            val_inputs.as_slice().unwrap(),
            Some(val_targets.as_slice().unwrap()),
        )?;
        info!("val loss {:.6}", model.mean_loss);

        info!("Training complete!");
    }

    #[cfg(not(feature = "cuda"))]
    {
        info!("Using CPU backend (not fully implemented)");
        anyhow::bail!("CPU backend not implemented. Please compile with --features cuda");
    }

    Ok(())
}
