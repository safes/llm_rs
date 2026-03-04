//! # llm_rs
//!
//! A Rust implementation of GPT-2 training, converted from llm.c.
//! This library provides efficient neural network layers, optimizers,
//! and utilities for training large language models.

pub mod data;
pub mod layers;
pub mod model;
pub mod optimizers;
pub mod utils;
pub mod logger;

#[cfg(feature = "cuda")]
pub mod cuda;

// Re-export commonly used types
pub use model::{GPT2, GPT2Config};
#[cfg(feature = "cuda")]
pub use model::GPT2Cuda;
pub use data::{DataLoader, Tokenizer};
pub use optimizers::AdamW;

/// Type alias for the default floating point type
pub type Float = f32;

/// Result type for library operations
pub type Result<T> = anyhow::Result<T>;
