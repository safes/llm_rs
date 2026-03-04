# llm_rs

A Rust implementation of GPT-2 training, converted from [llm.c](https://github.com/karpathy/llm.c).

## Overview

This project is a complete rewrite of the llm.c GPT-2 training implementation in Rust, maintaining the same functionality while leveraging Rust's safety guarantees and modern tooling.

## Features

- ✅ **Pure Rust Implementation**: All C/C++ code converted to safe Rust
- ✅ **CPU Training**: Efficient CPU-based training with Rayon parallelism (replaces OpenMP)
- ✅ **Neural Network Layers**: Complete implementation of all GPT-2 layers
  - Token and position embeddings
  - Layer normalization
  - Multi-head attention with causal masking
  - GELU activation
  - Feedforward networks
  - Residual connections
- ✅ **AdamW Optimizer**: With weight decay and bias correction
- ✅ **Data Loading**: Memory-mapped file I/O for efficient data loading
- ✅ **GPT-2 Tokenizer**: BPE tokenizer compatible with GPT-2
- 🚧 **CUDA Support**: Optional GPU acceleration (work in progress)

## Quick Start

### Prerequisites

- Rust 1.70 or later
- (Optional) CUDA toolkit for GPU support

### Installation

```bash
cd c:\Developments\llm.c\llm_rs
cargo build --release
```

### Training

1. Download the starter pack from the original llm.c repo:

```bash
cd c:\Developments\llm.c
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
```

2. Run training:

```bash
cd llm_rs
cargo run --release --bin train_gpt2
```

### Custom Training

```bash
cargo run --release --bin train_gpt2 -- \
    --checkpoint ../gpt2_124M.bin \
    --train-data ../dev/data/tinyshakespeare/tiny_shakespeare_train.bin \
    --val-data ../dev/data/tinyshakespeare/tiny_shakespeare_val.bin \
    --batch-size 4 \
    --seq-len 64 \
    --num-steps 40 \
    --learning-rate 0.0003
```

## Project Structure

```
llm_rs/
├── src/
│   ├── lib.rs              # Library root
│   ├── main.rs             # Training binary
│   ├── model/              # GPT-2 model structure
│   ├── layers/             # Neural network layers
│   │   ├── encoder.rs      # Token + position embeddings
│   │   ├── layernorm.rs    # Layer normalization
│   │   ├── attention.rs    # Multi-head attention
│   │   ├── matmul.rs       # Matrix multiplication
│   │   ├── gelu.rs         # GELU activation
│   │   ├── softmax.rs      # Softmax and cross-entropy
│   │   └── residual.rs     # Residual connections
│   ├── optimizers/         # Optimizers
│   │   └── adamw.rs        # AdamW optimizer
│   ├── data/               # Data loading
│   │   ├── dataloader.rs   # Memory-mapped data loader
│   │   └── tokenizer.rs    # GPT-2 tokenizer
│   ├── utils.rs            # Utility functions
│   └── logger.rs           # Training logger
├── tests/                  # Integration tests
├── benches/                # Performance benchmarks
└── Cargo.toml              # Package manifest
```

## Differences from llm.c

### Advantages

1. **Memory Safety**: Rust's ownership system prevents memory leaks and buffer overflows
2. **Type Safety**: Strong type system catches errors at compile time
3. **Modern Tooling**: Cargo for dependency management and building
4. **Parallelism**: Rayon provides safe, easy parallelism (replaces OpenMP)
5. **Error Handling**: Result types for explicit error handling

### Trade-offs

1. **Binary Compatibility**: Uses different serialization format (may require conversion)
2. **CUDA Integration**: More complex than C (requires FFI or Rust CUDA bindings)
3. **Learning Curve**: Requires Rust knowledge

## Performance

Performance is comparable to the C version on CPU:
- Uses Rayon for parallel execution (similar to OpenMP)
- Optional BLAS integration for matrix operations
- Memory-mapped I/O for efficient data loading

## Testing

Run unit tests:
```bash
cargo test
```

Run integration tests:
```bash
cargo test --test integration_test
```

Run benchmarks:
```bash
cargo bench
```

## Development

### Building with CUDA Support

```bash
cargo build --release --features cuda
# Run with CUDA enabled
cargo run --release --features cuda --bin train_gpt2
```

### Building with CPU Optimizations

```bash
cargo build --release --features cpu-optimized
```

## Contributing

This is a conversion project. For the original llm.c, see:
https://github.com/karpathy/llm.c

## License

MIT (same as llm.c)

## Acknowledgments

- Original llm.c by Andrej Karpathy
- Rust community for excellent libraries (ndarray, rayon, etc.)
