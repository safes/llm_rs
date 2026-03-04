//! Neural network layer implementations

pub mod encoder;
pub mod layernorm;
pub mod attention;
pub mod matmul;
pub mod gelu;
pub mod softmax;
pub mod residual;

// Re-export layer functions
pub use encoder::{encoder_forward, encoder_backward};
pub use layernorm::{layernorm_forward, layernorm_backward};
pub use attention::{attention_forward, attention_backward};
pub use matmul::{matmul_forward, matmul_backward};
pub use gelu::{gelu_forward, gelu_backward};
pub use softmax::{softmax_forward, crossentropy_forward, crossentropy_softmax_backward};
pub use residual::{residual_forward, residual_backward};
