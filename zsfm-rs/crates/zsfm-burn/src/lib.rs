//! Burn-backend inference.
//!
//! Shared Burn tensor ops (linear, norm, attention, RoPE), direct GGUF
//! weight loading via [`GgufWeights`], and op microbenchmarks. No model
//! weights needed for the op checks: they use fixed synthetic inputs so
//! they run offline in CI.
//!
//! Every measurement lands in `benchmark/burn_migration.md`.

pub mod attn;
pub mod bench;
pub mod gguf_weights;
pub mod linear;
pub mod norm;
pub mod rope;

pub use bench::{bench_op, max_abs_err, OpMeasurement};
pub use gguf_weights::{open_gguf_weights, GgufWeights};
