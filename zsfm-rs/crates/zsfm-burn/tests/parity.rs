//! Burn operator correctness + timing, offline (fixed synthetic inputs).
//!
//! Run with `-- --nocapture` to print the timing table. Every case checks
//! the Burn operator against an independent pure-Rust host reference and
//! records median ms over 50 iterations. (The original candle-vs-Burn
//! parity gates passed and were retired with the candle removal.)

use burn::tensor::{Device, Tensor, TensorData};
use zsfm_burn::bench::{bench_op, max_abs_err};

fn dev() -> Device {
    Device::flex()
}

fn pseudo_data(n: usize, seed: u64) -> Vec<f32> {
    // Deterministic pseudo-random in [-1, 1).
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Host reference: y = x @ w^T + b, w row-major [d_out, d_in].
fn host_linear(x: &[f32], lead: usize, w: &[f32], d_out: usize, b: &[f32]) -> Vec<f32> {
    let d_in = b.len();
    assert_eq!(w.len(), d_out * d_in);
    let mut y = vec![0.0f32; lead * d_out];
    for i in 0..lead {
        for o in 0..d_out {
            let mut acc = b[o];
            for k in 0..d_in {
                acc += w[o * d_in + k] * x[i * d_in + k];
            }
            y[i * d_out + o] = acc;
        }
    }
    y
}

/// Host reference: y = x / sqrt(mean(x^2) + eps) * w, per row.
fn host_rms_norm(x: &[f32], n: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    let mut y = vec![0.0f32; x.len()];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let s = 1.0 / (ms + eps).sqrt();
        for k in 0..d {
            y[i * d + k] = row[k] * s * w[k];
        }
    }
    y
}

/// Host reference: softmax(Q K^T * scale) V, single head, no mask.
fn host_attention(q: &[f32], k: &[f32], v: &[f32], sq: usize, sk: usize, hd: usize) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; sq * hd];
    for i in 0..sq {
        let mut scores = vec![0.0f32; sk];
        for j in 0..sk {
            let mut acc = 0.0f32;
            for d in 0..hd {
                acc += q[i * hd + d] * k[j * hd + d];
            }
            scores[j] = acc * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        for d in 0..hd {
            let mut acc = 0.0f32;
            for j in 0..sk {
                acc += scores[j] / sum * v[j * hd + d];
            }
            out[i * hd + d] = acc;
        }
    }
    out
}

#[test]
fn linear_correctness_and_timing() {
    let (lead, d_in, d_out) = (32, 64, 64);
    let xv = pseudo_data(lead * d_in, 1);
    let wv = pseudo_data(d_out * d_in, 2);
    let bv = pseudo_data(d_out, 3);

    let expected = host_linear(&xv, lead, &wv, d_out, &bv);

    let device = dev();
    let bx = Tensor::<2>::from_data(TensorData::new(xv.clone(), [lead, d_in]), &device);
    let bw = Tensor::<2>::from_data(TensorData::new(wv.clone(), [d_out, d_in]), &device);
    let bb = Tensor::<1>::from_data(TensorData::new(bv.clone(), [d_out]), &device);
    let got: Vec<f32> = zsfm_burn::linear::burn_linear_bias(bx.clone(), bw.clone(), bb.clone())
        .to_data()
        .try_to_vec()
        .unwrap();

    let err = max_abs_err(&expected, &got);
    let burn_ms = bench_op(50, || {
        let _ = zsfm_burn::linear::burn_linear_bias(bx.clone(), bw.clone(), bb.clone());
    });
    println!("linear [{lead}x{d_in}->{d_out}]: burn {burn_ms:.3}ms err {err:.2e}");
    assert!(err < 1e-5, "linear mismatch: {err}");
}

#[test]
fn rms_norm_correctness_and_timing() {
    let (n, d) = (32, 64);
    let xv = pseudo_data(n * d, 11);
    let wv = pseudo_data(d, 12);

    let expected = host_rms_norm(&xv, n, &wv, 1e-6);

    let device = dev();
    let bx = Tensor::<2>::from_data(TensorData::new(xv.clone(), [n, d]), &device);
    let bw = Tensor::<1>::from_data(TensorData::new(wv.clone(), [d]), &device);
    let got: Vec<f32> = zsfm_burn::norm::burn_rms_norm(bx.clone(), bw.clone(), 1e-6)
        .to_data()
        .try_to_vec()
        .unwrap();

    let err = max_abs_err(&expected, &got);
    let burn_ms = bench_op(50, || {
        let _ = zsfm_burn::norm::burn_rms_norm(bx.clone(), bw.clone(), 1e-6);
    });
    println!("rms_norm [{n}x{d}]: burn {burn_ms:.3}ms err {err:.2e}");
    assert!(err < 1e-4, "rms_norm mismatch: {err}");
}

#[test]
fn attention_correctness_and_timing() {
    let (b, h, sq, sk, hd) = (2, 4, 16, 16, 32);
    let qv = pseudo_data(b * h * sq * hd, 21);
    let kv = pseudo_data(b * h * sk * hd, 22);
    let vv = pseudo_data(b * h * sk * hd, 23);

    // Host reference per (batch, head).
    let mut expected = Vec::with_capacity(qv.len());
    for bh in 0..b * h {
        expected.extend_from_slice(&host_attention(
            &qv[bh * sq * hd..(bh + 1) * sq * hd],
            &kv[bh * sk * hd..(bh + 1) * sk * hd],
            &vv[bh * sk * hd..(bh + 1) * sk * hd],
            sq,
            sk,
            hd,
        ));
    }

    let device = dev();
    let bq = Tensor::<4>::from_data(TensorData::new(qv.clone(), [b, h, sq, hd]), &device);
    let bk = Tensor::<4>::from_data(TensorData::new(kv.clone(), [b, h, sk, hd]), &device);
    let bv = Tensor::<4>::from_data(TensorData::new(vv.clone(), [b, h, sk, hd]), &device);
    let got: Vec<f32> = zsfm_burn::attn::burn_attention(
        bq.clone(),
        bk.clone(),
        bv.clone(),
        1.0 / (hd as f32).sqrt(),
        None,
    )
    .to_data()
    .try_to_vec()
    .unwrap();

    let err = max_abs_err(&expected, &got);
    let burn_ms = bench_op(20, || {
        let _ = zsfm_burn::attn::burn_attention(
            bq.clone(),
            bk.clone(),
            bv.clone(),
            1.0 / (hd as f32).sqrt(),
            None,
        );
    });
    println!("attn [b{b} h{h} sq{sq} hd{hd}]: burn {burn_ms:.3}ms err {err:.2e}");
    assert!(err < 1e-4, "attention mismatch: {err}");
}
