//! TiRex inference on the Burn backend (Flex CPU) for the two tensor
//! stages, with all host kernels reused verbatim from `super`.
//!
//! TiRex is already ~90% engine-agnostic host code (sLSTM recurrence,
//! patching, scaling, SIMD kernels). Only `residual_block_forward` and
//! `ffn_forward` touch tensors, so this port re-implements just those two
//! on Burn and keeps every host helper in this module.
//! Weight loading reads GGUF straight via [`zsfm_burn::GgufWeights`].

use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use simdeez::math::{SimdMathF32Core, SimdMathF32Hyperbolic};
use simdeez::prelude::*;
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::{open_gguf_weights, GgufWeights};

use crate::config::TiRexConfig;

// ---------------------------------------------------------------------------
// SIMD kernels (cross-platform: SSE2 / AVX2 / AVX-512 / NEON / WASM / scalar)
// ---------------------------------------------------------------------------

simd_runtime_generate!(
    fn simd_sq_sum(row: &[f32]) -> f32 {
        let mut r = &row[..];
        let mut acc = S::Vf32::zeroes();
        while r.len() >= S::Vf32::WIDTH {
            let v = S::Vf32::load_from_slice(r);
            acc = v.mul_add(v, acc);
            r = &r[S::Vf32::WIDTH..];
        }
        let mut sum = acc.horizontal_add();
        for &x in r {
            sum += x * x;
        }
        sum
    }
);

simd_runtime_generate!(
    fn simd_scale_weight(row: &mut [f32], w: &[f32], scale: f32) {
        let sc = S::Vf32::set1(scale);
        let mut r = &mut row[..];
        let mut wv = &w[..];
        while r.len() >= S::Vf32::WIDTH {
            let v = S::Vf32::load_from_slice(r);
            let wi = S::Vf32::load_from_slice(wv);
            (v * sc * wi).copy_to_slice(r);
            r = &mut r[S::Vf32::WIDTH..];
            wv = &wv[S::Vf32::WIDTH..];
        }
        for i in 0..r.len() {
            r[i] *= scale * wv[i];
        }
    }
);

simd_runtime_generate!(
    fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut aa = &a[..];
        let mut bb = &b[..];
        let mut acc = S::Vf32::zeroes();
        while aa.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(aa);
            let vb = S::Vf32::load_from_slice(bb);
            acc = va.mul_add(vb, acc);
            aa = &aa[S::Vf32::WIDTH..];
            bb = &bb[S::Vf32::WIDTH..];
        }
        let mut sum = acc.horizontal_add();
        for (&x, &y) in aa.iter().zip(bb.iter()) {
            sum += x * y;
        }
        sum
    }
);

simd_runtime_generate!(
    fn simd_gate_update(
        iraw: &[f32],
        fraw: &[f32],
        zraw: &[f32],
        oraw: &[f32],
        c: &[f32],
        n: &[f32],
        m: &[f32],
        cnew: &mut [f32],
        nnew: &mut [f32],
        hnew: &mut [f32],
        mnew: &mut [f32],
        is_first: bool,
    ) {
        let clamp = S::Vf32::set1(15.0f32);
        let one = S::Vf32::set1(1.0f32);
        let eps = S::Vf32::set1(1e-8f32);
        let zero = S::Vf32::zeroes();

        let mut ir = &iraw[..];
        let mut fr = &fraw[..];
        let mut zr = &zraw[..];
        let mut or_ = &oraw[..];
        let mut cv = &c[..];
        let mut nv = &n[..];
        let mut mv = &m[..];
        let mut cnw = &mut cnew[..];
        let mut nnw = &mut nnew[..];
        let mut hnw = &mut hnew[..];
        let mut mnw = &mut mnew[..];

        while ir.len() >= S::Vf32::WIDTH {
            let iv = S::Vf32::load_from_slice(ir);
            let fv = S::Vf32::load_from_slice(fr).min(clamp);
            let zv = S::Vf32::load_from_slice(zr);
            let ov = S::Vf32::load_from_slice(or_);
            let cv_ = S::Vf32::load_from_slice(cv);
            let nv_ = S::Vf32::load_from_slice(nv);
            let mv_ = S::Vf32::load_from_slice(mv);

            // log_sigmoid(fv): stable two-branch form, select by sign
            let ls_pos = -(one + (-fv).exp_u35()).ln_u35(); // fv >= 0: -ln(1+exp(-fv))
            let ls_neg = fv - (one + fv.exp_u35()).ln_u35(); // fv < 0:  fv - ln(1+exp(fv))
            let log_sig_f = fv.cmp_lt(zero).blendv(ls_pos, ls_neg);

            let logfplusm = mv_ + log_sig_f;
            let mnew_v = if is_first { iv } else { iv.max(logfplusm) };

            let ogate = one / (one + (-ov).exp_u35()); // sigmoid(ov)
            let igate = (iv - mnew_v).exp_u35().min(one);
            let fgate = (logfplusm - mnew_v).exp_u35().min(one);
            let zgate = zv.tanh_u35();

            let cnew_v = fgate.mul_add(cv_, igate * zgate);
            let nnew_v = fgate.mul_add(nv_, igate);

            let mask = nnew_v.abs().cmp_gt(eps);
            let hnew_v = mask.blendv(zero, ogate * cnew_v / nnew_v);

            cnew_v.copy_to_slice(cnw);
            nnew_v.copy_to_slice(nnw);
            hnew_v.copy_to_slice(hnw);
            mnew_v.copy_to_slice(mnw);

            ir = &ir[S::Vf32::WIDTH..];
            fr = &fr[S::Vf32::WIDTH..];
            zr = &zr[S::Vf32::WIDTH..];
            or_ = &or_[S::Vf32::WIDTH..];
            cv = &cv[S::Vf32::WIDTH..];
            nv = &nv[S::Vf32::WIDTH..];
            mv = &mv[S::Vf32::WIDTH..];
            cnw = &mut cnw[S::Vf32::WIDTH..];
            nnw = &mut nnw[S::Vf32::WIDTH..];
            hnw = &mut hnw[S::Vf32::WIDTH..];
            mnw = &mut mnw[S::Vf32::WIDTH..];
        }

        for i in 0..ir.len() {
            let iv_s = ir[i];
            let fv_s = fr[i].min(15.0);
            let zv_s = zr[i];
            let ov_s = or_[i];
            let c_s = cv[i];
            let n_s = nv[i];
            let m_s = mv[i];
            let ls = if fv_s >= 0.0 {
                -(1.0 + (-fv_s).exp()).ln()
            } else {
                fv_s - (1.0 + fv_s.exp()).ln()
            };
            let lfpm = m_s + ls;
            let mn = if is_first { iv_s } else { iv_s.max(lfpm) };
            let og = 1.0 / (1.0 + (-ov_s).exp());
            let ig = (iv_s - mn).exp().min(1.0);
            let fg = (lfpm - mn).exp().min(1.0);
            let zg = zv_s.tanh();
            cnw[i] = fg * c_s + ig * zg;
            nnw[i] = fg * n_s + ig;
            hnw[i] = if nnw[i].abs() > 1e-8 {
                og * cnw[i] / nnw[i]
            } else {
                0.0
            };
            mnw[i] = mn;
        }
    }
);

/// StandardScaler: compute (loc, scale) ignoring NaN values.
fn standard_scaler(x: &[f32]) -> (f32, f32) {
    let eps = 1e-5f32;
    let valid: Vec<f32> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    if valid.is_empty() {
        return (0.0, 1.0);
    }
    let loc = valid.iter().sum::<f32>() / valid.len() as f32;
    let variance = valid.iter().map(|&v| (v - loc) * (v - loc)).sum::<f32>() / valid.len() as f32;
    let scale = variance.sqrt();
    let scale = if scale == 0.0 { loc.abs() + eps } else { scale };
    (loc, scale)
}

/// Pad (left with NaN) or truncate (from front) context to target_len.
fn adjust_context(x: &[f32], target_len: usize) -> Vec<f32> {
    if x.len() >= target_len {
        x[x.len() - target_len..].to_vec()
    } else {
        let pad = target_len - x.len();
        let mut out = vec![f32::NAN; target_len];
        out[pad..].copy_from_slice(x);
        out
    }
}

/// RMSNorm in-place: x = x / rms(x) * w
fn rms_norm_inplace(x: &mut [f32], w: &[f32], d: usize, eps: f32) {
    let s = x.len() / d;
    for t in 0..s {
        let row = &mut x[t * d..(t + 1) * d];
        let ss = simd_sq_sum(row);
        let scale = 1.0 / (ss / d as f32 + eps).sqrt();
        simd_scale_weight(row, w, scale);
    }
}

/// Fused headwise linear for ng gate groups: [S, D] → [S, ng*D] written into `out`.
///
/// w layout: [NH, ng*DH, DH] — for head h, gate g: rows g*DH..(g+1)*DH hold w[h,g,:].
/// Output layout: out[t, g, h, o] at flat index t*ng*d + g*d + h*dh + o.
/// This matches the x_g layout expected by the sLSTM recurrence, so no interleaving
/// copy is needed after calling this function.
fn headwise_linear_batch_ng(
    x: &[f32],
    w: &[f32],
    s: usize,
    nh: usize,
    dh: usize,
    ng: usize,
    out: &mut [f32],
) {
    let d = nh * dh;
    let ngdh = ng * dh;
    for t in 0..s {
        for h in 0..nh {
            let x_h = &x[t * d + h * dh..t * d + (h + 1) * dh];
            let w_h_off = h * ngdh * dh;
            for g in 0..ng {
                let out_base = t * ng * d + g * d + h * dh;
                let w_gate_off = w_h_off + g * dh * dh;
                for o in 0..dh {
                    let w_row = &w[w_gate_off + o * dh..w_gate_off + (o + 1) * dh];
                    out[out_base + o] = simd_dot(x_h, w_row);
                }
            }
        }
    }
}

/// Compute recurrent contribution Ry from h_prev and the sLSTM kernel.
///
/// kernel_t layout: [NH, NG*DH, DH] (transposed at load time from [NH, DH, NG*DH])
/// so the inner dot-product dimension (di) is contiguous, enabling simd_dot.
/// Output written into `out`: [NG*NH*DH = 2048] in [NG, NH, DH] order.
fn compute_ry(
    h: &[f32],
    kernel_t: &[f32],
    nh: usize,
    dh: usize,
    ng: usize,
    ry_raw: &mut [f32],
    out: &mut [f32],
) {
    // ry_raw[NH, NG*DH]: for each head, dot h_head with each row of kernel_t[head]
    for head in 0..nh {
        let h_h = &h[head * dh..(head + 1) * dh];
        let k_head = &kernel_t[head * ng * dh * dh..];
        for gate_d in 0..ng * dh {
            let k_row = &k_head[gate_d * dh..(gate_d + 1) * dh]; // contiguous — simd_dot applies
            ry_raw[head * ng * dh + gate_d] = simd_dot(h_h, k_row);
        }
    }
    // Permute [NH, NG, DH] → [NG, NH, DH]
    for head in 0..nh {
        for g in 0..ng {
            for d in 0..dh {
                out[g * nh * dh + head * dh + d] = ry_raw[head * ng * dh + g * dh + d];
            }
        }
    }
}

fn device() -> Device {
    Device::flex()
}

fn load_t(weights: &GgufWeights, name: &str) -> Result<Tensor<2>> {
    weights.load_2d(name)
}

fn load_vec(weights: &GgufWeights, name: &str) -> Result<Vec<f32>> {
    weights.load_flat(name)
}

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

struct BurnEmbedBlock {
    hidden_w: Tensor<2>,
    hidden_b: Vec<f32>,
    output_w: Tensor<2>,
    output_b: Vec<f32>,
    residual_w: Tensor<2>,
    residual_b: Vec<f32>,
}

struct BurnBlock {
    norm_slstm: Vec<f32>,
    fizo_w: Vec<f32>,
    slstm_kernel_t: Vec<f32>,
    slstm_bias: Vec<f32>,
    group_norm_w: Vec<f32>,
    norm_ffn: Vec<f32>,
    ffn_gate_w: Tensor<2>,
    ffn_up_w: Tensor<2>,
    ffn_down_w: Tensor<2>,
}

pub struct BurnTiRexModel {
    config: TiRexConfig,
    in_emb: BurnEmbedBlock,
    blocks: Vec<BurnBlock>,
    out_norm: Vec<f32>,
    out_emb: BurnEmbedBlock,
}

fn load_embed_block(weights: &GgufWeights, prefix: &str) -> Result<BurnEmbedBlock> {
    let p = |s: &str| format!("{prefix}.{s}");
    Ok(BurnEmbedBlock {
        hidden_w: load_t(weights, &p("hidden.weight"))?,
        hidden_b: load_vec(weights, &p("hidden.bias"))?,
        output_w: load_t(weights, &p("output.weight"))?,
        output_b: load_vec(weights, &p("output.bias"))?,
        residual_w: load_t(weights, &p("residual.weight"))?,
        residual_b: load_vec(weights, &p("residual.bias"))?,
    })
}

impl BurnTiRexModel {
    pub fn load(gguf_path: &Path, config: TiRexConfig) -> Result<Self> {
        let weights = open_gguf_weights(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;

        let in_emb = load_embed_block(&weights, "in_emb")?;
        let nh = config.num_heads;
        let dh = config.head_dim();
        let wpp = dh * dh;
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for n in 0..config.num_blocks {
            let p = |s: &str| format!("blk.{n}.{s}");
            let fgate_w = load_vec(&weights, &p("fgate.weight"))?;
            let igate_w = load_vec(&weights, &p("igate.weight"))?;
            let zgate_w = load_vec(&weights, &p("zgate.weight"))?;
            let ogate_w = load_vec(&weights, &p("ogate.weight"))?;
            let mut fizo_w = vec![0.0f32; nh * 4 * wpp];
            for h in 0..nh {
                fizo_w[h * 4 * wpp..h * 4 * wpp + wpp]
                    .copy_from_slice(&fgate_w[h * wpp..(h + 1) * wpp]);
                fizo_w[h * 4 * wpp + wpp..h * 4 * wpp + 2 * wpp]
                    .copy_from_slice(&igate_w[h * wpp..(h + 1) * wpp]);
                fizo_w[h * 4 * wpp + 2 * wpp..h * 4 * wpp + 3 * wpp]
                    .copy_from_slice(&zgate_w[h * wpp..(h + 1) * wpp]);
                fizo_w[h * 4 * wpp + 3 * wpp..h * 4 * wpp + 4 * wpp]
                    .copy_from_slice(&ogate_w[h * wpp..(h + 1) * wpp]);
            }
            let raw_kernel = load_vec(&weights, &p("slstm_kernel"))?;
            let ng = 4usize;
            let mut slstm_kernel_t = vec![0.0f32; nh * ng * dh * dh];
            for head in 0..nh {
                for gate_d in 0..(ng * dh) {
                    for di in 0..dh {
                        slstm_kernel_t[head * ng * dh * dh + gate_d * dh + di] =
                            raw_kernel[head * dh * ng * dh + di * ng * dh + gate_d];
                    }
                }
            }
            blocks.push(BurnBlock {
                norm_slstm: load_vec(&weights, &p("norm_slstm"))?,
                fizo_w,
                slstm_kernel_t,
                slstm_bias: load_vec(&weights, &p("slstm_bias"))?,
                group_norm_w: load_vec(&weights, &p("group_norm"))?,
                norm_ffn: load_vec(&weights, &p("norm_ffn"))?,
                ffn_gate_w: load_t(&weights, &p("ffn_gate.weight"))?,
                ffn_up_w: load_t(&weights, &p("ffn_up.weight"))?,
                ffn_down_w: load_t(&weights, &p("ffn_down.weight"))?,
            });
        }
        let out_norm = load_vec(&weights, "out_norm")?;
        let out_emb = load_embed_block(&weights, "out_emb")?;
        Ok(Self {
            config,
            in_emb,
            blocks,
            out_norm,
            out_emb,
        })
    }
}

/// Burn residual MLP block over host row-major data.
fn residual_block_forward_burn(
    x: &[f32],
    s: usize,
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
    emb: &BurnEmbedBlock,
) -> Vec<f32> {
    let dev = device();
    let xt = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [s, in_dim]), &dev);
    let bh = Tensor::<1>::from_data(TensorData::new(emb.hidden_b.clone(), [h_dim]), &dev);
    let h = activation::relu(burn_linear_nd(xt.clone(), emb.hidden_w.clone(), Some(bh)));
    let bo = Tensor::<1>::from_data(TensorData::new(emb.output_b.clone(), [out_dim]), &dev);
    let out = burn_linear_nd(h, emb.output_w.clone(), Some(bo));
    let br = Tensor::<1>::from_data(TensorData::new(emb.residual_b.clone(), [out_dim]), &dev);
    let res = burn_linear_nd(xt, emb.residual_w.clone(), Some(br));
    let y: Vec<f32> = (out + res).to_data().try_to_vec().unwrap();
    y
}

/// Burn SiLU-gated FFN over host row-major data.
fn ffn_forward_burn(
    x: &[f32],
    s: usize,
    in_dim: usize,
    up_dim: usize,
    blk: &BurnBlock,
) -> Vec<f32> {
    let dev = device();
    let xt = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [s, in_dim]), &dev);
    let gate = burn_linear_nd(xt.clone(), blk.ffn_gate_w.clone(), None);
    let up = burn_linear_nd(xt, blk.ffn_up_w.clone(), None);
    let h = activation::silu(gate) * up;
    let out = burn_linear_nd(h, blk.ffn_down_w.clone(), None);
    out.to_data().try_to_vec().unwrap()
}

impl BurnTiRexModel {
    pub fn forecast(
        &self,
        context: &[f32],
        prediction_length: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let cfg = &self.config;
        let patch_size = cfg.patch_size;
        let n_quantiles = cfg.num_quantiles;
        let median_idx = cfg
            .quantiles
            .iter()
            .position(|&q| (q - 0.5).abs() < 1e-6)
            .unwrap_or(4);
        let n_steps = prediction_length.div_ceil(patch_size);
        let mut all_q: Vec<f32> = Vec::with_capacity(n_steps * patch_size * n_quantiles);
        let mut ctx: Vec<f32> = context.to_vec();
        for _ in 0..n_steps {
            let full_ctx = adjust_context(&ctx, cfg.train_ctx_len);
            let (loc, scale) = standard_scaler(&full_ctx);
            let s_vals: Vec<f32> = full_ctx
                .iter()
                .map(|&x| {
                    if x.is_nan() {
                        0.0f32
                    } else {
                        (x - loc) / scale
                    }
                })
                .collect();
            let s_mask: Vec<f32> = full_ctx
                .iter()
                .map(|&x| if x.is_nan() { 0.0f32 } else { 1.0f32 })
                .collect();
            let num_patches = cfg.train_ctx_len / patch_size;
            let mut patched_vals = vec![0.0f32; num_patches * patch_size];
            let mut patched_mask = vec![0.0f32; num_patches * patch_size];
            for p in 0..num_patches {
                let start = p * patch_size;
                patched_vals[p * patch_size..(p + 1) * patch_size]
                    .copy_from_slice(&s_vals[start..start + patch_size]);
                patched_mask[p * patch_size..(p + 1) * patch_size]
                    .copy_from_slice(&s_mask[start..start + patch_size]);
            }
            let in_dim = patch_size * 2;
            let mut x_in = vec![0.0f32; num_patches * in_dim];
            for p in 0..num_patches {
                x_in[p * in_dim..p * in_dim + patch_size]
                    .copy_from_slice(&patched_vals[p * patch_size..(p + 1) * patch_size]);
                x_in[p * in_dim + patch_size..p * in_dim + in_dim]
                    .copy_from_slice(&patched_mask[p * patch_size..(p + 1) * patch_size]);
            }
            let mut hidden = residual_block_forward_burn(
                &x_in,
                num_patches,
                in_dim,
                cfg.input_ff_dim,
                cfg.embedding_dim,
                &self.in_emb,
            );
            let ng = 4usize;
            let d = cfg.embedding_dim;
            let mut sc_xg = vec![0.0f32; num_patches * ng * d];
            let mut sc_hout = vec![0.0f32; num_patches * d];
            let mut sc_y = vec![0.0f32; num_patches * d];
            let mut sc_xn = vec![0.0f32; num_patches * d];
            let mut sc_raw = vec![0.0f32; ng * d];
            let mut sc_ry_raw = vec![0.0f32; ng * d];
            let mut sc_ry_out = vec![0.0f32; ng * d];
            let mut sc_hnew = vec![0.0f32; d];
            let mut sc_cnew = vec![0.0f32; d];
            let mut sc_nnew = vec![0.0f32; d];
            let mut sc_mnew = vec![0.0f32; d];
            for block in &self.blocks {
                hidden = self.forward_slstm_block(
                    &hidden,
                    num_patches,
                    block,
                    &mut sc_xg,
                    &mut sc_hout,
                    &mut sc_y,
                    &mut sc_xn,
                    &mut sc_raw,
                    &mut sc_ry_raw,
                    &mut sc_ry_out,
                    &mut sc_hnew,
                    &mut sc_cnew,
                    &mut sc_nnew,
                    &mut sc_mnew,
                )?;
            }
            rms_norm_inplace(&mut hidden, &self.out_norm, cfg.embedding_dim, 1e-6);
            let out_dim = cfg.output_dim();
            let preds = residual_block_forward_burn(
                &hidden,
                num_patches,
                cfg.embedding_dim,
                cfg.input_ff_dim,
                out_dim,
                &self.out_emb,
            );
            let last = &preds[(num_patches - 1) * out_dim..num_patches * out_dim];
            for q in 0..n_quantiles {
                for d in 0..patch_size {
                    let v = last[q * patch_size + d] * scale + loc;
                    all_q.push(v);
                }
            }
            ctx.extend(std::iter::repeat(f32::NAN).take(patch_size));
        }
        let total = n_steps * patch_size;
        let mut quantiles: Vec<Vec<f32>> = vec![Vec::with_capacity(prediction_length); n_quantiles];
        for t in 0..prediction_length.min(total) {
            let step = t / patch_size;
            let d = t % patch_size;
            for q in 0..n_quantiles {
                quantiles[q].push(all_q[step * n_quantiles * patch_size + q * patch_size + d]);
            }
        }
        let median: Vec<f32> = (0..prediction_length)
            .map(|t| quantiles[median_idx][t])
            .collect();
        Ok((quantiles, median))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_slstm_block(
        &self,
        x: &[f32],
        s: usize,
        blk: &BurnBlock,
        sc_xg: &mut [f32],
        sc_hout: &mut [f32],
        sc_y: &mut [f32],
        sc_xn: &mut [f32],
        sc_raw: &mut [f32],
        sc_ry_raw: &mut [f32],
        sc_ry_out: &mut [f32],
        sc_hnew: &mut Vec<f32>,
        sc_cnew: &mut Vec<f32>,
        sc_nnew: &mut Vec<f32>,
        sc_mnew: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let d = cfg.embedding_dim;
        let nh = cfg.num_heads;
        let dh = cfg.head_dim();
        let ng = 4usize;
        sc_xn[..s * d].copy_from_slice(&x[..s * d]);
        rms_norm_inplace(&mut sc_xn[..s * d], &blk.norm_slstm, d, 1e-6);
        headwise_linear_batch_ng(&sc_xn[..s * d], &blk.fizo_w, s, nh, dh, ng, sc_xg);
        let mut h = vec![0.0f32; d];
        let mut c = vec![0.0f32; d];
        let mut n = vec![0.0f32; d];
        let mut m = vec![f32::NEG_INFINITY; d];
        for t in 0..s {
            let wx = &sc_xg[t * ng * d..(t + 1) * ng * d];
            compute_ry(&h, &blk.slstm_kernel_t, nh, dh, ng, sc_ry_raw, sc_ry_out);
            for i in 0..ng * d {
                sc_raw[i] = wx[i] + sc_ry_out[i] + blk.slstm_bias[i];
            }
            let is_first = t == 0;
            simd_gate_update(
                &sc_raw[..d],
                &sc_raw[d..2 * d],
                &sc_raw[2 * d..3 * d],
                &sc_raw[3 * d..],
                &c,
                &n,
                &m,
                sc_cnew,
                sc_nnew,
                sc_hnew,
                sc_mnew,
                is_first,
            );
            std::mem::swap(&mut h, sc_hnew);
            std::mem::swap(&mut c, sc_cnew);
            std::mem::swap(&mut n, sc_nnew);
            std::mem::swap(&mut m, sc_mnew);
            sc_hout[t * d..(t + 1) * d].copy_from_slice(&h);
        }
        for t in 0..s {
            let h_t = &sc_hout[t * d..(t + 1) * d];
            for head in 0..nh {
                let start = head * dh;
                let h_slice = &h_t[start..start + dh];
                let mean = h_slice.iter().sum::<f32>() / dh as f32;
                let var = h_slice
                    .iter()
                    .map(|&v| (v - mean) * (v - mean))
                    .sum::<f32>()
                    / dh as f32;
                let inv_std = 1.0 / (var + 1e-5f32).sqrt();
                let w_slice = &blk.group_norm_w[start..start + dh];
                for d_i in 0..dh {
                    sc_y[t * d + start + d_i] =
                        (h_t[start + d_i] - mean) * inv_std * (1.0 + w_slice[d_i]);
                }
            }
        }
        let mut x_out = x.to_vec();
        for i in 0..s * d {
            x_out[i] += sc_y[i];
        }
        sc_xn[..s * d].copy_from_slice(&x_out[..s * d]);
        rms_norm_inplace(&mut sc_xn[..s * d], &blk.norm_ffn, d, 1e-6);
        let ffn_out = ffn_forward_burn(&sc_xn[..s * d], s, d, cfg.ffn_up_dim, blk);
        for i in 0..s * d {
            x_out[i] += ffn_out[i];
        }
        Ok(x_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zsfm_gguf::{GGMLType, GGUFMetaValue, GGUFWriter};

    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
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

    fn tiny_config() -> TiRexConfig {
        TiRexConfig {
            patch_size: 4,
            num_blocks: 1,
            embedding_dim: 16,
            num_heads: 2,
            input_ff_dim: 32,
            ffn_up_dim: 24,
            train_ctx_len: 16,
            quantiles: vec![0.25, 0.5, 0.75],
            num_quantiles: 3,
        }
    }

    fn put2(w: &mut GGUFWriter, name: &str, rows: usize, cols: usize, seed: &mut u64, scale: f32) {
        *seed += 1;
        let data: Vec<f32> = pseudo(rows * cols, *seed)
            .iter()
            .map(|v| v * scale)
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor(name, vec![cols as u64, rows as u64], GGMLType::F32, bytes);
    }

    fn put1(w: &mut GGUFWriter, name: &str, n: usize, seed: &mut u64) {
        *seed += 1;
        let bytes: Vec<u8> = pseudo(n, *seed)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.add_tensor(name, vec![n as u64], GGMLType::F32, bytes);
    }

    fn put_emb(
        w: &mut GGUFWriter,
        prefix: &str,
        h: usize,
        d_in: usize,
        o: usize,
        seed: &mut u64,
        s: f32,
    ) {
        put2(w, &format!("{prefix}.hidden.weight"), h, d_in, seed, s);
        put1(w, &format!("{prefix}.hidden.bias"), h, seed);
        put2(w, &format!("{prefix}.output.weight"), o, h, seed, s);
        put1(w, &format!("{prefix}.output.bias"), o, seed);
        put2(w, &format!("{prefix}.residual.weight"), o, d_in, seed, s);
        put1(w, &format!("{prefix}.residual.bias"), o, seed);
    }

    #[test]
    fn burn_forecast_golden() {
        let dir = std::env::temp_dir().join(format!("tirex-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("tirex".into()),
        );
        let mut seed = 2900u64;
        let s = 0.05;
        put_emb(&mut w, "in_emb", 32, 8, 16, &mut seed, s);
        for t in ["fgate", "igate", "zgate", "ogate"] {
            put2(&mut w, &format!("blk.0.{t}.weight"), 16, 16, &mut seed, s);
        }
        // slstm_kernel logical [NH, DH, NG*DH] = [2,8,32].
        put1(&mut w, "blk.0.slstm_kernel", 2 * 8 * 32, &mut seed);
        // NOTE: raw kernel values unscaled (structure, not numerics, matters here).
        put1(&mut w, "blk.0.slstm_bias", 64, &mut seed);
        put1(&mut w, "blk.0.norm_slstm", 16, &mut seed);
        put1(&mut w, "blk.0.group_norm", 16, &mut seed);
        put1(&mut w, "blk.0.norm_ffn", 16, &mut seed);
        put2(&mut w, "blk.0.ffn_gate.weight", 24, 16, &mut seed, s);
        put2(&mut w, "blk.0.ffn_up.weight", 24, 16, &mut seed, s);
        put2(&mut w, "blk.0.ffn_down.weight", 16, 24, &mut seed, s);
        put1(&mut w, "out_norm", 16, &mut seed);
        put_emb(&mut w, "out_emb", 32, 16, 12, &mut seed, s);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let burn = BurnTiRexModel::load(&path, cfg).unwrap();
        let ctx: Vec<f32> = (0..8).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let (aq, am) = burn.forecast(&ctx, 2).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (bq, bm) = burn.forecast(&ctx, 2).unwrap();
        assert_eq!(aq, bq, "quantiles must be deterministic");
        assert_eq!(am, bm, "median must be deterministic");
        assert!(aq.iter().all(|r| r.iter().all(|v| v.is_finite())));
        assert!(am.iter().all(|v| v.is_finite()));
        println!("tirex synthetic: burn {burn_ms:.1}ms q={aq:?} m={am:?}");
        // Golden recorded from the verified Burn port (ex-candle parity 0.00e0).
        let golden_q: Vec<Vec<f32>> = vec![
            vec![10.033624, 10.955177],
            vec![10.487212, 9.479774],
            vec![10.795641, 10.196359],
        ];
        let golden_m: Vec<f32> = vec![10.487212, 9.479774];
        assert_eq!(aq.len(), golden_q.len());
        assert_eq!(am.len(), golden_m.len());
        for (ra, re) in aq.iter().zip(golden_q.iter()) {
            for (x, y) in ra.iter().zip(re.iter()) {
                assert!((x - y).abs() < 1e-4, "golden drift: {x} vs {y}");
            }
        }
        for (x, y) in am.iter().zip(golden_m.iter()) {
            assert!((x - y).abs() < 1e-4, "golden drift: {x} vs {y}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
