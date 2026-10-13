//! Moirai-1.0 inference on the Burn backend (Flex CPU).
//!
//! Mean-scale norm, patch embedding, masked bidirectional encoder with
//! QK-norm attention, split-half RoPE, variate bias, SwiGLU FFN,
//! Student-t loc head, with weights loaded straight from GGUF via
//! [`zsfm_burn::GgufWeights`].

use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use zsfm_burn::{open_gguf_weights, GgufWeights};

use zsfm_burn::linear::{burn_linear_bias, burn_linear_nobias};
use zsfm_burn::norm::burn_rms_norm_nd;

use crate::config::MoiraiConfig;

const PATCH_SIZE: usize = 32;
const PATCH_IDX: usize = 2;

fn device() -> Device {
    Device::flex()
}

fn load_t(weights: &GgufWeights, name: &str) -> Result<Tensor<2>> {
    weights.load_2d(name)
}

fn load_v(weights: &GgufWeights, name: &str) -> Result<Tensor<1>> {
    weights.load_1d(name)
}

fn load_3d(weights: &GgufWeights, name: &str) -> Result<Tensor<3>> {
    weights.load_3d(name)
}

fn load_flat(weights: &GgufWeights, name: &str) -> Result<Vec<f32>> {
    weights.load_flat(name)
}

fn linear(x: Tensor<2>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<2> {
    match b {
        Some(bias) => burn_linear_bias(x, w, bias),
        None => burn_linear_nobias(x, w),
    }
}

fn linear_nd<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    zsfm_burn::linear::burn_linear_nd(x, w, b)
}

fn rms<const D: usize>(x: Tensor<D>, w: Tensor<1>) -> Tensor<D> {
    burn_rms_norm_nd(x, w, 1e-6)
}

fn swiglu_ffn(x: Tensor<2>, fc1_w: Tensor<2>, fc2_w: Tensor<2>, gate_w: Tensor<2>) -> Tensor<2> {
    let content = activation::silu(linear(x.clone(), fc1_w, None));
    let gate = linear(x, gate_w, None);
    linear(content * gate, fc2_w, None)
}

struct BurnEncoderBlock {
    norm1_w: Tensor<1>,
    norm2_w: Tensor<1>,
    attn_qkv_w: Tensor<2>,
    attn_o_w: Tensor<2>,
    attn_qn_w: Tensor<1>,
    attn_kn_w: Tensor<1>,
    vbias_obs: Tensor<3>,
    vbias_mask: Tensor<3>,
    ffn_fc1_w: Tensor<2>,
    ffn_fc2_w: Tensor<2>,
    ffn_gate_w: Tensor<2>,
}

pub struct BurnMoiraiModel {
    config: MoiraiConfig,
    in_proj_w: Tensor<3>,
    in_proj_b: Tensor<2>,
    mask_embed: Tensor<2>,
    blocks: Vec<BurnEncoderBlock>,
    norm_f_w: Tensor<1>,
    head_st_loc_w: Tensor<3>,
    head_st_loc_b: Tensor<2>,
    rope_inv_freq: Vec<f32>,
}

impl BurnMoiraiModel {
    pub fn load(gguf_path: &Path, config: MoiraiConfig) -> Result<Self> {
        let weights = open_gguf_weights(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let dev = device();

        let in_proj_w = load_3d(&weights, "in_proj.weight")?;
        let in_proj_b = {
            let d: Vec<f32> = load_flat(&weights, "in_proj.bias")?;
            anyhow::ensure!(d.len().is_multiple_of(5), "in_proj.bias len");
            let d_model = d.len() / 5;
            Tensor::<2>::from_data(TensorData::new(d, [5, d_model]), &dev)
        };
        let mask_embed = load_t(&weights, "mask_embed.weight")?;

        let n_heads = config.n_heads;
        let mut blocks = Vec::with_capacity(config.n_layers);
        for n in 0..config.n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            let qw = load_t(&weights, &p("attn_q.weight"))?;
            let kw = load_t(&weights, &p("attn_k.weight"))?;
            let vw = load_t(&weights, &p("attn_v.weight"))?;
            let attn_qkv_w = Tensor::cat(vec![qw, kw, vw], 0);
            let vbias_raw: Vec<f32> = load_flat(&weights, &p("attn_vbias.weight"))?;
            let vbias_obs = Tensor::<3>::from_data(
                TensorData::new(vbias_raw[0..n_heads].to_vec(), [n_heads, 1, 1]),
                &dev,
            );
            let vbias_mask = Tensor::<3>::from_data(
                TensorData::new(vbias_raw[n_heads..2 * n_heads].to_vec(), [n_heads, 1, 1]),
                &dev,
            );
            blocks.push(BurnEncoderBlock {
                norm1_w: load_v(&weights, &p("norm1.weight"))?,
                norm2_w: load_v(&weights, &p("norm2.weight"))?,
                attn_qkv_w,
                attn_o_w: load_t(&weights, &p("attn_o.weight"))?,
                attn_qn_w: load_v(&weights, &p("attn_qn.weight"))?,
                attn_kn_w: load_v(&weights, &p("attn_kn.weight"))?,
                vbias_obs,
                vbias_mask,
                ffn_fc1_w: load_t(&weights, &p("ffn_fc1.weight"))?,
                ffn_fc2_w: load_t(&weights, &p("ffn_fc2.weight"))?,
                ffn_gate_w: load_t(&weights, &p("ffn_gate.weight"))?,
            });
        }

        let norm_f_w = load_v(&weights, "norm_f.weight")?;
        let head_st_loc_w = load_3d(&weights, "head.st_loc.weight")?;
        let head_st_loc_b = {
            let d: Vec<f32> = load_flat(&weights, "head.st_loc.bias")?;
            anyhow::ensure!(d.len().is_multiple_of(5), "head bias len");
            let max_ps = d.len() / 5;
            Tensor::<2>::from_data(TensorData::new(d, [5, max_ps]), &dev)
        };

        let half = config.head_dim / 2;
        let rope_inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0_f32 / 10000_f32.powf(2.0 * i as f32 / config.head_dim as f32))
            .collect();

        Ok(Self {
            config,
            in_proj_w,
            in_proj_b,
            mask_embed,
            blocks,
            norm_f_w,
            head_st_loc_w,
            head_st_loc_b,
            rope_inv_freq,
        })
    }

    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let patch_size = PATCH_SIZE;
        let patch_idx = PATCH_IDX;
        let loc = context.iter().map(|&v| v as f64).sum::<f64>() / context.len() as f64;
        let scale =
            context.iter().map(|&v| (v as f64 - loc).abs()).sum::<f64>() / context.len() as f64;
        let scale = (scale.max(1e-8)) as f32;
        let loc = loc as f32;
        let max_ts = cfg.max_seq_len;
        let mut ctx_scaled: Vec<f32> = context.iter().map(|&v| (v - loc) / scale).collect();
        if ctx_scaled.len() > max_ts {
            let start = ctx_scaled.len() - max_ts;
            ctx_scaled = ctx_scaled[start..].to_vec();
        }
        let ctx_len = ctx_scaled.len();
        let ctx_padded_len = ctx_len.div_ceil(patch_size) * patch_size;
        if ctx_padded_len > ctx_len {
            let mut padded = vec![0.0f32; ctx_padded_len - ctx_len];
            padded.extend_from_slice(&ctx_scaled);
            ctx_scaled = padded;
        }
        let n_ctx_patches = ctx_scaled.len() / patch_size;
        let n_fc_patches = horizon.div_ceil(patch_size);
        let total_patches = n_ctx_patches + n_fc_patches;
        let d_model = cfg.d_model;
        let max_ps = cfg.max_patch_size;
        let dev = device();

        let proj_w_full: Tensor<2> = self
            .in_proj_w
            .clone()
            .narrow(0, patch_idx, 1)
            .squeeze_dim::<2>(0);
        let proj_w: Tensor<2> = proj_w_full.narrow(1, 0, patch_size);
        let proj_b: Tensor<1> = self
            .in_proj_b
            .clone()
            .narrow(0, patch_idx, 1)
            .squeeze_dim::<1>(0);
        let mut patch_flat = vec![0.0f32; n_ctx_patches * patch_size];
        for i in 0..n_ctx_patches {
            let src = &ctx_scaled[i * patch_size..(i + 1) * patch_size];
            patch_flat[i * patch_size..(i + 1) * patch_size].copy_from_slice(src);
        }
        let ctx_patches_t = Tensor::<2>::from_data(
            TensorData::new(patch_flat, [n_ctx_patches, patch_size]),
            &dev,
        );
        let ctx_emb = linear(ctx_patches_t, proj_w, Some(proj_b));
        let mask_embed: Tensor<2> = self.mask_embed.clone().reshape([1, d_model]);
        let fc_emb = mask_embed.expand([n_fc_patches, d_model]);
        let mut h = Tensor::cat(vec![ctx_emb, fc_emb], 0);

        let mut is_masked = vec![0u8; total_patches];
        is_masked[n_ctx_patches..total_patches].fill(1);
        for blk in &self.blocks {
            h = self.forward_block(h, blk, &is_masked, total_patches)?;
        }
        h = rms(h, self.norm_f_w.clone());

        let loc_w_3d: Tensor<2> = self
            .head_st_loc_w
            .clone()
            .narrow(0, patch_idx, 1)
            .squeeze_dim::<2>(0);
        let loc_w: Tensor<2> = loc_w_3d.narrow(1, 0, patch_size);
        let loc_b: Tensor<1> = self
            .head_st_loc_b
            .clone()
            .narrow(0, patch_idx, 1)
            .squeeze_dim::<1>(0)
            .narrow(0, 0, patch_size);
        let future_h: Tensor<2> = h.narrow(0, n_ctx_patches, n_fc_patches);
        let pred = linear(future_h, loc_w, Some(loc_b));
        let pred_flat: Vec<f32> = pred
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let _ = max_ps;
        Ok(pred_flat
            .iter()
            .take(horizon)
            .map(|&v| v * scale + loc)
            .collect())
    }

    fn forward_block(
        &self,
        hidden: Tensor<2>,
        blk: &BurnEncoderBlock,
        is_masked: &[u8],
        seq_len: usize,
    ) -> Result<Tensor<2>> {
        let h = rms(hidden.clone(), blk.norm1_w.clone());
        let h = self.qk_attn(h, blk, is_masked, seq_len)?;
        let h = h + hidden;
        let h2 = rms(h.clone(), blk.norm2_w.clone());
        let h2 = swiglu_ffn(
            h2,
            blk.ffn_fc1_w.clone(),
            blk.ffn_fc2_w.clone(),
            blk.ffn_gate_w.clone(),
        );
        Ok(h2 + h)
    }

    fn qk_attn(
        &self,
        hidden: Tensor<2>,
        blk: &BurnEncoderBlock,
        is_masked: &[u8],
        seq_len: usize,
    ) -> Result<Tensor<2>> {
        let cfg = &self.config;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let d_model = cfg.d_model;
        let qkv = linear_nd(hidden, blk.attn_qkv_w.clone(), None);
        let q: Tensor<2> = qkv.clone().narrow(1, 0, d_model);
        let k: Tensor<2> = qkv.clone().narrow(1, d_model, d_model);
        let v: Tensor<2> = qkv.narrow(1, 2 * d_model, d_model);
        let q: Tensor<3> = q.reshape([seq_len, n_heads, head_dim]);
        let k: Tensor<3> = k.reshape([seq_len, n_heads, head_dim]);
        let q = qk_norm_heads(q, blk.attn_qn_w.clone(), seq_len, n_heads, head_dim);
        let k = qk_norm_heads(k, blk.attn_kn_w.clone(), seq_len, n_heads, head_dim);
        let q: Tensor<3> = q.permute([1, 0, 2]);
        let k: Tensor<3> = k.permute([1, 0, 2]);
        let (cos_t, sin_t) = self.rope_tables(seq_len);
        let q = apply_rope_with_tables(q, cos_t.clone(), sin_t.clone(), head_dim);
        let k = apply_rope_with_tables(k, cos_t, sin_t, head_dim);
        let v: Tensor<3> = v.reshape([seq_len, n_heads, head_dim]).permute([1, 0, 2]);
        let scores = q
            .matmul(k.permute([0, 2, 1]))
            .mul_scalar(1.0 / (head_dim as f32).sqrt());
        let scores = apply_var_attn_bias(
            scores,
            is_masked,
            seq_len,
            blk.vbias_obs.clone(),
            blk.vbias_mask.clone(),
        );
        let attn = activation::softmax(scores, 2);
        let out = attn.matmul(v);
        let out: Tensor<2> = out.permute([1, 0, 2]).reshape([seq_len, d_model]);
        Ok(linear_nd(out, blk.attn_o_w.clone(), None))
    }

    fn rope_tables(&self, seq_len: usize) -> (Tensor<3>, Tensor<3>) {
        let dev = device();
        let half = self.rope_inv_freq.len();
        let mut cos_v = vec![0.0f32; seq_len * half];
        let mut sin_v = vec![0.0f32; seq_len * half];
        for pos in 0..seq_len {
            for i in 0..half {
                let theta = pos as f32 * self.rope_inv_freq[i];
                cos_v[pos * half + i] = theta.cos();
                sin_v[pos * half + i] = theta.sin();
            }
        }
        let cos_t = Tensor::<2>::from_data(TensorData::new(cos_v, [seq_len, half]), &dev)
            .unsqueeze_dim::<3>(0);
        let sin_t = Tensor::<2>::from_data(TensorData::new(sin_v, [seq_len, half]), &dev)
            .unsqueeze_dim::<3>(0);
        (cos_t, sin_t)
    }
}

fn qk_norm_heads(
    x: Tensor<3>,
    weight: Tensor<1>,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Tensor<3> {
    let x_flat: Tensor<2> = x.reshape([seq_len * n_heads, head_dim]);
    burn_rms_norm_nd(x_flat, weight, 1e-6).reshape([seq_len, n_heads, head_dim])
}

fn apply_var_attn_bias(
    scores: Tensor<3>,
    is_masked: &[u8],
    seq_len: usize,
    vbias_obs: Tensor<3>,
    vbias_mask: Tensor<3>,
) -> Tensor<3> {
    let dev = device();
    let mask_f: Vec<f32> = is_masked.iter().map(|&m| m as f32).collect();
    let mask_t = Tensor::<3>::from_data(TensorData::new(mask_f, [1, 1, seq_len]), &dev);
    let delta = vbias_mask - vbias_obs.clone();
    let bias = vbias_obs + delta * mask_t;
    scores + bias
}

fn apply_rope_with_tables(
    x: Tensor<3>,
    cos_t: Tensor<3>,
    sin_t: Tensor<3>,
    head_dim: usize,
) -> Tensor<3> {
    let half = head_dim / 2;
    let x1 = x.clone().narrow(2, 0, half);
    let x2 = x.clone().narrow(2, half, half);
    let rot1 = x1.clone() * cos_t.clone() - x2.clone() * sin_t.clone();
    let rot2 = x1 * sin_t + x2 * cos_t;
    Tensor::cat(vec![rot1, rot2], 2)
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

    fn tiny_config() -> MoiraiConfig {
        MoiraiConfig {
            d_model: 32,
            n_layers: 1,
            n_heads: 4,
            head_dim: 8,
            d_ff: 64,
            max_seq_len: 64,
            patch_sizes: vec![8, 16, 32, 64, 128],
            max_patch_size: 32,
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

    fn put3(
        w: &mut GGUFWriter,
        name: &str,
        d0: usize,
        d1: usize,
        d2: usize,
        seed: &mut u64,
        scale: f32,
    ) {
        *seed += 1;
        let data: Vec<f32> = pseudo(d0 * d1 * d2, *seed)
            .iter()
            .map(|v| v * scale)
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor(
            name,
            vec![d2 as u64, d1 as u64, d0 as u64],
            GGMLType::F32,
            bytes,
        );
    }

    #[test]
    fn burn_forecast_golden() {
        // d=32, heads=4, hd=8, 1 layer, patch 32.
        let dir = std::env::temp_dir().join(format!("moirai-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("moirai".into()),
        );
        let mut seed = 1300u64;
        let s = 0.1;
        put3(&mut w, "in_proj.weight", 5, 32, 32, &mut seed, s);
        put2(&mut w, "in_proj.bias", 5, 32, &mut seed, 1.0);
        put2(&mut w, "mask_embed.weight", 1, 32, &mut seed, 1.0);
        put2(&mut w, "blk.0.attn_q.weight", 32, 32, &mut seed, s);
        put2(&mut w, "blk.0.attn_k.weight", 32, 32, &mut seed, s);
        put2(&mut w, "blk.0.attn_v.weight", 32, 32, &mut seed, s);
        put1(&mut w, "blk.0.norm1.weight", 32, &mut seed);
        put1(&mut w, "blk.0.norm2.weight", 32, &mut seed);
        put2(&mut w, "blk.0.attn_o.weight", 32, 32, &mut seed, s);
        put1(&mut w, "blk.0.attn_qn.weight", 8, &mut seed);
        put1(&mut w, "blk.0.attn_kn.weight", 8, &mut seed);
        put1(&mut w, "blk.0.attn_vbias.weight", 8, &mut seed);
        put2(&mut w, "blk.0.ffn_fc1.weight", 64, 32, &mut seed, s);
        put2(&mut w, "blk.0.ffn_fc2.weight", 32, 64, &mut seed, s);
        put2(&mut w, "blk.0.ffn_gate.weight", 64, 32, &mut seed, s);
        put1(&mut w, "norm_f.weight", 32, &mut seed);
        put3(&mut w, "head.st_loc.weight", 5, 32, 32, &mut seed, s);
        put2(&mut w, "head.st_loc.bias", 5, 32, &mut seed, 1.0);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let burn = BurnMoiraiModel::load(&path, cfg).unwrap();
        let ctx: Vec<f32> = (0..32).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let a = burn.forecast(&ctx, 4).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let b = burn.forecast(&ctx, 4).unwrap();
        assert_eq!(a.len(), 4);
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(a.iter().any(|&v| v != 0.0));
        assert_eq!(a, b, "forecast must be deterministic");
        println!("moirai synthetic: burn {burn_ms:.1}ms out={a:?}");
        // Golden values recorded from the verified Burn port (ex-candle parity Xe-Y).
        let golden: [f32; 4] = [10.968519, 12.493711, 11.747968, 11.247049];
        assert_eq!(a.len(), golden.len());
        for (x, y) in a.iter().zip(golden.iter()) {
            assert!((x - y).abs() < 1e-3, "golden drift: {x} vs {y}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
