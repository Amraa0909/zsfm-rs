//! Toto-2 inference on the Burn backend (Flex CPU).
//!
//! Patched input scaler, transformer with GQA time/variate layers, xPos RoPE,
//! PerDimScale, unit-scaled linears, SwiGLU FFN, and quantile output head,
//! with weights loaded straight from GGUF via [`zsfm_burn::GgufWeights`].
//!
//! The `compute_f64` path is not ported: pass a config with
//! `compute_f64 = false` (the default). The Burn port bails otherwise.

use std::path::Path;
use zsfm_core::{json_bool, json_f64, json_usize};

use anyhow::{bail, Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::{open_gguf_weights, GgufWeights};

const SILU_SCALE: f32 = 1.766782948312328;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct InferConfig {
    d_model: usize,
    num_layers: usize,
    num_heads: usize,
    num_groups: usize,
    qk_dim: usize,
    v_dim: usize,
    patch_size: usize,
    norm_eps: f64,
    layer_group_size: usize,
    num_variate_layers_per_group: usize,
    variate_layer_first: bool,
    use_xpos: bool,
    residual_mult: f64,
    residual_attn_ratio: f64,
    /// Run forward pass in F64 to match PyTorch numerical precision.
    /// Weights are cast to F64 at load time (~2× memory). Default: false.
    compute_f64: bool,
}

impl Default for InferConfig {
    /// Matches the defaults of the upstream Datadog/Toto-2.0 `config.json`.
    fn default() -> Self {
        InferConfig {
            d_model: 512,
            num_layers: 6,
            num_heads: 8,
            num_groups: 8,
            qk_dim: 64,
            v_dim: 64,
            patch_size: 32,
            norm_eps: 5e-4,
            layer_group_size: 48,
            num_variate_layers_per_group: 1,
            variate_layer_first: false,
            use_xpos: true,
            residual_mult: 0.75,
            residual_attn_ratio: 5.136215466577748,
            compute_f64: false,
        }
    }
}

impl InferConfig {
    fn is_variate_layer(&self, idx: usize) -> bool {
        if self.variate_layer_first {
            idx % self.layer_group_size < self.num_variate_layers_per_group
        } else {
            idx % self.layer_group_size >= self.layer_group_size - self.num_variate_layers_per_group
        }
    }
    fn q_size(&self) -> usize {
        self.qk_dim * self.num_heads
    }
    fn k_size(&self) -> usize {
        self.qk_dim * self.num_groups
    }
    fn v_size(&self) -> usize {
        self.v_dim * self.num_groups
    }

    // -- builder-style setters — chain from `InferConfig::default()` ---------

    pub fn with_d_model(mut self, v: usize) -> Self {
        self.d_model = v;
        self
    }
    pub fn with_num_layers(mut self, v: usize) -> Self {
        self.num_layers = v;
        self
    }
    pub fn with_num_heads(mut self, v: usize) -> Self {
        self.num_heads = v;
        self
    }
    pub fn with_num_groups(mut self, v: usize) -> Self {
        self.num_groups = v;
        self
    }
    pub fn with_qk_dim(mut self, v: usize) -> Self {
        self.qk_dim = v;
        self
    }
    pub fn with_v_dim(mut self, v: usize) -> Self {
        self.v_dim = v;
        self
    }
    pub fn with_patch_size(mut self, v: usize) -> Self {
        self.patch_size = v;
        self
    }
    pub fn with_norm_eps(mut self, v: f64) -> Self {
        self.norm_eps = v;
        self
    }
    pub fn with_layer_group_size(mut self, v: usize) -> Self {
        self.layer_group_size = v;
        self
    }
    pub fn with_num_variate_layers_per_group(mut self, v: usize) -> Self {
        self.num_variate_layers_per_group = v;
        self
    }
    pub fn with_variate_layer_first(mut self, v: bool) -> Self {
        self.variate_layer_first = v;
        self
    }
    pub fn with_use_xpos(mut self, v: bool) -> Self {
        self.use_xpos = v;
        self
    }
    pub fn with_residual_mult(mut self, v: f64) -> Self {
        self.residual_mult = v;
        self
    }
    pub fn with_residual_attn_ratio(mut self, v: f64) -> Self {
        self.residual_attn_ratio = v;
        self
    }

    /// Run the forward pass in F64 (double precision) to match PyTorch numerical
    /// accuracy. Uses ~2× memory. Default: F32.
    pub fn with_compute_f64(mut self, v: bool) -> Self {
        self.compute_f64 = v;
        self
    }

    // -- read-only accessors needed outside this crate ------------------------

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }
    pub fn compute_f64(&self) -> bool {
        self.compute_f64
    }

    /// Overlay every recognised field from a parsed `config.json` onto
    /// `InferConfig::default()`, using the same field names as the upstream
    /// Datadog/Toto-2.0 config. Missing/unrecognised fields keep their default.
    /// Shared by the CLI and (later) the Python bindings so the field-by-field
    /// `.as_u64().unwrap_or(...)` boilerplate lives in exactly one place.
    pub fn from_json_value(v: &serde_json::Value) -> Self {
        Self::default()
            .with_d_model(512)
            .with_num_layers(6)
            .with_num_heads(8)
            .with_num_groups(8)
            .with_qk_dim(64)
            .with_v_dim(64)
            .with_patch_size(json_usize(v, "patch_size", 32))
            .with_norm_eps(json_f64(v, "norm_eps", 5e-4))
            .with_layer_group_size(json_usize(v, "layer_group_size", 48))
            .with_num_variate_layers_per_group(json_usize(v, "num_variate_layers_per_group", 1))
            .with_variate_layer_first(json_bool(v, "variate_layer_first", false))
            .with_use_xpos(json_bool(v, "use_xpos", true))
            .with_residual_mult(json_f64(v, "residual_mult", 0.75))
            .with_residual_attn_ratio(json_f64(v, "residual_attn_ratio", 5.136215466577748))
    }
}

fn device() -> Device {
    Device::flex()
}

fn load_t(weights: &GgufWeights, name: &str) -> Result<Tensor<2>> {
    weights.load_2d(name)
}

fn load_v(weights: &GgufWeights, name: &str) -> Result<Tensor<1>> {
    weights.load_1d(name)
}

fn try_load_v(weights: &GgufWeights, name: &str) -> Result<Option<Tensor<1>>> {
    match weights.load_1d(name) {
        Ok(t) => Ok(Some(t)),
        Err(_) => Ok(None),
    }
}

fn load_w(weights: &GgufWeights, name: &str, expected_d_out: usize) -> Result<Tensor<2>> {
    weights.load_weight_2d(name, 512)
}

/// Unit-scaled linear: `scale * (x @ w^T + b)` with `scale = 1/sqrt(d_in)`.
fn uu_linear<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    let d_in = w.dims()[1] as f32;
    burn_linear_nd(x, w, b).mul_scalar(1.0 / d_in.sqrt())
}

fn uu_linear_readout<const D: usize>(
    x: Tensor<D>,
    w: Tensor<2>,
    b: Option<Tensor<1>>,
) -> Tensor<D> {
    let d_in = w.dims()[1] as f32;
    burn_linear_nd(x, w, b).mul_scalar(1.0 / d_in)
}

fn uu_silu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::silu(x).mul_scalar(SILU_SCALE)
}

fn rms_norm_weightless<const D: usize>(x: Tensor<D>, eps: f64) -> Tensor<D> {
    let dims = x.dims();
    let last = dims[D - 1];
    let lead: usize = dims[..D - 1].iter().product();
    let flat: Tensor<2> = x.reshape([lead, last]);
    let mean_sq = flat.clone().square().mean_dim(1);
    let rms = (mean_sq.add_scalar(eps as f32)).sqrt();
    (flat / rms).reshape(dims)
}

fn residual_add<const D: usize>(h: Tensor<D>, skip: Tensor<D>, tau: f64) -> Tensor<D> {
    let denom = (1.0 + tau * tau).sqrt() as f32;
    h.mul_scalar(tau as f32 / denom) + skip.mul_scalar(1.0 / denom)
}

fn apply_per_dim_scale<const D: usize>(q: Tensor<D>, pds_w: Tensor<1>) -> Tensor<D> {
    // softplus(pds) / ln2, broadcast over the last dim.
    let sp = pds_w.exp() + 1.0;
    let r = sp.log() / std::f32::consts::LN_2;
    let dims = q.dims();
    let last = dims[D - 1];
    let lead: usize = dims[..D - 1].iter().product();
    (q.reshape([lead, last]) * r.unsqueeze_dim::<2>(0)).reshape(dims)
}

struct BurnXposRope {
    proj_width: usize,
    cos: Vec<Vec<f32>>,
    sin: Vec<Vec<f32>>,
    xpos_base_scale: Vec<f32>,
}

impl BurnXposRope {
    fn build(qk_dim: usize, max_len: usize) -> Self {
        let proj_width = qk_dim / 2;
        let half = proj_width / 2;
        let base = 10000_f32;
        let theta: Vec<f32> = (0..half)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / proj_width as f32))
            .collect();
        let mut cos = vec![vec![0.0f32; proj_width]; max_len];
        let mut sin = vec![vec![0.0f32; proj_width]; max_len];
        for m in 0..max_len {
            for i in 0..half {
                let angle = m as f32 * theta[i];
                cos[m][2 * i] = angle.cos();
                cos[m][2 * i + 1] = angle.cos();
                sin[m][2 * i] = angle.sin();
                sin[m][2 * i + 1] = angle.sin();
            }
        }
        let xpos_base_scale: Vec<f32> = (0..half)
            .map(|i| ((2 * i) as f32 + 0.4 * proj_width as f32) / (1.4 * proj_width as f32))
            .collect();
        Self {
            proj_width,
            cos,
            sin,
            xpos_base_scale,
        }
    }

    fn apply(&self, x: Tensor<4>, seq_ids: &[u32], xpos_exponent: f32) -> Tensor<4> {
        let dims = x.dims();
        let qk_dim = dims[3];
        let proj_width = self.proj_width;
        let half = proj_width / 2;
        let seq_len = seq_ids.len();
        let max_pos = seq_ids.iter().copied().max().unwrap_or(0) as f32;
        let center = ((max_pos as u32 + 1) / 2) as f32;
        let mut cos_data = vec![0.0f32; seq_len * proj_width];
        let mut sin_data = vec![0.0f32; seq_len * proj_width];
        for (si, &pos) in seq_ids.iter().enumerate() {
            let power = (pos as f32 - center) / 256.0;
            for i in 0..half {
                let xpos_s = self.xpos_base_scale[i].powf(power).powf(xpos_exponent);
                let c = self.cos[pos as usize][2 * i] * xpos_s;
                let s = self.sin[pos as usize][2 * i] * xpos_s;
                cos_data[si * proj_width + 2 * i] = c;
                cos_data[si * proj_width + 2 * i + 1] = c;
                sin_data[si * proj_width + 2 * i] = s;
                sin_data[si * proj_width + 2 * i + 1] = s;
            }
        }
        let dev = device();
        let pos_cos =
            Tensor::<4>::from_data(TensorData::new(cos_data, [1, 1, seq_len, proj_width]), &dev);
        let pos_sin =
            Tensor::<4>::from_data(TensorData::new(sin_data, [1, 1, seq_len, proj_width]), &dev);
        // Interleaved rotate_half: [a0,b0,a1,b1..] -> [-b0,a0,-b1,a1..]
        let x_rot = x.clone().narrow(3, 0, proj_width);
        let shp = x_rot.dims();
        let pairs = x_rot.clone().reshape([shp[0], shp[1], shp[2], half, 2]);
        let x1 = pairs.clone().narrow(4, 0, 1);
        let x2 = pairs.narrow(4, 1, 1);
        let rot_x = Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 4)
            .reshape([shp[0], shp[1], shp[2], proj_width]);
        let rotated = x_rot * pos_cos + rot_x * pos_sin;
        if qk_dim > proj_width {
            let x_pass = x.narrow(3, proj_width, qk_dim - proj_width);
            Tensor::cat(vec![rotated, x_pass], 3)
        } else {
            rotated
        }
    }
}

struct BurnResidualMlpW {
    l1_w: Tensor<2>,
    l1_b: Tensor<1>,
    l2_w: Tensor<2>,
    l2_b: Tensor<1>,
    skip_w: Tensor<2>,
    skip_b: Tensor<1>,
    tau: f64,
    is_output: bool,
}

struct BurnBlockW {
    attn_qkv_w: Tensor<2>,
    attn_qkv_b: Option<Tensor<1>>,
    attn_out_w: Tensor<2>,
    attn_out_b: Option<Tensor<1>>,
    attn_pds: Option<Tensor<1>>,
    attn_tau: f64,
    ffn_up_w: Tensor<2>,
    ffn_down_w: Tensor<2>,
    mlp_tau: f64,
}

pub struct BurnTotoModel {
    config: InferConfig,
    rope: BurnXposRope,
    patch_proj: BurnResidualMlpW,
    blocks: Vec<BurnBlockW>,
    output_head: BurnResidualMlpW,
}

fn load_residual_mlp(
    weights: &GgufWeights,
    prefix: &str,
    tau: f64,
    is_output: bool,
) -> Result<BurnResidualMlpW> {
    Ok(BurnResidualMlpW {
        l1_w: load_t(weights, &format!("{prefix}.linear1.weight"))?,
        l1_b: load_v(weights, &format!("{prefix}.linear1.bias"))?,
        l2_w: load_t(weights, &format!("{prefix}.linear2.weight"))?,
        l2_b: load_v(weights, &format!("{prefix}.linear2.bias"))?,
        skip_w: load_t(weights, &format!("{prefix}.skip_proj.weight"))?,
        skip_b: load_v(weights, &format!("{prefix}.skip_proj.bias"))?,
        tau,
        is_output,
    })
}

fn compute_taus(
    num_layers: usize,
    residual_mult: f64,
    residual_attn_ratio: f64,
) -> (Vec<f64>, Vec<f64>) {
    let total_depth = 2 * num_layers;
    let alpha_mlp = residual_mult * (2.0 / (1.0 + residual_attn_ratio.powi(2))).sqrt();
    let alpha_attn = residual_attn_ratio * alpha_mlp;
    let tau = |index: usize| -> f64 {
        let n_attn = (index + 1) / 2;
        let n_mlp = index / 2;
        let num = if index % 2 == 0 {
            alpha_attn
        } else {
            alpha_mlp
        };
        let den = (total_depth as f64 / 2.0
            + n_attn as f64 * alpha_attn.powi(2)
            + n_mlp as f64 * alpha_mlp.powi(2))
        .sqrt();
        num / den
    };
    (0..num_layers)
        .map(|i| (tau(2 * i), tau(2 * i + 1)))
        .unzip()
}

impl BurnTotoModel {
    pub fn patch_size(&self) -> usize {
        self.config.patch_size()
    }

    pub fn load(gguf_path: &Path, config: InferConfig) -> Result<Self> {
        if config.compute_f64 {
            bail!("BurnTotoModel supports compute_f64=false only");
        }
        let weights = open_gguf_weights(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;

        let patch_proj = load_residual_mlp(&weights, "patch_proj", 1.0, false)?;

        let (attn_taus, mlp_taus) = compute_taus(
            config.num_layers,
            config.residual_mult,
            config.residual_attn_ratio,
        );
        let mut blocks = Vec::with_capacity(config.num_layers);
        for n in 0..config.num_layers {
            blocks.push(BurnBlockW {
                attn_qkv_w: load_t(&weights, &format!("blk.{n}.attn_qkv.weight"))?,
                attn_qkv_b: try_load_v(&weights, &format!("blk.{n}.attn_qkv.bias"))?,
                attn_out_w: load_t(&weights, &format!("blk.{n}.attn_output.weight"))?,
                attn_out_b: try_load_v(&weights, &format!("blk.{n}.attn_output.bias"))?,
                attn_pds: try_load_v(&weights, &format!("blk.{n}.attn_pds.weight"))?,
                attn_tau: attn_taus[n],
                ffn_up_w: load_t(&weights, &format!("blk.{n}.ffn_up.weight"))?,
                // Mirror the Q8 realignment the loader applies to ffn_down.
                ffn_down_w: load_w(
                    &weights,
                    &format!("blk.{n}.ffn_down.weight"),
                    512,
                )?,
                mlp_tau: mlp_taus[n],
            });
        }
        let output_head = load_residual_mlp(&weights, "output_head", 1.0, true)?;
        let rope = BurnXposRope::build(config.qk_dim, 8192);
        Ok(Self {
            config,
            rope,
            patch_proj,
            blocks,
            output_head,
        })
    }

    pub fn forecast(
        &self,
        target: &[Vec<f32>],
        mask: &[Vec<bool>],
        prediction_length: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let n_var = target.len();
        let ctx_len = target[0].len();
        let patch_size = self.config.patch_size;
        if ctx_len % patch_size != 0 {
            bail!("ctx_len ({ctx_len}) must be divisible by patch_size ({patch_size})");
        }
        let ctx_patches = ctx_len / patch_size;
        let fcst_patches = (prediction_length + patch_size - 1) / patch_size;
        let total_patches = ctx_patches + fcst_patches;

        let mut locs = Vec::with_capacity(n_var);
        let mut scales = Vec::with_capacity(n_var);
        for v in 0..n_var {
            let (loc, scale) = causal_patched_std_scaler(&target[v], &mask[v], patch_size);
            locs.push(loc);
            scales.push(scale);
        }

        let mut patch_data = vec![0.0f32; n_var * total_patches * 2 * patch_size];
        for v in 0..n_var {
            for p in 0..ctx_patches {
                let base = v * total_patches * 2 * patch_size + p * 2 * patch_size;
                for i in 0..patch_size {
                    let t = p * patch_size + i;
                    let obs = mask[v][t];
                    let val = if obs {
                        (((target[v][t] - locs[v][t]) / scales[v][t]) as f64).asinh() as f32
                    } else {
                        0.0
                    };
                    patch_data[base + i] = val;
                    patch_data[base + patch_size + i] = if obs { 0.0 } else { 1.0 };
                }
            }
            for p in ctx_patches..total_patches {
                let base = v * total_patches * 2 * patch_size + p * 2 * patch_size;
                for i in 0..patch_size {
                    patch_data[base + i] = 0.0;
                    patch_data[base + patch_size + i] = 1.0;
                }
            }
        }

        let dev = device();
        let x = Tensor::<4>::from_data(
            TensorData::new(patch_data, [1, n_var, total_patches, 2 * patch_size]),
            &dev,
        );
        let x = self.forward_residual_mlp(x, &self.patch_proj)?;
        let x = self.forward_transformer(x, n_var, total_patches)?;
        let x = self.forward_residual_mlp(x, &self.output_head)?;
        let x: Tensor<4> = x.narrow(2, ctx_patches - 1, fcst_patches);
        let fcst_steps = fcst_patches * patch_size;
        let out: Tensor<5> = x.reshape([1, n_var, fcst_patches, patch_size, 9]);
        let out: Tensor<5> = out.permute([4, 0, 1, 2, 3]);
        let out: Tensor<3> = out.reshape([9, n_var, fcst_steps]);
        let h = out.dims()[2];
        let out: Tensor<3> = if fcst_steps > prediction_length {
            out.narrow(2, 0, prediction_length)
        } else {
            out
        };
        let out_data: Vec<f32> = out
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let loc_final: Vec<f32> = (0..n_var).map(|v| locs[v][ctx_len - 1]).collect();
        let scale_final: Vec<f32> = (0..n_var).map(|v| scales[v][ctx_len - 1]).collect();
        let mut result = vec![vec![vec![0.0f32; prediction_length]; n_var]; 9];
        for q in 0..9 {
            for v in 0..n_var {
                for t in 0..prediction_length {
                    let raw = out_data[(q * n_var + v) * prediction_length + t] as f64;
                    result[q][v][t] = (raw.sinh() as f32) * scale_final[v] + loc_final[v];
                }
            }
        }
        Ok(result)
    }

    fn forward_residual_mlp<const D: usize>(
        &self,
        x: Tensor<D>,
        w: &BurnResidualMlpW,
    ) -> Result<Tensor<D>> {
        let h = uu_silu(uu_linear(x.clone(), w.l1_w.clone(), Some(w.l1_b.clone())));
        let h = if w.is_output {
            uu_linear_readout(h, w.l2_w.clone(), Some(w.l2_b.clone()))
        } else {
            uu_linear(h, w.l2_w.clone(), Some(w.l2_b.clone()))
        };
        let skip = if w.is_output {
            uu_linear_readout(x, w.skip_w.clone(), Some(w.skip_b.clone()))
        } else {
            uu_linear(x, w.skip_w.clone(), Some(w.skip_b.clone()))
        };
        Ok(residual_add(h, skip, w.tau))
    }

    fn forward_transformer(
        &self,
        mut x: Tensor<4>,
        n_var: usize,
        num_patches: usize,
    ) -> Result<Tensor<4>> {
        for (idx, blk) in self.blocks.iter().enumerate() {
            x = if self.config.is_variate_layer(idx) {
                self.forward_variate_layer(x, blk, n_var, num_patches)?
            } else {
                self.forward_time_layer(x, blk, n_var, num_patches)?
            };
        }
        Ok(rms_norm_weightless(x, self.config.norm_eps))
    }

    fn forward_time_layer(
        &self,
        x: Tensor<4>,
        blk: &BurnBlockW,
        n_var: usize,
        num_patches: usize,
    ) -> Result<Tensor<4>> {
        let cfg = &self.config;
        let state: Tensor<3> = x.reshape([n_var, num_patches, cfg.d_model]);
        let normed = rms_norm_weightless(state.clone(), cfg.norm_eps);
        let seq_ids: Vec<u32> = (0..num_patches as u32).collect();
        let attn_out = self.forward_attention(&normed, blk, &seq_ids, false)?;
        let state = residual_add(attn_out, state, blk.attn_tau);
        let normed = rms_norm_weightless(state.clone(), cfg.norm_eps);
        let ffn_out = self.forward_ffn(&normed, blk)?;
        let state = residual_add(ffn_out, state, blk.mlp_tau);
        Ok(state.reshape([1, n_var, num_patches, cfg.d_model]))
    }

    fn forward_variate_layer(
        &self,
        x: Tensor<4>,
        blk: &BurnBlockW,
        n_var: usize,
        num_patches: usize,
    ) -> Result<Tensor<4>> {
        let cfg = &self.config;
        let state: Tensor<3> = x
            .permute([0, 2, 1, 3])
            .reshape([num_patches, n_var, cfg.d_model]);
        let normed = rms_norm_weightless(state.clone(), cfg.norm_eps);
        let attn_out = self.forward_attention(&normed, blk, &[], true)?;
        let state = residual_add(attn_out, state, blk.attn_tau);
        let normed = rms_norm_weightless(state.clone(), cfg.norm_eps);
        let ffn_out = self.forward_ffn(&normed, blk)?;
        let state = residual_add(ffn_out, state, blk.mlp_tau);
        Ok(state
            .reshape([1, num_patches, n_var, cfg.d_model])
            .permute([0, 2, 1, 3]))
    }

    fn forward_attention(
        &self,
        state: &Tensor<3>,
        blk: &BurnBlockW,
        seq_ids: &[u32],
        is_variate: bool,
    ) -> Result<Tensor<3>> {
        let cfg = &self.config;
        let dims = state.dims();
        let (batch, seq) = (dims[0], dims[1]);
        let qkv = uu_linear(
            state.clone(),
            blk.attn_qkv_w.clone(),
            blk.attn_qkv_b.clone(),
        );
        let q: Tensor<3> = qkv.clone().narrow(2, 0, 512);
        let k: Tensor<3> = qkv.clone().narrow(2, 512, 512);
        let v: Tensor<3> = qkv.narrow(2, 1024, 512);
        let pack = |t: Tensor<3>, heads: usize, hd: usize| {
            t.reshape([batch, seq, heads, hd]).permute([0, 2, 1, 3])
        };
        let (q, k, v) = (
            pack(q, cfg.num_heads, cfg.qk_dim),
            pack(k, cfg.num_groups, cfg.qk_dim),
            pack(v, cfg.num_groups, cfg.v_dim),
        );
        let q = match blk.attn_pds.as_ref() {
            Some(pds) => apply_per_dim_scale(q, pds.clone()),
            None => q,
        };
        let (q, k) = if !is_variate && !seq_ids.is_empty() && cfg.use_xpos {
            (
                self.rope.apply(q, seq_ids, 1.0),
                self.rope.apply(k, seq_ids, -1.0),
            )
        } else {
            (q, k)
        };
        let scale = 1.0f32 / cfg.qk_dim as f32;
        let scores = q.matmul(k.transpose()).mul_scalar(scale);
        let scores = if !is_variate {
            scores + burn_causal_mask(seq)
        } else {
            scores
        };
        let attn = activation::softmax(scores, 3);
        let out = attn.matmul(v);
        let out: Tensor<3> =
            out.permute([0, 2, 1, 3])
                .reshape([batch, seq, cfg.num_heads * cfg.v_dim]);
        Ok(uu_linear(
            out,
            blk.attn_out_w.clone(),
            blk.attn_out_b.clone(),
        ))
    }

    fn forward_ffn(&self, x: &Tensor<3>, blk: &BurnBlockW) -> Result<Tensor<3>> {
        let fc1_out = uu_linear(x.clone(), blk.ffn_up_w.clone(), None);
        let half = fc1_out.dims()[2] / 2;
        let gate = fc1_out.clone().narrow(2, 0, half);
        let val = fc1_out.narrow(2, half, half);
        let activated = gate * activation::silu(val);
        Ok(uu_linear(activated, blk.ffn_down_w.clone(), None))
    }
}

fn burn_causal_mask(seq: usize) -> Tensor<4> {
    let dev = device();
    let mut data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::<2>::from_data(TensorData::new(data, [seq, seq]), &dev)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0)
}

fn causal_patched_std_scaler(
    data: &[f32],
    mask: &[bool],
    patch_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = data.len();
    let num_patches = n / patch_size;
    let mut cum_count = 0.0f64;
    let mut m1 = 0.0f64;
    let mut m2 = 0.0f64;
    let correction = 1.0f64;
    let minimum_scale = 1e-6f64;
    let mut patch_loc = vec![0.0f32; num_patches];
    let mut patch_scale = vec![1e-6f32; num_patches];
    for p in 0..num_patches {
        for i in 0..patch_size {
            let t = p * patch_size + i;
            if mask[t] {
                let x = data[t] as f64;
                cum_count += 1.0;
                let prev_m1 = m1;
                m1 += (x - m1) / cum_count;
                m2 += (x - prev_m1) * (x - m1);
            }
        }
        patch_loc[p] = m1 as f32;
        let denom = (cum_count - correction).max(1.0);
        patch_scale[p] = (m2 / denom).sqrt().max(minimum_scale) as f32;
    }
    let mut loc = vec![0.0f32; n];
    let mut scale = vec![1e-6f32; n];
    for p in 0..num_patches {
        for i in 0..patch_size {
            let t = p * patch_size + i;
            loc[t] = patch_loc[p];
            scale[t] = patch_scale[p];
        }
    }
    (loc, scale)
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

    fn tiny_config() -> InferConfig {
        InferConfig::default()
            .with_d_model(512)
            .with_num_layers(6)
            .with_num_heads(8)
            .with_num_groups(8)
            .with_qk_dim(64)
            .with_v_dim(64)
            .with_patch_size(4)
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

    fn put_mlp(
        w: &mut GGUFWriter,
        prefix: &str,
        h1: usize,
        d_in: usize,
        h2: usize,
        seed: &mut u64,
        s: f32,
    ) {
        put2(w, &format!("{prefix}.linear1.weight"), h1, d_in, seed, s);
        put1(w, &format!("{prefix}.linear1.bias"), h1, seed);
        put2(w, &format!("{prefix}.linear2.weight"), h2, h1, seed, s);
        put1(w, &format!("{prefix}.linear2.bias"), h2, seed);
        put2(w, &format!("{prefix}.skip_proj.weight"), h2, d_in, seed, s);
        put1(w, &format!("{prefix}.skip_proj.bias"), h2, seed);
    }

    #[test]
    fn burn_forecast_golden() {
        // d=16, heads=groups=4, qk=v=8, patch=4, 1 layer, all time layers.
        let dir = std::env::temp_dir().join(format!("toto-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata("general.architecture", GGUFMetaValue::String("toto".into()));
        let mut seed = 1100u64;
        let s = 0.05;
        put_mlp(&mut w, "patch_proj", 32, 8, 16, &mut seed, s);
        put2(&mut w, "blk.0.attn_qkv.weight", 96, 16, &mut seed, s);
        put1(&mut w, "blk.0.attn_qkv.bias", 96, &mut seed);
        put2(&mut w, "blk.0.attn_output.weight", 16, 32, &mut seed, s);
        put1(&mut w, "blk.0.attn_output.bias", 16, &mut seed);
        put1(&mut w, "blk.0.attn_pds.weight", 8, &mut seed);
        put2(&mut w, "blk.0.ffn_up.weight", 32, 16, &mut seed, s);
        put2(&mut w, "blk.0.ffn_down.weight", 16, 16, &mut seed, s);
        put_mlp(&mut w, "output_head", 32, 16, 36, &mut seed, s);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        assert!(!cfg.compute_f64);
        let burn = BurnTotoModel::load(&path, cfg).unwrap();
        let ctx = vec![(0..8).map(|i| 10.0 + 0.5 * i as f32).collect::<Vec<_>>()];
        let mask = vec![vec![true; 8]];
        let t0 = std::time::Instant::now();
        let a = burn.forecast(&ctx, &mask, 4).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let b = burn.forecast(&ctx, &mask, 4).unwrap();
        assert_eq!(a.len(), 9);
        for q in &a {
            assert_eq!(q.len(), 1);
            for v in q {
                assert_eq!(v.len(), 4);
            }
        }
        assert!(a.iter().flatten().flatten().all(|v| v.is_finite()));
        assert!(a.iter().flatten().flatten().any(|&v| v != 0.0));
        assert_eq!(a, b, "forecast must be deterministic");
        println!("toto synthetic: burn {burn_ms:.1}ms out={a:?}");
        // Golden values recorded from the verified Burn port (ex-candle parity 1e-3).
        let flat: Vec<f32> = a.iter().flatten().flatten().copied().collect();
        let golden: [f32; 36] = [
            11.704744, 11.739721, 11.7525215, 11.725984, 11.722208, 11.724002, 11.743584,
            11.738862, 11.74826, 11.730129, 11.741938, 11.736687, 11.761163, 11.724331, 11.729776,
            11.752375, 11.711724, 11.757467, 11.735854, 11.761217, 11.6961, 11.734921, 11.741099,
            11.715156, 11.7233715, 11.694718, 11.721759, 11.7119465, 11.716567, 11.721282,
            11.730694, 11.710778, 11.740484, 11.723714, 11.738366, 11.721424,
        ];
        assert_eq!(flat.len(), golden.len());
        for (x, y) in flat.iter().zip(golden.iter()) {
            assert!((x - y).abs() < 1e-3, "golden drift: {x} vs {y}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
