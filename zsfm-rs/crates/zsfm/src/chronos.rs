use std::path::PathBuf;

use crate::common::DtypeArg;
use anyhow::Context;
use clap::Subcommand;

use zsfm_chronos::config::Chronos2Config;
use zsfm_chronos::convert::{convert, ConvertOptions};
use zsfm_gguf::GGMLType;

#[derive(Subcommand)]
pub enum Command {
    /// Download a Chronos-2 model from HuggingFace and convert it to GGUF.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF to
    /// `<model_dir>/<owner>__<name>/model-f32.gguf`, deleting the (large) downloaded
    /// weight files afterward. Later conversions (any --dtype) recast from that
    /// cached F32 GGUF instead of re-downloading — pass --redownload to force a
    /// fresh download anyway (e.g. the repo was updated).
    Convert {
        #[arg(short, long, default_value = "amazon/chronos-2")]
        model: String,
        #[arg(short, long, default_value = "gguf/chronos-f16.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f16")]
        dtype: DtypeArg,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        /// Force a fresh download even if a cached F32 GGUF already exists.
        #[arg(long)]
        redownload: bool,
    },
    /// Print all tensor names in a local checkpoint file (safetensors, PyTorch
    /// pickle, GGUF, or npy/npz — format auto-detected).
    InspectTensors { path: PathBuf },
    /// Run Chronos-2 forecasting from a GGUF file.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format with all quantile levels.
    /// The `point` field contains the median quantile (q0.5). Univariate only.
    Infer {
        #[arg(short, long, default_value = "gguf/chronos-f16.gguf")]
        gguf: PathBuf,
        #[arg(long, default_value = "models/amazon__chronos-2/config.json")]
        config: PathBuf,
        /// Group IDs for multivariate forecasting (comma-separated, e.g. "0,0,1,1").
        /// Variates with the same group ID attend to each other; different groups are isolated.
        /// If omitted, all variates share one group. Can also be supplied via JSON: {"group_ids": [0,0,1,1]}.
        #[arg(long)]
        group_ids: Option<String>,
    },
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        #[arg(short, long, default_value = "amaye15/chronos-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Delete cached model files for this model.
    ///
    /// Removes the canonical F32 GGUF and config cache at
    /// `<model_dir>/<owner>__<name>/` created by `convert`. By default only
    /// the cache is removed; pass `--output <path>` to also delete a
    /// converted GGUF file (e.g. `gguf/...`).
    Delete {
        /// HuggingFace repo id (must match the `convert` --model you used).
        #[arg(short, long, default_value = "amazon/chronos-2")]
        model: String,
        /// Directory where the cache was written (must match `convert` --model-dir).
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        /// Also delete this output GGUF file if it exists.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Delete {
            model,
            model_dir,
            output,
        } => {
            let canonical = zsfm_hub::canonical_gguf_path(&model_dir, &model);
            crate::common::delete_cached_model(&canonical, output.as_deref())?;
        }
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::Convert {
            model,
            output,
            dtype,
            model_dir,
            token,
            redownload,
        } => {
            let canonical = zsfm_hub::canonical_gguf_path(&model_dir, &model);
            if crate::common::try_recast_from_cache(&canonical, &output, dtype.into(), redownload)?
            {
                return Ok(());
            }

            eprintln!("Downloading {model} into {} …", model_dir.display());
            let files = zsfm_hub::download_model(&model, token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config_str =
                std::fs::read_to_string(&files.config_json).context("read config.json")?;
            let config = Chronos2Config::from_json(&config_str).context("parse config.json")?;

            println!(
                "Config: {} layers, d_model={}, d_ff={}, {} heads, patch_size={}",
                config.num_layers,
                config.d_model,
                config.d_ff,
                config.num_heads,
                config.chronos_config.input_patch_size,
            );

            let f32_opts = ConvertOptions {
                output_dtype: GGMLType::F32,
            };
            convert(&model, &files, &config, &f32_opts, &canonical)?;
            eprintln!("Wrote canonical F32 GGUF to {} …", canonical.display());
            files.cleanup_weights();

            zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
            eprintln!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => crate::common::inspect_tensors(&path)?,

        Command::Infer { gguf, config, group_ids } => {
            return run_infer_burn(&gguf, &config, group_ids).await;
        }
    }

    Ok(())
}

async fn run_infer_burn(
    gguf: &std::path::PathBuf,
    config: &std::path::PathBuf,
    cli_group_ids: Option<String>,
) -> anyhow::Result<()> {
    use zsfm_chronos::infer::BurnChronosModel;

    let buf = zsfm_core::read_stdin_limited()?;
    let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

    let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
    let horizon = zsfm_core::parse_horizon(&req)?;

    // Group IDs precedence: --group-ids (CLI) > JSON "group_ids" > None (all one group).
    let req_group_ids: Option<Vec<usize>> = req
        .get("group_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_u64().map(|n| n as usize))
                .collect::<Vec<_>>()
        });
    let group_ids: Option<Vec<usize>> = if let Some(s) = cli_group_ids.as_deref() {
        let parsed: Result<Vec<usize>, _> = s
            .split(',')
            .map(|p| p.trim().parse::<usize>())
            .collect();
        Some(parsed.context(
            "--group-ids: expected comma-separated integers, e.g. \"0,0,1,1\"",
        )?)
    } else {
        req_group_ids
    };

    let config_str =
        std::fs::read_to_string(config).with_context(|| format!("read {}", config.display()))?;
    let c2_config = Chronos2Config::from_json(&config_str).context("parse config.json")?;

    eprintln!("Loading Burn model from {} …", gguf.display());
    let model = BurnChronosModel::load(gguf, zsfm_chronos::infer::InferConfig::from(&c2_config))
        .context("load model")?;

    let patch_size = model.patch_size();
    let patch_stride = model.patch_stride();
    let max_ctx = model.context_length();
    let quantile_levels = model.quantiles().to_vec();
    let median_idx = crate::common::median_index(&quantile_levels);

    let mut fc_outputs = Vec::new();
    let mut total_ctx: usize = 0;
    for raw_variates in &contexts {
        anyhow::ensure!(!raw_variates.is_empty(), "context series must not be empty");
        let n_var = raw_variates.len();
        for (vi, s) in raw_variates.iter().enumerate() {
            anyhow::ensure!(!s.is_empty(), "context[{vi}] series must not be empty");
        }
        let min_len = raw_variates.iter().map(|s| s.len()).min().unwrap_or(0);
        total_ctx += min_len;
        let usable = min_len.min(max_ctx);
        let ctx_len = (usable / patch_stride) * patch_stride;
        anyhow::ensure!(
            ctx_len >= patch_size,
            "context too short - need at least {patch_size} timesteps, got {ctx_len}"
        );
        let start = min_len.saturating_sub(ctx_len);
        let trimmed: Vec<Vec<f32>> = raw_variates
            .iter()
            .map(|s| s[start..start + ctx_len].to_vec())
            .collect();
        let qmat: zsfm_core::QuantileMatrix = if n_var == 1 {
            eprintln!(
                "Running univariate forecast ({} context steps -> {horizon} future steps) ...",
                ctx_len
            );
            let quantile_mat = model.forecast(&trimmed[0], horizon).context("forecast")?;
            quantile_mat.into_iter().map(|row| vec![row]).collect()
        } else {
            let groups_for_this: Option<&[usize]> = match group_ids.as_deref() {
                Some(g) => {
                    anyhow::ensure!(
                        g.len() == n_var,
                        "--group-ids / group_ids length {} != variate count {}",
                        g.len(),
                        n_var
                    );
                    Some(g)
                }
                None => None,
            };
            let groups_msg = match groups_for_this {
                Some(g) => format!(", group_ids={g:?}"),
                None => String::new(),
            };
            eprintln!(
                "Running multivariate forecast ({} variates x {} context steps -> {horizon} future steps{}) ...",
                n_var, ctx_len, groups_msg
            );
            let per_variate = model
                .forecast_multivariate(&trimmed, groups_for_this, horizon)
                .context("forecast_multivariate")?;
            let n_q = per_variate[0].len();
            let mut out: zsfm_core::QuantileMatrix = vec![vec![Vec::new(); n_var]; n_q];
            for v in 0..n_var {
                for q in 0..n_q {
                    out[q][v] = per_variate[v][q].clone();
                }
            }
            out
        };
        fc_outputs.push(zsfm_core::quantile_matrix_to_output(
            &qmat,
            &quantile_levels,
            median_idx,
        ));
    }

    println!(
        "{}",
        zsfm_core::forecast_response_json("chronos", total_ctx, horizon, fc_outputs)?
    );
    Ok(())
}
