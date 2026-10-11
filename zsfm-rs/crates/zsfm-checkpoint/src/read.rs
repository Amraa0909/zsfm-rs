//! Format-agnostic checkpoint loading.
//!
//! Supported inputs (detected by extension, with a magic-byte fallback):
//! - `.safetensors` — single file or several shard files
//! - `model.safetensors.index.json` — HF sharded-model index (shards resolved
//!   relative to the index file)
//! - `.pt` / `.pth` / `.bin` / `.ckpt` — PyTorch pickle checkpoints (zip-based
//!   `torch.save` format) via the hardened `pytorch-reader` crate
//! - `.npy` / `.npz` — NumPy arrays
//! - `.gguf` — existing GGUF files (tensors dequantized to F32; metadata carried
//!   through), which makes dtype re-quantization possible
//!
//! Every tensor is returned as raw little-endian bytes plus a [`SrcDtype`]
//! (F32/F16/BF16 preserved exactly; other dtypes — F64, integers — are cast to
//! F32 with a warning).

use std::collections::HashSet;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use safetensors::{Dtype as StDtype, SafeTensors};

use zsfm_gguf::GGUFMetaValue;

use crate::cast::{f32_to_bytes, SrcDtype};

/// One tensor read from a checkpoint: name, row-major (python-order) shape,
/// source dtype, and raw little-endian bytes.
pub struct RawTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: SrcDtype,
    pub data: Vec<u8>,
}

/// A loaded checkpoint: all tensors plus any metadata carried over from the
/// source (only GGUF inputs have metadata).
pub struct Checkpoint {
    pub tensors: Vec<RawTensor>,
    pub metadata: Vec<(String, GGUFMetaValue)>,
}

#[derive(Default)]
pub struct LoadOptions {
    /// Descend into this top-level dict before collecting tensors (pickle
    /// inputs only), e.g. `state_dict` for PyTorch Lightning checkpoints.
    pub pickle_key: Option<String>,
    /// Strip this prefix from every tensor name that carries it.
    pub strip_prefix: Option<String>,
}

/// Load one or more checkpoint files into a single [`Checkpoint`].
/// Multiple inputs (e.g. safetensors shards) are merged; duplicate tensor
/// names across inputs are an error.
pub fn load_checkpoint(inputs: &[PathBuf], opts: &LoadOptions) -> Result<Checkpoint> {
    anyhow::ensure!(!inputs.is_empty(), "no input files given");

    let mut tensors: Vec<RawTensor> = Vec::new();
    let mut metadata: Vec<(String, GGUFMetaValue)> = Vec::new();

    for path in inputs {
        let ckpt =
            load_one(path, opts).with_context(|| format!("load checkpoint {}", path.display()))?;
        tensors.extend(ckpt.tensors);
        metadata.extend(ckpt.metadata);
    }

    if let Some(prefix) = &opts.strip_prefix {
        for t in &mut tensors {
            if let Some(rest) = t.name.strip_prefix(prefix.as_str()) {
                t.name = rest.to_string();
            }
        }
    }

    let mut seen = HashSet::new();
    for t in &tensors {
        if !seen.insert(t.name.as_str()) {
            bail!("duplicate tensor name across inputs: {}", t.name);
        }
    }

    Ok(Checkpoint { tensors, metadata })
}

/// TensorFlow SavedModel / TF-checkpoint weights use the TensorBundle format,
/// which has no trustworthy pure-Rust reader — point at the bundled exporter
/// script instead of failing cryptically.
const TF_BUNDLE_HELP: &str =
    "TensorFlow SavedModel / checkpoint weights (TensorBundle format) can't be read \
     natively. Export them to safetensors first with the bundled script (needs \
     tensorflow + safetensors installed):\n\n    \
     python zsfm-rs/tools/export-weights.py <saved_model_dir | ckpt_prefix> weights.safetensors\n\n\
     then run: zsfm convert weights.safetensors -o model.gguf\n\
     (Keras .h5 / .keras files ARE supported natively — no export needed.)";

fn load_one(path: &Path, opts: &LoadOptions) -> Result<Checkpoint> {
    let fname = path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if path.is_dir() {
        if path.join("saved_model.pb").exists() {
            bail!(
                "{} is a TensorFlow SavedModel directory. {TF_BUNDLE_HELP}",
                path.display()
            );
        }
        bail!(
            "{} is a directory — pass a checkpoint file (or for sharded safetensors, \
             the model.safetensors.index.json)",
            path.display()
        );
    }
    if ext == "index" || fname.contains(".data-00") {
        bail!(
            "{} looks like a TensorFlow checkpoint file. {TF_BUNDLE_HELP}",
            path.display()
        );
    }

    if fname.ends_with(".safetensors.index.json") {
        return load_safetensors_index(path);
    }
    match ext.as_str() {
        "safetensors" => load_safetensors(path),
        "pt" | "pth" | "bin" | "ckpt" => load_pickle(path, opts),
        "npz" => load_npz(path),
        "npy" => load_npy(path),
        "gguf" => load_gguf(path),
        "onnx" => crate::onnx::load_onnx(path),
        "h5" | "hdf5" => crate::hdf5_keras::load_hdf5(path),
        "keras" => crate::hdf5_keras::load_keras(path),
        _ => match sniff_format(path)? {
            Sniffed::SafeTensors => load_safetensors(path),
            Sniffed::Gguf => load_gguf(path),
            Sniffed::Hdf5 => crate::hdf5_keras::load_hdf5(path),
            Sniffed::Unknown => bail!(
                "unrecognized checkpoint format for {} — supported: .safetensors, \
                 model.safetensors.index.json, .pt/.pth/.bin/.ckpt (PyTorch pickle), \
                 .npy/.npz, .onnx, .h5/.hdf5, .keras, .gguf",
                path.display()
            ),
        },
    }
}

enum Sniffed {
    SafeTensors,
    Gguf,
    Hdf5,
    Unknown,
}

fn sniff_format(path: &Path) -> Result<Sniffed> {
    use std::io::Read;
    let mut head = [0u8; 16];
    let n = std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .read(&mut head)?;
    if n >= 4 && &head[..4] == b"GGUF" {
        return Ok(Sniffed::Gguf);
    }
    if n >= 8 && &head[..8] == b"\x89HDF\r\n\x1a\n" {
        return Ok(Sniffed::Hdf5);
    }
    // safetensors: u64 LE header length followed by a JSON object.
    if n >= 9 {
        let header_len = u64::from_le_bytes(head[..8].try_into().unwrap());
        let file_len = std::fs::metadata(path)?.len();
        if header_len > 0 && header_len.saturating_add(8) <= file_len && head[8] == b'{' {
            return Ok(Sniffed::SafeTensors);
        }
    }
    Ok(Sniffed::Unknown)
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

fn load_safetensors(path: &Path) -> Result<Checkpoint> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("deserialize safetensors")?;

    let mut tensors = Vec::with_capacity(st.len());
    for (name, view) in st.tensors() {
        let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
        let (dtype, data) = match view.dtype() {
            StDtype::F32 => (SrcDtype::F32, view.data().to_vec()),
            StDtype::F16 => (SrcDtype::F16, view.data().to_vec()),
            StDtype::BF16 => (SrcDtype::BF16, view.data().to_vec()),
            other => {
                eprintln!("note: tensor {name}: casting {other:?} to F32");
                let n: usize = view.shape().iter().product();
                let t = safetensors_view_to_f32(&view, other, n)
                    .with_context(|| format!("tensor {name}: unsupported dtype {other:?}"))?;
                (SrcDtype::F32, f32_to_bytes(&t))
            }
        };
        tensors.push(RawTensor {
            name: name.to_string(),
            shape,
            dtype,
            data,
        });
    }
    Ok(Checkpoint {
        tensors,
        metadata: Vec::new(),
    })
}

fn safetensors_view_to_f32(
    view: &safetensors::tensor::TensorView,
    dtype: StDtype,
    n_elems: usize,
) -> Result<Vec<f32>> {
    let data = view.data();
    let mut out = Vec::with_capacity(n_elems);
    match dtype {
        StDtype::F64 => {
            for c in data.chunks_exact(8) {
                out.push(f64::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        StDtype::I64 => {
            for c in data.chunks_exact(8) {
                out.push(i64::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        StDtype::I32 => {
            for c in data.chunks_exact(4) {
                out.push(i32::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        StDtype::I16 => {
            for c in data.chunks_exact(2) {
                out.push(i16::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        StDtype::I8 => out.extend(data.iter().map(|&b| b as i8 as f32)),
        StDtype::U8 => out.extend(data.iter().map(|&b| b as f32)),
        StDtype::BOOL => out.extend(data.iter().map(|&b| (b != 0) as u8 as f32)),
        other => bail!("safetensors dtype {other:?} not supported"),
    }
    Ok(out)
}

fn load_safetensors_index(path: &Path) -> Result<Checkpoint> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw).context("parse index json")?;
    let map = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("index json has no weight_map object")?;

    let mut shard_names: Vec<String> = map
        .values()
        .filter_map(|s| s.as_str().map(str::to_string))
        .collect();
    shard_names.sort();
    shard_names.dedup();

    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut tensors = Vec::new();
    for name in &shard_names {
        let shard_path = dir.join(name);
        let ckpt = load_safetensors(&shard_path)
            .with_context(|| format!("load shard {}", shard_path.display()))?;
        tensors.extend(ckpt.tensors);
    }
    Ok(Checkpoint {
        tensors,
        metadata: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// PyTorch pickle (via pytorch-reader: hardened limits, no PyTorch needed)
// ---------------------------------------------------------------------------

fn load_pickle(path: &Path, opts: &LoadOptions) -> Result<Checkpoint> {
    // Plain `torch.save(model.state_dict())` checkpoints hold tensors at the top
    // level, but wrapper checkpoints (PyTorch Lightning, training states) nest
    // them under "state_dict" — retry with the conventional key before giving up.
    let reader = match open_pickle_reader(path, opts.pickle_key.as_deref()) {
        Ok(r) if !r.tensors().is_empty() => r,
        _ if opts.pickle_key.is_none() => {
            let retried = open_pickle_reader(path, Some("state_dict"))
                .with_context(|| format!("read PyTorch checkpoint {}", path.display()))?;
            if !retried.tensors().is_empty() {
                eprintln!("note: no top-level tensors — using the \"state_dict\" dict instead");
                retried
            } else {
                open_pickle_reader(path, opts.pickle_key.as_deref())
                    .with_context(|| format!("read PyTorch checkpoint {}", path.display()))?
            }
        }
        Err(e) => return Err(e),
        Ok(r) => r,
    };
    anyhow::ensure!(
        !reader.tensors().is_empty(),
        "no tensors found in {} (if the checkpoint nests weights under a custom dict, \
         pass --pickle-key <KEY>)",
        path.display()
    );

    let mut names: Vec<String> = reader.tensors().keys().cloned().collect();
    names.sort();
    let mut tensors = Vec::with_capacity(names.len());
    for name in &names {
        let t = reader
            .tensors()
            .get(name)
            .with_context(|| format!("tensor {name} vanished"))?;
        let shape: Vec<u64> = t.shape().iter().map(|&d| d as u64).collect();
        let raw = t
            .read()
            .with_context(|| format!("read tensor {name} bytes"))?;
        let (dtype, data) = pickle_bytes_to_raw(name, t.dtype(), &raw)?;
        tensors.push(RawTensor {
            name: name.clone(),
            shape,
            dtype,
            data,
        });
    }
    Ok(Checkpoint {
        tensors,
        metadata: Vec::new(),
    })
}

fn open_pickle_reader(path: &Path, key: Option<&str>) -> Result<pytorch_reader::PytorchReader> {
    let path_str = path.to_string_lossy();
    match key {
        Some(k) => pytorch_reader::PytorchReader::with_top_level_key(path_str.as_ref(), k)
            .with_context(|| format!("read PyTorch checkpoint {}", path.display())),
        None => pytorch_reader::PytorchReader::new(path_str.as_ref())
            .with_context(|| format!("read PyTorch checkpoint {}", path.display())),
    }
}

/// Map a pickle tensor's bytes to [`RawTensor`] parts. F32/F16/BF16 pass
/// through exactly; anything else casts to F32 exactly like the safetensors
/// path above (same rounding, same notes on stderr).
fn pickle_bytes_to_raw(
    name: &str,
    dtype: pytorch_reader::DType,
    raw: &[u8],
) -> Result<(SrcDtype, Vec<u8>)> {
    use pytorch_reader::DType as P;
    match dtype {
        P::F32 => Ok((SrcDtype::F32, raw.to_vec())),
        P::F16 => Ok((SrcDtype::F16, raw.to_vec())),
        P::BF16 => Ok((SrcDtype::BF16, raw.to_vec())),
        P::F64 => Ok((
            SrcDtype::F32,
            raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect::<Vec<_>>()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        )),
        P::I64 => {
            eprintln!("note: tensor {name}: casting I64 to F32");
            Ok((
                SrcDtype::F32,
                raw.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect::<Vec<_>>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ))
        }
        P::I32 => {
            eprintln!("note: tensor {name}: casting I32 to F32");
            Ok((
                SrcDtype::F32,
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect::<Vec<_>>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ))
        }
        P::I16 => {
            eprintln!("note: tensor {name}: casting I16 to F32");
            Ok((
                SrcDtype::F32,
                raw.chunks_exact(2)
                    .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect::<Vec<_>>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ))
        }
        P::I8 => {
            eprintln!("note: tensor {name}: casting I8 to F32");
            Ok((
                SrcDtype::F32,
                f32_to_bytes(&raw.iter().map(|&b| b as i8 as f32).collect::<Vec<_>>()),
            ))
        }
        P::U64 => {
            eprintln!("note: tensor {name}: casting U64 to F32");
            Ok((
                SrcDtype::F32,
                raw.chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect::<Vec<_>>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ))
        }
        P::U32 => {
            eprintln!("note: tensor {name}: casting U32 to F32");
            Ok((
                SrcDtype::F32,
                raw.chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect::<Vec<_>>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ))
        }
        P::U16 => {
            eprintln!("note: tensor {name}: casting U16 to F32");
            Ok((
                SrcDtype::F32,
                raw.chunks_exact(2)
                    .map(|c| u16::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect::<Vec<_>>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ))
        }
        P::U8 => {
            eprintln!("note: tensor {name}: casting U8 to F32");
            Ok((
                SrcDtype::F32,
                f32_to_bytes(&raw.iter().map(|&b| b as f32).collect::<Vec<_>>()),
            ))
        }
        P::Bool => {
            eprintln!("note: tensor {name}: casting Bool to F32");
            Ok((
                SrcDtype::F32,
                f32_to_bytes(
                    &raw.iter()
                        .map(|&b| (b != 0) as u8 as f32)
                        .collect::<Vec<_>>(),
                ),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// NumPy (manual .npy parser — no tensor library needed)
// ---------------------------------------------------------------------------

/// One parsed `.npy` array: row-major little-endian bytes plus dtype/shape.
struct NpyArray {
    kind: NpyDtype,
    shape: Vec<u64>,
    data: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NpyDtype {
    F32,
    F16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

fn load_npz(path: &Path) -> Result<Checkpoint> {
    let file = std::fs::File::open(path).with_context(|| format!("read npz {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).context("open npz zip")?;
    let mut names: Vec<String> = (0..zip.len())
        .filter_map(|i| zip.by_index(i).ok().map(|f| f.name().to_string()))
        .filter(|n| n.ends_with(".npy"))
        .collect();
    names.sort();
    anyhow::ensure!(
        !names.is_empty(),
        "npz {} holds no .npy entries",
        path.display()
    );
    let mut tensors = Vec::with_capacity(names.len());
    for name in &names {
        let mut f = zip
            .by_name(name)
            .with_context(|| format!("npz entry {name} missing"))?;
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut f, &mut bytes)?;
        let arr = parse_npy(&bytes, name)?;
        tensors.push(npy_to_raw(name.trim_end_matches(".npy"), arr)?);
    }
    Ok(Checkpoint {
        tensors,
        metadata: Vec::new(),
    })
}

fn load_npy(path: &Path) -> Result<Checkpoint> {
    let bytes = std::fs::read(path).with_context(|| format!("read npy {}", path.display()))?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tensor")
        .to_string();
    let arr = parse_npy(&bytes, &name)?;
    Ok(Checkpoint {
        tensors: vec![npy_to_raw(&name, arr)?],
        metadata: Vec::new(),
    })
}

/// Parse a `.npy` v1/v2 buffer. Supports C-order little-endian numeric
/// dtypes; Fortran-order arrays are rejected (convert with `np.ascontiguousarray`).
fn parse_npy(bytes: &[u8], name: &str) -> Result<NpyArray> {
    anyhow::ensure!(bytes.len() >= 10, "tensor {name}: file too short for .npy");
    anyhow::ensure!(&bytes[..6] == b"\x93NUMPY", "tensor {name}: bad .npy magic");
    let (header_len, header_off) = match bytes[6] {
        1 => {
            anyhow::ensure!(bytes.len() >= 10, "tensor {name}: truncated .npy header");
            (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
        }
        2 => {
            anyhow::ensure!(bytes.len() >= 12, "tensor {name}: truncated .npy header");
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => anyhow::bail!("tensor {name}: unsupported .npy version {v}"),
    };
    anyhow::ensure!(
        bytes.len() >= header_off + header_len,
        "tensor {name}: truncated .npy header"
    );
    let header = std::str::from_utf8(&bytes[header_off..header_off + header_len])
        .context(format!("tensor {name}: .npy header is not UTF-8"))?;
    let descr = header
        .split("'descr'")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .with_context(|| format!("tensor {name}: .npy header lacks 'descr'"))?;
    let fortran = header.contains("'fortran_order': True");
    anyhow::ensure!(!fortran, "tensor {name}: Fortran-order .npy not supported");
    let shape: Vec<u64> = header
        .split("'shape'")
        .nth(1)
        .and_then(|s| s.split(['(', ')']).nth(1))
        .map(|s| {
            s.split(',')
                .filter_map(|n| n.trim().parse::<u64>().ok())
                .collect()
        })
        .unwrap_or_default();
    let (kind, elem) = match descr {
        "<f4" => (NpyDtype::F32, 4),
        "<f2" => (NpyDtype::F16, 2),
        ">f4" | ">f2" => anyhow::bail!("tensor {name}: big-endian .npy not supported ({descr})"),
        "<f8" => (NpyDtype::F64, 8),
        "<i8" => (NpyDtype::I64, 8),
        "<i4" => (NpyDtype::I32, 4),
        "<i2" => (NpyDtype::I16, 2),
        "<i1" | "|i1" => (NpyDtype::I8, 1),
        "<u8" => (NpyDtype::U64, 8),
        "<u4" => (NpyDtype::U32, 4),
        "<u2" => (NpyDtype::U16, 2),
        "|u1" => (NpyDtype::U8, 1),
        "|b1" => (NpyDtype::Bool, 1),
        _ => anyhow::bail!("tensor {name}: unsupported .npy dtype {descr}"),
    };
    // BF16 has no NumPy short code; the only 2-byte LE float here is F16.
    let data = bytes[header_off + header_len..].to_vec();
    let n_elems: usize = shape.iter().product::<u64>().max(1) as usize;
    anyhow::ensure!(
        data.len() == n_elems * elem,
        "tensor {name}: .npy body is {} bytes, want {}",
        data.len(),
        n_elems * elem
    );
    Ok(NpyArray { kind, shape, data })
}

fn npy_to_raw(name: &str, arr: NpyArray) -> Result<RawTensor> {
    let (dtype, data) = match arr.kind {
        NpyDtype::F32 => (SrcDtype::F32, arr.data),
        NpyDtype::F16 => (SrcDtype::F16, arr.data),
        _ => {
            eprintln!("note: tensor {name}: casting {:?} to F32", arr.kind);
            (SrcDtype::F32, npy_cast_to_f32(&arr.data, arr.kind)?)
        }
    };
    Ok(RawTensor {
        name: name.to_string(),
        shape: arr.shape,
        dtype,
        data,
    })
}

/// Cast a non-F32/F16 `.npy` body to F32 LE bytes.
fn npy_cast_to_f32(data: &[u8], kind: NpyDtype) -> Result<Vec<u8>> {
    let vals: Vec<f32> = match kind {
        NpyDtype::F32 | NpyDtype::F16 => {
            anyhow::bail!("npy_cast_to_f32 called for passthrough dtype")
        }
        NpyDtype::F64 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        NpyDtype::I64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        NpyDtype::I32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        NpyDtype::I16 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        NpyDtype::I8 => data.iter().map(|&b| b as i8 as f32).collect(),
        NpyDtype::U64 => data
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        NpyDtype::U32 => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        NpyDtype::U16 => data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        NpyDtype::U8 => data.iter().map(|&b| b as f32).collect(),
        NpyDtype::Bool => data.iter().map(|&b| (b != 0) as u8 as f32).collect(),
    };
    Ok(f32_to_bytes(&vals))
}

// ---------------------------------------------------------------------------
// GGUF (enables re-quantization)
// ---------------------------------------------------------------------------

/// GGUF loading uses the in-crate minimal reader, which covers everything the
/// zsfm writer emits (F32, F16, BF16, Q8_0). Foreign quantization formats
/// (Q4_K and friends) are rejected with a pointer back to the source
/// checkpoint — convert from safetensors/pickle instead of re-quantizing
/// an already-quantized file.
fn load_gguf(path: &Path) -> Result<Checkpoint> {
    load_gguf_minimal(path).with_context(|| {
        format!(
            "GGUF {} uses a quantization this reader does not support \
             (supported: F32, F16, BF16, Q8_0) — convert from the source \
             safetensors/pickle checkpoint instead",
            path.display()
        )
    })
}

fn load_gguf_minimal(path: &Path) -> Result<Checkpoint> {
    use zsfm_gguf::GGUFFile;
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut file = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
    let gguf = GGUFFile::read(&mut file)?;

    let mut tensors = Vec::with_capacity(gguf.tensors.len());
    for info in &gguf.tensors {
        // Stored dims are GGUF order (reversed row-major) — flip back.
        let shape: Vec<u64> = info.shape.iter().rev().copied().collect();
        let (dtype, data) = match info.dtype {
            zsfm_gguf::GGMLType::F32 => (SrcDtype::F32, gguf.tensor_bytes(&mut file, info)?),
            zsfm_gguf::GGMLType::F16 => (SrcDtype::F16, gguf.tensor_bytes(&mut file, info)?),
            zsfm_gguf::GGMLType::BF16 => (SrcDtype::BF16, gguf.tensor_bytes(&mut file, info)?),
            zsfm_gguf::GGMLType::Q8_0 => (
                SrcDtype::F32,
                f32_to_bytes(&gguf.tensor_f32(&mut file, info)?),
            ),
        };
        tensors.push(RawTensor {
            name: info.name.clone(),
            shape,
            dtype,
            data,
        });
    }
    Ok(Checkpoint {
        tensors,
        metadata: gguf.metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_safetensors(path: &Path) {
        // Minimal .safetensors writer: header JSON + raw LE bodies.
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = serde_json::json!({
            "a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, a_bytes.len()]},
            "b": {"dtype": "F32", "shape": [32], "data_offsets": [a_bytes.len(), a_bytes.len() + b_bytes.len()]},
        });
        let header_str = header.to_string();
        let mut file = vec![0u8; 8];
        file[..8].copy_from_slice(&(header_str.len() as u64).to_le_bytes());
        file.extend_from_slice(header_str.as_bytes());
        file.extend_from_slice(&a_bytes);
        file.extend_from_slice(&b_bytes);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn safetensors_roundtrip_and_shape_order() {
        let dir = std::env::temp_dir().join("zsfm-checkpoint-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        write_test_safetensors(&path);

        let ckpt = load_checkpoint(&[path], &LoadOptions::default()).unwrap();
        assert_eq!(ckpt.tensors.len(), 2);
        let a = ckpt.tensors.iter().find(|t| t.name == "a").unwrap();
        assert_eq!(a.shape, vec![2, 3]); // python order
        assert_eq!(a.dtype, SrcDtype::F32);
        let vals: Vec<f32> = a
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn duplicate_names_rejected() {
        let dir = std::env::temp_dir().join("zsfm-checkpoint-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dup.safetensors");
        write_test_safetensors(&path);
        let err = match load_checkpoint(&[path.clone(), path], &LoadOptions::default()) {
            Ok(_) => panic!("expected duplicate-name error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("duplicate tensor name"));
    }

    #[test]
    fn sniffs_safetensors_without_extension() {
        let dir = std::env::temp_dir().join("zsfm-checkpoint-test");
        std::fs::create_dir_all(&dir).unwrap();
        let st = dir.join("noext-src.safetensors");
        write_test_safetensors(&st);
        let noext = dir.join("noext_checkpoint");
        std::fs::copy(&st, &noext).unwrap();
        let ckpt = load_checkpoint(&[noext], &LoadOptions::default()).unwrap();
        assert_eq!(ckpt.tensors.len(), 2);
    }
}
