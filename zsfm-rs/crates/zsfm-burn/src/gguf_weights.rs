//! Direct GGUF weight loading for Burn inference — no candle.
//!
//! Reads F32/F16/BF16/Q8_0 tensors with the in-workspace `zsfm_gguf`
//! reader, decodes everything to F32 host bytes, and builds Burn tensors.
//! Replaces the migration-era bridge that went through candle's GGUF
//! reader (`burn_weight_2d` and friends now delegate to this).

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{Device, Tensor, TensorData};
use zsfm_checkpoint::cast::{decode_to_f32, f32_to_bytes, SrcDtype};
use zsfm_gguf::{GGMLType, GGUFFile, GGUFMetaValue};

/// All F32-decoded tensors of one GGUF file, keyed by tensor name.
pub struct GgufWeights {
    tensors: HashMap<String, (Vec<u64>, Vec<u8>)>,
    meta: Vec<(String, GGUFMetaValue)>,
}

fn device() -> Device {
    Device::flex()
}

/// Open `path` and decode every tensor to F32.
pub fn open_gguf_weights(path: &Path) -> Result<GgufWeights> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
    let gguf = GGUFFile::read(&mut reader).context("parse GGUF header")?;
    let mut tensors = HashMap::with_capacity(gguf.tensors.len());
    for info in &gguf.tensors {
        // Stored dims are GGUF order (reversed row-major) — flip back.
        let shape: Vec<u64> = info.shape.iter().rev().copied().collect();
        let bytes = match info.dtype {
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 => {
                gguf.tensor_bytes(&mut reader, info)?
            }
            GGMLType::Q8_0 => {
                let vals = gguf.tensor_f32(&mut reader, info)?;
                f32_to_bytes(&vals)
            }
        };
        // Validate element count before accepting.
        let n: usize = shape.iter().product::<u64>() as usize;
        let src = match info.dtype {
            GGMLType::F32 => SrcDtype::F32,
            GGMLType::F16 => SrcDtype::F16,
            GGMLType::BF16 => SrcDtype::BF16,
            GGMLType::Q8_0 => SrcDtype::F32,
        };
        let expect = n * src.bytes_per_elem();
        anyhow::ensure!(
            bytes.len() == expect,
            "tensor {}: {} bytes, want {expect}",
            info.name,
            bytes.len()
        );
        tensors.insert(info.name.clone(), (shape, bytes));
    }
    Ok(GgufWeights {
        tensors,
        meta: gguf.metadata,
    })
}

impl GgufWeights {
    fn get(&self, name: &str) -> Result<&(Vec<u64>, Vec<u8>)> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor '{name}' not found in GGUF"))
    }

    fn f32_vec(&self, name: &str) -> Result<Vec<f32>> {
        let (shape, bytes) = self.get(name)?;
        let n: usize = shape.iter().product::<u64>() as usize;
        let mut vals = decode_to_f32(bytes, SrcDtype::F32)?;
        anyhow::ensure!(
            vals.len() == n,
            "tensor {name}: decoded {} values, want {n}",
            vals.len()
        );
        // decode_to_f32 returns exactly n values; keep the binding for clarity.
        let _ = n;
        Ok(std::mem::take(&mut vals))
    }

    /// U32 metadata value (accepts u32/u64/i32/i64 spellings).
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        use GGUFMetaValue::*;
        self.meta
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| match v {
                Uint32(x) => Some(*x),
                Uint64(x) => u32::try_from(*x).ok(),
                Int32(x) => u32::try_from(*x).ok(),
                Int64(x) => u32::try_from(*x).ok(),
                _ => None,
            })
    }

    /// Load a 2D `[rows, cols]` F32 tensor.
    pub fn load_2d(&self, name: &str) -> Result<Tensor<2>> {
        let (shape, _) = self.get(name)?;
        anyhow::ensure!(shape.len() == 2, "tensor {name}: want 2D, got {shape:?}");
        let data = self.f32_vec(name)?;
        Ok(Tensor::<2>::from_data(
            TensorData::new(data, [shape[0] as usize, shape[1] as usize]),
            &device(),
        ))
    }

    /// Load a 2D weight with the `load_weight` transpose guard: PyTorch stores
    /// `(d_out, d_in)`; if the file holds it transposed (the Q8_0 transpose
    /// trick some converters apply), flip it back.
    pub fn load_weight_2d(&self, name: &str, expected_d_out: usize) -> Result<Tensor<2>> {
        let (shape, _) = self.get(name)?;
        anyhow::ensure!(shape.len() == 2, "tensor {name}: want 2D, got {shape:?}");
        let (r, c) = (shape[0] as usize, shape[1] as usize);
        let data = self.f32_vec(name)?;
        if r == expected_d_out {
            return Ok(Tensor::<2>::from_data(
                TensorData::new(data, [r, c]),
                &device(),
            ));
        }
        anyhow::ensure!(
            c == expected_d_out,
            "tensor {name}: dim(0) is {r}, want {expected_d_out} (transpose fallback needs {c})"
        );
        let mut t = vec![0.0f32; data.len()];
        for i in 0..r {
            for j in 0..c {
                t[j * r + i] = data[i * c + j];
            }
        }
        Ok(Tensor::<2>::from_data(
            TensorData::new(t, [c, r]),
            &device(),
        ))
    }

    /// Load a 1D `[n]` F32 tensor.
    pub fn load_1d(&self, name: &str) -> Result<Tensor<1>> {
        let (shape, _) = self.get(name)?;
        anyhow::ensure!(shape.len() == 1, "tensor {name}: want 1D, got {shape:?}");
        let data = self.f32_vec(name)?;
        Ok(Tensor::<1>::from_data(
            TensorData::new(data, [shape[0] as usize]),
            &device(),
        ))
    }

    /// Load a 3D `[d0, d1, d2]` F32 tensor.
    pub fn load_3d(&self, name: &str) -> Result<Tensor<3>> {
        let (shape, _) = self.get(name)?;
        anyhow::ensure!(shape.len() == 3, "tensor {name}: want 3D, got {shape:?}");
        let data = self.f32_vec(name)?;
        Ok(Tensor::<3>::from_data(
            TensorData::new(
                data,
                [shape[0] as usize, shape[1] as usize, shape[2] as usize],
            ),
            &device(),
        ))
    }

    /// Raw F32 values of any tensor (for embedding tables, rope vectors, …).
    pub fn load_flat(&self, name: &str) -> Result<Vec<f32>> {
        self.f32_vec(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_tensor_errors() {
        let w = GgufWeights {
            tensors: HashMap::new(),
            meta: Vec::new(),
        };
        assert!(w.load_2d("nope").is_err());
    }
}
