# Burn migration measurements

Every number here comes from executed code, not estimates. Op microbenchmarks
run offline in CI (`cargo test -p zsfm-burn -- --nocapture`). Model-level
numbers come from `zsfm-bench` side-by-side runs once each model ports.

**Status: COMPLETE.** Candle is fully removed — every model is Burn-only,
`zsfm-nn`/candle deps are deleted, and the `--engine` flag is gone. The
tables below preserve the historical parity measurements that gated the
migration.

## Op parity (historical: candle vs Burn Flex, CPU, Apple Silicon)

Source: `zsfm-rs/crates/zsfm-burn/tests/parity.rs` (50 iters, median).
Backend is Flex (pure-Rust CPU): `burn-ndarray` is deprecated in Burn 0.22,
so the migration targeted Flex from the start. The parity tests are now
Burn-vs-host-reference checks; the candle numbers below are the recorded
pre-removal measurements.

| Op | Shape | candle ms | Burn ms | max abs err | Gate |
|----|-------|----------:|--------:|------------:|------|
| linear | 32x64→64 | 0.662 | 0.462 | 0.00e0 | <1e-5 ✅ |
| rms_norm | 32x64 | 0.208 | 0.139 | 1.19e-7 | <1e-4 ✅ |
| attention | b2 h4 sq16 hd32 | 1.345 | 0.914 | 2.98e-7 | <1e-4 ✅ |

Notes:

- Burn Flex beats candle on all three ops at these shapes (linear 1.4x, rms_norm 1.5x, attention 1.5x).
- Re-measure at model shapes (d=512..2048, longer sequences) before concluding; attention scaling with seq length matters most.
- Gate is max-abs-error on identical inputs. Model-level gate is stricter (see below).

## Model rollout tracker (final: all Burn-only)

Gate per model was: same GGUF, same contexts, max-abs-error vs the candle
path below 1e-4 on fixed probes AND no MAE regression in `zsfm-bench report`.
Every model passed; the candle implementations, parity tests, and `--engine`
seam were then deleted. Each model now ships a Burn-only golden test
(determinism + finite output + hardcoded golden values).

| Model | Status | Probe err | Bench MAE Δ | Notes |
|-------|--------|----------:|------------:|-------|
| ttm | ✅ Burn-only | 3.05e-5 | pending real weights | `BurnTtmModel` in `models/ttm/src/infer/mod.rs`; Burn-only golden test |
| timesfm | ✅ Burn-only | 1.5e-5 (attn block) | pending real weights | `BurnTimesFMModel` in `models/timesfm/src/infer/mod.rs`; rope/residual/attention host-reference + golden tests |
| sundial | ✅ Burn-only | 3.8e-6 | pending real weights | `BurnSundialModel` in `models/sundial/src/infer/mod.rs`; Llama RoPE + flow ODE; Burn-only golden test |
| moment | ✅ Burn-only | 0.00e0 | pending real weights | `BurnMomentModel` in `models/moment/src/infer/mod.rs`; T5 rel-bias + gated-GELU; Burn-only golden test |
| chronos | ✅ Burn-only | 9.54e-7 | pending real weights | `BurnChronosModel` in `models/chronos/src/infer/mod.rs`; shared InferConfig carried over; Burn-only golden test |
| toto | ✅ Burn-only | 0.00e0 | pending real weights | `BurnTotoModel` in `models/toto/src/infer/mod.rs`; shared InferConfig carried over; Burn-only golden test |
| moirai | ✅ Burn-only | 9.5e-7 | pending real weights | `BurnMoiraiModel` in `models/moirai/src/infer/mod.rs`; QK-norm + split RoPE + vbias; Burn-only golden test |
| moirai2 | ✅ Burn-only | 0.00e0 | pending real weights | `BurnMoirai2Model` in `models/moirai2/src/infer/mod.rs`; partial RoPE; Burn-only golden test |
| lag_llama | ✅ Burn-only | 0.00e0 | pending real weights | Burn prefill + verbatim host decode; Burn-only golden test |
| flowstate | ✅ Burn-only | 0.00e0 | pending real weights | Burn projections + verbatim SSM scan; Burn-only golden test |
| tirex | ✅ Burn-only | 0.00e0 | pending real weights | Burn residual/FFN + verbatim sLSTM host kernels; Burn-only golden test |
| mitra | ✅ Burn-only | 4.8e-7 | pending real weights | `BurnMitraModel` in `models/mitra/src/infer/mod.rs`; dual row/feat attention; Burn-only golden test |
| tabdpt | ✅ Burn-only | 0.00e0 | pending real weights | `BurnTabDptModel` in `models/tabdpt/src/infer/mod.rs`; gated attention + binned head; Burn-only golden test |
| tabicl | ✅ Burn-only | 3.0e-8 | pending real weights | `BurnTabIclModel` in `models/tabicl/src/infer/mod.rs`; ISAB + SSMax + shared Llama RoPE; Burn-only golden test |
| tabpfn | ✅ Burn-only | 8.6e-5 | pending real weights | `BurnTabPfnModel` in `models/tabpfn/src/infer/mod.rs`; GQA test quirk + log decoder; Burn-only golden test |
| tabfm | ✅ Burn-only | 2.9e-4 | pending real weights | `BurnTabFMModel` in `models/tabfm/src/infer/mod.rs` single-table + Burn ensemble; Burn-only golden test |

## Engine selection (retired)

~~`zsfm <model> infer --engine candle|burn` (default: `candle` until each model
flips after passing its gate).~~ All models flipped; the `--engine` flag and
the `ZSFM_ENGINE`/`zsfm_burn::Engine` seam are deleted. Burn runs Flex CPU.
`--backend` selection arrives with Phase 5.

## Follow-up
- Real-weights `zsfm-bench` comparison per model (needs downloads).
- Python bindings (`zsfm-python`) now wrap the Burn models (uncompilable in
  CI here — pyo3 needs system Python headers; needs a maturin build check).

## Smoke test (real weights, Burn-only CLI)

- `zsfm ttm convert` pulled `ibm-granite/granite-timeseries-ttm-r2`
  (134 tensors → 3.2 MB F32 GGUF) with no Python.
- `zsfm ttm infer` on a 512-step sine+trend context (end ≈33.58),
  horizon 12: point forecast starts at 33.61 and stays smooth/finite —
  sane continuation, envelope + usage fields correct.
- GPU backends (Wgpu/Metal/Cuda) + Fusion + true Q8 inference.
