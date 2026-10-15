# Chronos-2 + Toto-2 Multivariate Forecasting — PROJECT.md

Огноо: 2026-10-15 | Төхөөрөмж: Helio G99, 6GB RAM, Termux (Android) | Хэл: 100% Rust | GitHub: https://github.com/Amraa0909/zsfm-rs

---

## 1. Төслийн зорилго

Amazon-ы Chronos-2 болон Datadog-ийн Toto-2 загваруудыг Rust дээр Termux-д ажиллуулж, multivariate цаг хугацааны таамаглал гаргах. Python огт ашиглахгүй.

Гол амжилтууд:
- Chronos-2 multivariate (GroupSelfAttention бүрэн хэрэгжүүлсэн)
- Toto-2 22M multivariate (bug fix)
- --group-ids флаг (бүлэг тусгаарлалт)
- 100% Rust магадлалын шинжилгээ (prob_rust)
- 10 synthetic pattern benchmark
- XAUUSD + BTCUSDT backtest (150 window)
- OHLCV логик 10/10 PASS

---

## 2. Загварууд

### Chronos-2 (Amazon)
- HuggingFace: amazon/chronos-2
- Параметр: 120M | Файл: gguf/chronos-q8.gguf (127MB, Q8)
- Config: models/amazon__chronos-2/config.json
- Quantile: 24 түвшин (0.01-0.99)
- Архитектур: Encoder-only T5 | Multivariate: GroupSelfAttention | Context: 8192

Config:
{
  "d_model": 768, "d_kv": 64, "d_ff": 3072,
  "num_layers": 12, "num_heads": 12,
  "chronos_config": {
    "context_length": 8192,
    "input_patch_size": 16,
    "input_patch_stride": 16,
    "quantiles": [0.01, 0.05, ..., 0.99]
  }
}

### Toto-2 22M (Datadog)
- HuggingFace: Datadog/Toto-2.0-22m
- Параметр: 22M | Файл: gguf/toto-22m-q8.gguf (35MB, Q8)
- Config: models/Datadog__Toto-2.0-22m/config.json
- Quantile: 9 түвшин (0.10-0.90)
- Архитектур: Decoder-only, u-μP | Multivariate: Time/variate | Context: 4096

Config:
{
  "d_model": 512, "d_ff": 1368,
  "num_layers": 6, "num_heads": 8,
  "num_groups": 8, "patch_size": 32,
  "num_output_patches": 1, "layer_group_size": 6
}

---

## 3. Кодын өөрчлөлтүүд

### 3.1 Chronos-2 Multivariate (crates/models/chronos/src/infer/mod.rs)

Шинэ функц: forecast_multivariate(series, group_ids, prediction_length)

Гол өөрчлөлтүүд:
- forward_time_attn: batch параметрийг динамик болгосон
- forward_group_attn_univariate -> forward_group_attn_full:
  - Бүрэн Q/K/V attention, Group mask, Batch хоорондын attention
- forward_block, forward_encoder: group_mask: Option<&Tensor<2>> параметр
- Хуучин forecast функц хэвээр (univariate)

### 3.2 Chronos-2 CLI (crates/zsfm/src/chronos.rs)

- run_infer_burn доторх raw_variates.len() == 1 шалгалтыг арилгасан
- n_var == 1 -> model.forecast() | n_var > 1 -> model.forecast_multivariate()
- --group-ids флаг нэмсэн
- JSON-д "group_ids": [...] дэмжинэ

### 3.3 Toto-2 Bug Fix (crates/models/toto/src/infer/mod.rs)

Алдаа: h хувьсагчийг narrow-ийн өмнө тооцоолж байсан -> index out of bounds

Хуучин (буруу):
  let h = out.dims()[2];
  let out = out.narrow(2, 0, prediction_length);

Шинэ (зөв):
  let out = out.narrow(2, 0, prediction_length);
  let h = out.dims()[2];

### 3.4 100% Rust Probability Tool (prob_rust/src/main.rs)

- normal_cdf(z) — Abramowitz & Stegun формула
- prob_above(median, q10, q90, level) — магадлал
- prob_up(prev_close, median, q10, q90) — UP магадлал
- Model дуудах: std::process::Command
- JSON: serde_json
- Binary: ~/zsfm-rs/target/release/prob_rust

---

## 4. Тестийн үр дүн

### 4.1 Synthetic Pattern Benchmark (10 patterns)

Univariate (MAE):
| Pattern | MAE | Grade |
|---------|-----|-------|
| sine_wave | 0.041 | A |
| trend_up | 0.114 | A |
| garch | 0.172 | A |
| flat | 0.185 | A |
| trend_down | 0.191 | A |
| exp_growth | 0.270 | A |
| mean_revert | 0.297 | A |
| regime_switch | 0.494 | A |
| random_walk | 2.017 | D |
| jumps | 5.110 | F |

Multivariate (5 variate OHLCV): 10/10 OHLCV логик PASS
Multivariate нь univariate-ээс 24-78% илүү нарийвчлалтай

### 4.2 BTCUSDT Backtest (150 rolling windows)

| Metric | Chronos-2 | Toto 22M | Ялагч |
|--------|-----------|----------|-------|
| Avg MAE | $2,866.97 | $2,646.84 | Toto |
| Median MAE | $1,772.68 | $1,673.32 | Toto |
| Direction | 49.7% | 49.3% | ~Тэнцүү |
| Coverage | 75.5% | 77.2% | Toto |
| VaR 95% | $8,125.71 | $8,089.19 | Toto |
| CVaR 95% | $9,987.11 | $9,603.94 | Toto |
| Max MAE | $12,638.97 | $11,990.14 | Toto |
| Хугацаа | 56.1s | 5.2s | Toto (12x) |

Дүгнэлт: Toto 22M нь 3.6x жижиг, 12x хурдан, илүү нарийвчлалтай.

### 4.3 XAUUSD Backtest (150 rolling windows)

| Metric | Chronos-2 | Toto 22M |
|--------|-----------|----------|
| Avg MAE | $6.88 | $6.97 |
| Median MAE | $5.98 | $6.59 |
| Direction | 50.6% | 49.9% |
| Coverage | 70.8% | 70.7% |

### 4.4 OHLC vs OHLCV (XAUUSD)

| Variate | OHLC (4 var) | OHLCV (5 var) | Ялгаа |
|---------|--------------|---------------|-------|
| Open | 0.59% | 0.30% | 2x сайн |
| High | 0.63% | 0.35% | 1.8x сайн |
| Low | 0.84% | 0.52% | 1.6x сайн |
| Close | 0.53% | 0.33% | 1.6x сайн |

Volume нь Chronos-2-д чиглэл өгдөг. Volume-гүй үед flat, Volume-той үед bearish.

---

## 5. Ашиглах командууд

### 5.1 Chronos-2 Univariate
echo '{"context": [[1,2,...,32]], "horizon": 10}' | ./target/release/zsfm chronos infer --gguf gguf/chronos-q8.gguf --config models/amazon__chronos-2/config.json

### 5.2 Chronos-2 Multivariate (3 variate)
echo '{"context": [[[1,2,...,32],[100,...,131],[1000,...,1310]]], "horizon": 10}' | ./target/release/zsfm chronos infer --gguf gguf/chronos-q8.gguf --config models/amazon__chronos-2/config.json --group-ids 0,0,0

### 5.3 Chronos-2 Multivariate (2 groups)
echo '{"context": [[[v1],[v2],[v3],[v4]]], "horizon": 10, "group_ids": [0,0,1,1]}' | ./target/release/zsfm chronos infer --gguf gguf/chronos-q8.gguf --config models/amazon__chronos-2/config.json

### 5.4 Toto-2 22M
echo '{"context": [[[1,2,...,32],[100,...,131]]], "horizon": 10}' | ./target/release/zsfm toto infer --gguf gguf/toto-22m-q8.gguf --config models/Datadog__Toto-2.0-22m/config.json

### 5.5 100% Rust Probability Analysis
cd ~/zsfm-rs/prob_rust && ~/zsfm-rs/target/release/prob_rust 2>&1 | tee prob.log

Гаралт: UP/DOWN магадлал, 80% интервал, түвшин хүрэх магадлал, CDF, VaR

---

## 6. Файлын бүтэц

~/zsfm-rs/
├── PROJECT.md                        <- Энэ файл
├── HANDOFF.md                        <- Дараагийн AI-д
├── README-MULTIVARIATE.md            <- Multivariate баримт
├── VALIDATION.md                     <- Backtest дүгнэлт
├── Cargo.toml
├── gguf/
│   ├── chronos-q8.gguf              (127MB, Q8)
│   └── toto-22m-q8.gguf              (35MB, Q8)
├── models/
│   ├── amazon__chronos-2/
│   │   ├── config.json
│   │   └── model-f32.gguf (455MB)
│   └── Datadog__Toto-2.0-22m/
│       ├── config.json
│       └── model-f32.gguf (88MB)
├── prob_rust/
│   ├── Cargo.toml
│   └── src/main.rs
├── target/release/
│   ├── zsfm
│   └── prob_rust
├── zsfm-rs/crates/
│   ├── models/chronos/src/infer/mod.rs
│   ├── models/toto/src/infer/mod.rs
│   └── zsfm/src/chronos.rs
├── gold_ohlcv.json                   <- XAUUSD
├── real_ohlcv.json                   <- BTCUSDT
├── btc_200d.json                     <- 200 өдөр BTC
└── gold_raw.csv                      <- 5391 мөр XAUUSD

---

## 7. Мэдэгдэж буй асуудлууд

### 7.1 "Text file busy" (os error 26)
Шалтгаан: Termux-ийн incremental compilation
Шийдэл:
  echo 'incremental = false' >> ~/.cargo/config.toml
  cargo build --release --jobs 1

### 7.2 DNS ажиллахгүй
Шалтгаан: Termux-ийн Android libc нь $PREFIX/etc/resolv.conf-ийг уншдаггүй
Шийдэл:
  ~/fix_dns.sh          # 30+ домэйн
  ~/refresh_dns.sh      # IP өөрчлөгдсөн үед

### 7.3 Toto-2 panic (index out of bounds)
Шалтгаан: h хувьсагч narrow-ийн өмнө тооцоологдсон
Шийдэл: h = out.dims()[2] мөрийг narrow-ийн дараа зөөх

### 7.4 OHLC логик зөрчигдөх
Шалтгаан: Chronos/Toto нь статистик загвар
Шийдэл:
  let h_fixed = max(o[i], h[i], c[i]);
  let l_fixed = min(o[i], l[i], c[i]);

### 7.5 prob_rust нь workspace дотор
Анхаар: Binary нь ~/zsfm-rs/target/release/prob_rust дотор (дотоод target/ биш).

---

## 8. Дараагийн AI-д зориулсан зөвлөгөө

Эхлэх алхамууд:
1. git log --oneline -10 — сүүлийн commit
2. cat README-MULTIVARIATE.md — multivariate
3. cat VALIDATION.md — backtest
4. ls gguf/ models/ — файлууд

Шинэ тест:
- XAUUSD: gold_ohlcv.json (5 variate)
- BTCUSDT: real_ohlcv.json (5 variate)
- Univariate: {"context": [[1,2,...,32]], "horizon": 10}
- Multivariate: {"context": [[[v1],[v2],[v3]]], "horizon": 10}

Алдаа гарвал:
- DNS: ~/fix_dns.sh
- Compile: cargo build --release --jobs 1
- Toto panic: h мөрийн байрлал
- JSON parse: context-ийн 3D формат

Чухал зарчмууд:
1. Python ашиглахгүй — зөвхөн Rust
2. Termux-д зориулсан — команд шууд ажиллуулж болохоор
3. Хэрэглэгч хөгжүүлэлтийн мэдлэггүй — бүх кодыг бэлэн болгох
4. Нарийн алдаа засах — panic, DNS, JSON

Загварын шинж чанарууд (чухал!):
- Chronos-2: Direction ~50%, MAE ~2%, Coverage 75-77%, 24 quantile
- Toto-2: Direction ~50%, MAE ~2%, Coverage 77%, 9 quantile
- Direction ~50% = efficient market — бүтэлгүйтэл БИШ
- Гол хэрэглээ: Probabilistic forecasting (VaR, интервал), directional trading БИШ

---

## 9. Холбоосууд

- GitHub: https://github.com/Amraa0909/zsfm-rs
- Original repo: https://github.com/amaye15/zsfm-rs
- Chronos-2: https://huggingface.co/amazon/chronos-2
- Toto-2: https://huggingface.co/Datadog/Toto-2.0-22m
- Chronos paper: https://arxiv.org/abs/2403.07815
- Toto paper: https://arxiv.org/abs/2402.01037

---

## 10. Хамгийн чухал дүгнэлт

Chronos-2 болон Toto-2 нь "шидэт бөмбөг" БИШ, "микроскоп" юм:

- "Ирээдүй ямар байх вэ" — хэлэхгүй
- "Ирээдүй ямар байх боломжтой" — хэлнэ
- "Ямар магадлалтайгаар" — хэлнэ
- "Хэр итгэлтэй" — хэлнэ

Тэдний үнэ цэнэ:
- Risk management (VaR, CVaR)
- Volatility estimation
- Confidence intervals
- Stop-loss placement
- Directional trading (long/short signals) БИШ

Direction ~50% нь efficient market hypothesis-тай нийцэж байна — онолын хязгаар, бүтэлгүйтэл биш.

---

Эцсийн шинэчлэл: 2026-10-15
Дараагийн AI: Энэ файлыг уншаад шууд үргэлжлүүлж чадна.
