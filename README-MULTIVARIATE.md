# Chronos-2 Multivariate (Rust + Termux)

## 🎯 Төслийн зорилго
Amazon-ы Chronos-2 (120M параметр) загварыг Rust дээр Termux (Android)-д ажиллуулж, multivariate цаг хугацааны таамаглал гаргах.

## 📦 Ашигласан репо
- Эх репо: https://github.com/amaye15/zsfm-rs
- Fork: https://github.com/Amraa0909/zsfm-rs
- Загвар: amazon/chronos-2 (HuggingFace)
- GGUF: gguf/chronos-f16.gguf (127MB, Q8 формат)

## 🛠️ Хийсэн өөрчлөлтүүд

### Файл 1: zsfm-rs/crates/models/chronos/src/infer/mod.rs
- forward_time_attn: batch-ийг динамик болгосон
- forward_group_attn_univariate -> forward_group_attn_full: Бүрэн Q/K/V attention, group mask, batch хоорондын attention
- forward_block, forward_encoder: group_mask параметр нэмсэн
- forecast_multivariate: Шинэ функц, олон хувьсагч хүлээж авна
- Хуучин forecast функц хэвээр (univariate)

### Файл 2: zsfm-rs/crates/zsfm/src/chronos.rs
- run_infer_burn доторх raw_variates.len() == 1 шалгалтыг арилгасан
- n_var == 1 -> model.forecast() (хуучин)
- n_var > 1 -> model.forecast_multivariate()
- Гаралт автоматаар Multivariate { variates: [...] } JSON болно

## 🏗️ Архитектур

Chronos-2 encoder layer:
  Input -> TimeSelfAttention (Llama RoPE) -> GroupSelfAttention -> FFN -> Output

- TimeSelfAttention: Цаг хугацааны тэнхлэг дагуух attention
- GroupSelfAttention: Олон хувьсагчийн хоорондын attention, group_ids-аас 2D mask
- RoPE: Llama-style rotate_half
- Attention scale = 1.0 (стандарт 1/sqrt(d) биш)

### Multivariate JSON формат
{"context": [[[v1_t0, v1_t1, ...], [v2_t0, ...], [v3_t0, ...]]], "horizon": 8}

- 3D: [batch][variate][time]
- batch=1, n_variate=3 -> 1 choice, variates: [3 forecast]
- АНХААР: 2D ([[v1], [v2], [v3]]) -> 3 choice, тус бүр univariate

## 🚀 Ажиллуулах

### Загвар татах + Q8 хөрвүүлэх
cd ~/zsfm-rs
./target/release/zsfm chronos convert --dtype q8

### Univariate тест
echo '{"context": [1,2,...,32], "horizon": 8}' | ./target/release/zsfm chronos infer --gguf gguf/chronos-f16.gguf --config models/amazon__chronos-2/config.json

### Multivariate тест (3 variate)
echo '{"context": [[[1,2,...,32],[100,101,...,131],[1000,1010,...,1310]]], "horizon": 8}' | ./target/release/zsfm chronos infer --gguf gguf/chronos-f16.gguf --config models/amazon__chronos-2/config.json

## ⚠️ Мэдэгдэж буй асуудлууд

### Асуудал 1: "Text file busy" (os error 26)
Шалтгаан: Termux-ийн incremental compilation
Шийдэл: ~/.cargo/config.toml-д incremental = false, --jobs 1 ашиглах

### Асуудал 2: DNS ажиллахгүй (curl/wget/git)
Шалтгаан: Termux-ийн Android libc нь $PREFIX/etc/resolv.conf-ийг уншдаггүй
Шийдэл: nslookup-ээс IP авч ашиглах:
IP=$(nslookup github.com 2>/dev/null | awk '/Address: / {print $2}' | grep -E '^[0-9]+\.' | head -1)
git -c http.curloptResolve="github.com:443:$IP" push origin main

### Асуудал 3: OHLC логик зөрчигдөх
Шалтгаан: Chronos-2 нь статистик загвар, OHLC дүрмийг мэдэхгүй
Шийдэл: Post-processing:
h_fixed = [max(o[i], h[i], c[i]) for i in range(len(o))]
l_fixed = [min(o[i], l[i], c[i]) for i in range(len(o))]

## 📊 Баталгаажсан үр дүн
- Univariate тест: 32 context -> 8 forecast, 24 quantile
- Multivariate тест (3 variate): variates: [3 forecast] зөв
- OHLC тест (4 variate): 6/10 raw, 10/10 post-processed
- Cross-variate attention ажиллаж байна

## 🔧 Хязгаарлалтууд
- CLI-д --group-ids флаг байхгүй
- Бүх variate ижил урттай байх ёстой
- Зөвхөн 24 quantile (0.01-0.99)

---
Огноо: 2026-10-15
Төхөөрөмж: Helio G99, 6GB RAM, Termux (Android)
Rust: 1.82+
Загвар: amazon/chronos-2 (120M параметр, Q8)
GitHub: https://github.com/Amraa0909/zsfm-rs


## 🎯 Group IDs (--group-ids)

Олон хувьсагчийг бүлэгт хувааж, зөвхөн ижил бүлгийн хувьсагчид хоорондоо attention хийхийг хязгаарлаж болно.

### CLI-д
    zsfm chronos infer --group-ids 0,0,1,1

### JSON-д
    {"context": [[[v1], [v2], [v3], [v4]]], "horizon": 8, "group_ids": [0, 0, 1, 1]}

### Дарааллын дүрэм (precedence)
1. --group-ids (CLI) — хамгийн өндөр эрх
2. group_ids (JSON) — CLI байхгүй бол
3. None — бүх variate нэг бүлэгт (анхдагч)

### Хэрэглээ
- Санхүү: OHLC бүгд нэг бүлэгт (хамааралтай)
- Олон хөрөнгө: Хувьцаа, бонд, алт тус тусдаа бүлэгт
- Sensor: Өөр өөр газар байрлах sensor-ууд тусдаа бүлэгт

### Валидаци
- --group-ids урт нь variate тоотой тэнцүү байх ёстой
- Алдаа: --group-ids / group_ids length N != variate count M

### Жишээ stderr
    Running multivariate forecast (4 variates x 32 context steps -> 8 future steps, group_ids=[0, 0, 1, 1]) ...


## 📈 OHLCV жишээ (5 variate)

### Тестийн өгөгдөл
- 48 өдрийн хиймэл OHLCV
- 32 context + 10 forecast
- group_ids = [0,0,0,0,0] (бүгд нэг бүлэгт)

### Үр дүн
- 5 variate x 10 өдөр
- OHLCV логик: 9/10 raw, 10/10 post-processed
- Volume scale: ~1M тогтвортой
- Close 80% интервал (D+1): [122.27, 126.43]

### Volume-ийн онцлог
- OHLC (~100) vs Volume (~1M) — 10,000 дахин ялгаа
- Instance normalization нь variate тус бүрд тусдаа хийгддэг тул асуудалгүй
- Post-processing: Volume > 0 (max(0, x))


## 📈 Жинхэнэ BTCUSDT тест (Binance API)

### Өгөгдөл татах
Binance API нь API key шаарддаггүй:
    curl -s 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=32' -o btc_raw.json

### Формат
Binance kline: [open_time, open, high, low, close, volume, close_time, ...]
Chronos формат: 5 variate = [Open, High, Low, Close, Volume]

### Үр дүн (жишээ)
- OHLCV логик: 10/10 PASS
- 80% интервал D+1 (Close): [79863 .. 84497], spread 4634
- 80% интервал D+10 (Close): [75559 .. 86469], spread 10910
- Spread цаг хугацаанд өсөж байна (зөв зан төлөв)

### Шаардлагатай домэйнууд (hosts файлд)
- api.binance.com
- api1.binance.com
- api2.binance.com


## 🧪 Synthetic Pattern Benchmark (10 patterns)

### Тестийн дизайн
- 10 өөр зах зээлийн хэв маяг (32 context + 8 forecast)
- Univariate: 1 variate (Close)
- Multivariate: 5 variate (O, H, L, C, V) OHLCV

### Univariate үр дүн
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

### Multivariate үр дүн (Close variate)
| Pattern | Uni MAE | MV MAE | Сайжрал |
|---------|---------|--------|---------|
| trend_up | 0.114 | 0.083 | 27% |
| trend_down | 0.191 | 0.114 | 40% |
| sine_wave | 0.041 | 0.031 | 24% |
| mean_revert | 0.297 | 0.185 | 38% |
| garch | 0.172 | 0.122 | 29% |
| flat | 0.185 | 0.085 | 54% |
| exp_growth | 0.270 | 0.140 | 48% |
| regime_switch | 0.494 | 0.108 | 78% |

### Гол дүгнэлт
- OHLCV логик: 10/10 PASS (multivariate)
- Multivariate нь univariate-ээс 24-78% илүү нарийвчлалтай
- Cross-variate attention бүрэн ажиллаж байна
- Random walk, jumps нь таамаглах боломжгүй (хэвийн)

### Chronos-2-ын хязгаарлалтууд
- Structural breaks (jumps) - mean reversion гэж үздэг
- Random walk - coverage 0%
- Volume - 20-30% MAE (OHLC-ээс илүү дуу чимээтэй)


## 📊 Backtest: BTCUSDT (150 rolling windows)

### Арга
- Binance-ээс 200 өдрийн BTCUSDT (daily)
- 32 өдрийн context → 8 өдрийн forecast
- 1 өдрөөр шилжиж, 150 window
- Univariate (Close)

### Үр дүн: Chronos-2 vs Toto 2.0 22M

| Metric | Chronos-2 (120M) | Toto 22M | Ялгаа |
|--------|------------------|----------|-------|
| Average MAE | ,866.97 | **,646.84** | -7.7% |
| Average RMSE | ,279.89 | **,013.66** | -8.1% |
| Median MAE | ,779 | **,698** | -4.5% |
| Direction | 49.7% | 49.3% | ~0 |
| Coverage | 75.5% | **77.2%** | +1.7% |
| q0.10 breach | 10.8% | **10.0%** | Төгс |
| q0.90 breach | 13.7% | **12.8%** | Сайжсан |
| Хугацаа (150 win) | 56.1s | **5.2s** | 12× хурдан |
| Файл | 127 MB | **35 MB** | 3.6× жижиг |

### Гол дүгнэлт
- **Toto 22M нь 5.5× бага параметртэй ч илүү нарийвчлалтай**
- **Direction ~50%** — BTC-ийн чиглэл санамсаргүй (efficient market)
- **Quantile calibration** — Toto илүү сайн (10.0% vs 10.8%)
- **Pareto improvement** — Toto бүх хэмжүүрээр илүү сайн

### Хоёулаа ижил сул талтай
- Direction ~50% (таамаглах боломжгүй)
- Worst windows W109-W111 (0k-2k MAE)
- Coverage >90% (underconfident) ~51%

### Хэрэглээ
- ✅ **Probabilistic forecast** (quantile интервал)
- ✅ **Risk management** (VaR)
- ✅ **Volatility estimation**
- ❌ **Directional trading** (long/short signal)


## 🥇 XAUUSD (Gold) — Chronos-2 vs Toto 2.0 22M

### Өгөгдөл
- Эх сурвалж: GitHub (XAUUSD-Gold-Price, 2004-2025)
- 5392 өдрийн OHLCV
- Сүүлийн 32 өдрийн context

### Model Agreement (10 өдөр)
| Variate | Avg Diff | % Diff |
|---------|----------|--------|
| Open | $10.04 | 0.30% |
| High | $11.96 | 0.35% |
| Low | $17.53 | 0.52% |
| Close | $11.07 | 0.33% |
| Volume | 24,071 | 19.92% |

### OHLCV Logic
- Chronos-2: 10/10 PASS
- Toto 22M: 10/10 PASS

### Forecast (D+1)
- Chronos-2 Close: $3,354.85
- Toto 22M Close: $3,350.46
- Хоёулаа бууралт (context $3,368.94)

### 80% интервал (Close)
| Day | Chronos-2 | Toto 22M |
|-----|-----------|----------|
| D+1 | [3287..3422] | [3281..3404] |
| D+10 | [3206..3456] | [3219..3423] |

### Гол дүгнэлт
- **0.30-0.52% зөрүү** — гайхалтай нийцэл
- Хоёулаа бууралт таамаглаж байна (1.2-1.3%)
- Toto интервал 10-20% нарийссан
- Volume 20% зөрүү (хэвийн, дуу чимээтэй)
