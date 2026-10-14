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
