#!/data/data/com.termux/files/usr/bin/bash

DATA=data/btc_close_3y.json
HORIZON=24

LAST=$(jq '.[-1]' $DATA)
CTX_JSON=$(jq -c '.' $DATA)

echo "═══════════════════════════════════════════════════════════════"
echo "  3 Загварын харьцуулалт — BTC 24h"
echo "═══════════════════════════════════════════════════════════════"
echo "Сүүлийн Close: \$$LAST"
echo "Context: $(jq 'length' $DATA) утга"
echo "═══════════════════════════════════════════════════════════════"

# === 1. FlowState (F32) ===
echo ""
echo "▸ [1/3] FlowState-R1 (F32)..."
echo "{\"context\": $CTX_JSON, \"horizon\": $HORIZON}" | \
  ./target/release/zsfm flowstate infer \
  --gguf models/ibm-granite__granite-timeseries-flowstate-r1/model-f32.gguf \
  --quiet 2>/dev/null > /tmp/out_flowstate.json

if [ -s /tmp/out_flowstate.json ]; then
  FS_MED=$(jq '.choices[0].forecast.quantiles["0.50"][-1]' /tmp/out_flowstate.json 2>/dev/null)
  FS_Q10=$(jq '.choices[0].forecast.quantiles["0.10"][-1]' /tmp/out_flowstate.json 2>/dev/null)
  FS_Q90=$(jq '.choices[0].forecast.quantiles["0.90"][-1]' /tmp/out_flowstate.json 2>/dev/null)
  echo "  ✓ FlowState OK (median=\$$FS_MED)"
else
  FS_MED=""; FS_Q10=""; FS_Q90=""
  echo "  ✗ FlowState алдаа"
fi

# === 2. Chronos-2 ===
echo ""
echo "▸ [2/3] Chronos-2..."
echo "{\"context\": $CTX_JSON, \"horizon\": $HORIZON}" | \
  ./target/release/zsfm chronos infer \
  --gguf gguf/chronos-f16.gguf \
  --config models/amazon__chronos-2/config.json \
  --quiet 2>/dev/null > /tmp/out_chronos.json

if [ -s /tmp/out_chronos.json ]; then
  CH_MED=$(jq '.choices[0].forecast.quantiles["0.5"][-1] // .choices[0].forecast.quantiles["0.50"][-1]' /tmp/out_chronos.json 2>/dev/null)
  CH_Q10=$(jq '.choices[0].forecast.quantiles["0.1"][-1] // .choices[0].forecast.quantiles["0.10"][-1]' /tmp/out_chronos.json 2>/dev/null)
  CH_Q90=$(jq '.choices[0].forecast.quantiles["0.9"][-1] // .choices[0].forecast.quantiles["0.90"][-1]' /tmp/out_chronos.json 2>/dev/null)
  echo "  ✓ Chronos OK (median=\$$CH_MED)"
else
  CH_MED=""; CH_Q10=""; CH_Q90=""
  echo "  ✗ Chronos алдаа"
fi

# === 3. Toto-2 (Q8) ===
echo ""
echo "▸ [3/3] Toto-2 22M (Q8)..."
echo "{\"context\": $CTX_JSON, \"horizon\": $HORIZON}" | \
  ./target/release/zsfm toto infer \
  --gguf gguf/toto-22m-q8.gguf \
  --config models/Datadog__Toto-2.0-22m/config.json \
  --quiet 2>/dev/null > /tmp/out_toto.json

if [ -s /tmp/out_toto.json ]; then
  TT_MED=$(jq '.choices[0].forecast.quantiles["0.5"][-1] // .choices[0].forecast.quantiles["0.50"][-1]' /tmp/out_toto.json 2>/dev/null)
  TT_Q10=$(jq '.choices[0].forecast.quantiles["0.1"][-1] // .choices[0].forecast.quantiles["0.10"][-1]' /tmp/out_toto.json 2>/dev/null)
  TT_Q90=$(jq '.choices[0].forecast.quantiles["0.9"][-1] // .choices[0].forecast.quantiles["0.90"][-1]' /tmp/out_toto.json 2>/dev/null)
  echo "  ✓ Toto OK (median=\$$TT_MED)"
else
  TT_MED=""; TT_Q10=""; TT_Q90=""
  echo "  ✗ Toto алдаа"
fi

# === Харьцуулалт ===
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Үр дүн"
echo "═══════════════════════════════════════════════════════════════"
printf "%-12s | %12s | %10s | %10s | %10s\n" "Загвар" "Медиан" "Медиан%" "VaR10%" "q90%"
echo "─────────────┼──────────────┼────────────┼────────────┼───────────"

awk -v lc="$LAST" -v m="$FS_MED" -v q1="$FS_Q10" -v q9="$FS_Q90" 'BEGIN {
  if (m != "" && m != "null") printf "%-12s | %12.2f | %+9.2f%% | %+9.2f%% | %+9.2f%%\n", "FlowState", m, (m-lc)/lc*100, (q1-lc)/lc*100, (q9-lc)/lc*100
}'
awk -v lc="$LAST" -v m="$CH_MED" -v q1="$CH_Q10" -v q9="$CH_Q90" 'BEGIN {
  if (m != "" && m != "null") printf "%-12s | %12.2f | %+9.2f%% | %+9.2f%% | %+9.2f%%\n", "Chronos-2", m, (m-lc)/lc*100, (q1-lc)/lc*100, (q9-lc)/lc*100
}'
awk -v lc="$LAST" -v m="$TT_MED" -v q1="$TT_Q10" -v q9="$TT_Q90" 'BEGIN {
  if (m != "" && m != "null") printf "%-12s | %12.2f | %+9.2f%% | %+9.2f%% | %+9.2f%%\n", "Toto-2 22M", m, (m-lc)/lc*100, (q1-lc)/lc*100, (q9-lc)/lc*100
}'
echo "═══════════════════════════════════════════════════════════════"
