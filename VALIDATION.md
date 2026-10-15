# Chronos-2 vs Toto-2 Validation (BTCUSDT, 150 windows)

## Results

| Metric | Chronos-2 | Toto 22M | Winner |
|--------|-----------|----------|--------|
| Avg MAE | $2,866.97 | $2,646.84 | Toto |
| Median MAE | $1,772.68 | $1,673.32 | Toto |
| Direction | 49.7% | 49.3% | ~Tie |
| Coverage | 75.5% | 77.2% | Toto |
| VaR 95% | $8,125.71 | $8,089.19 | Toto |
| CVaR 95% | $9,987.11 | $9,603.94 | Toto |
| Max MAE | $12,638.97 | $11,990.14 | Toto |

## Key Findings

### ❌ NOT usable for directional trading
- Direction ~50% (efficient market hypothesis)
- Only 34-35% of windows achieve 55%+ direction
- Random walk behavior confirmed

### ✅ Usable for probabilistic forecasts
- MAE ~2% of price (excellent)
- Coverage 75-77% (well-calibrated)
- VaR/CVaR well-defined

### 🏆 Toto 22M wins 3/4 metrics
- 3.6× smaller file (35 MB vs 127 MB)
- 12× faster (5.2s vs 56.1s per 150 windows)
- Better MAE, Coverage, CVaR

## Practical Use Cases

1. **Position sizing** (VaR-based)
2. **Stop-loss placement** (q0.10)
3. **Options strategies** (straddle, protective puts)
4. **Risk management** (CVaR)
5. **Volatility estimation**

## NOT recommended for
- Directional trading (long/short signals)
- Trend following
- Breakout strategies
- Entry/exit timing
