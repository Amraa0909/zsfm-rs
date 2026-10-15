import json, math, statistics

def load_backtest(file):
    return json.load(open(file))

def analyze(results, name):
    n = len(results)
    maes = [r['mae'] for r in results]
    rmses = [r['rmse'] for r in results]
    dirs = [r['dir'] for r in results]
    covs = [r['cov'] for r in results]

    print()
    print("=" * 70)
    print(f"=== {name} ===")
    print("=" * 70)

    # Basic metrics
    print(f"\nBasic metrics (n={n}):")
    print(f"  MAE:     mean={statistics.mean(maes):.2f}, median={statistics.median(maes):.2f}, "
          f"stdev={statistics.stdev(maes):.2f}")
    print(f"  RMSE:    mean={statistics.mean(rmses):.2f}, median={statistics.median(rmses):.2f}")
    print(f"  RMSE/MAE ratio: {statistics.mean(rmses)/statistics.mean(maes):.2f} "
          f"({'OK' if statistics.mean(rmses)/statistics.mean(maes) < 1.5 else 'OUTLIERS'})")

    # Direction
    print(f"\nDirection accuracy:")
    print(f"  Mean: {statistics.mean(dirs):.1f}%")
    print(f"  >=55%: {sum(1 for d in dirs if d >= 55)}/{n} ({sum(1 for d in dirs if d >= 55)/n*100:.0f}%)")
    print(f"  >=60%: {sum(1 for d in dirs if d >= 60)}/{n} ({sum(1 for d in dirs if d >= 60)/n*100:.0f}%)")
    print(f"  >=70%: {sum(1 for d in dirs if d >= 70)}/{n} ({sum(1 for d in dirs if d >= 70)/n*100:.0f}%)")
    print(f"  Assessment: {'USABLE' if statistics.mean(dirs) >= 55 else 'RANDOM (efficient market)'}")

    # Coverage
    print(f"\nCoverage (target 80%):")
    print(f"  Mean: {statistics.mean(covs):.1f}%")
    print(f"  Well-calibrated (70-90%): {sum(1 for c in covs if 70 <= c <= 90)}/{n}")
    print(f"  Overconfident (<70%):     {sum(1 for c in covs if c < 70)}/{n}")
    print(f"  Underconfident (>90%):    {sum(1 for c in covs if c > 90)}/{n}")
    print(f"  Assessment: {'CALIBRATED' if 75 <= statistics.mean(covs) <= 85 else 'MISCALIBRATED'}")

    # Risk metrics
    sorted_maes = sorted(maes)
    var_95 = sorted_maes[int(0.95 * n)]
    cvar_95 = sum(sorted_maes[int(0.95 * n):]) / max(1, n - int(0.95 * n))
    print(f"\nRisk metrics:")
    print(f"  VaR 95%:  ${var_95:.2f} (5% worst case)")
    print(f"  CVaR 95%: ${cvar_95:.2f} (avg of worst 5%)")
    print(f"  Max MAE:  ${max(maes):.2f} (worst window)")

    # Consistency
    win_rate = sum(1 for m in maes if m < statistics.median(maes)) / n * 100
    print(f"\nConsistency:")
    print(f"  MAE < median: {win_rate:.0f}%")
    print(f"  MAE p10-p90 range: ${sorted_maes[int(0.1*n)]:.2f} - ${sorted_maes[int(0.9*n)]:.2f}")

    return {
        'mae_mean': statistics.mean(maes),
        'mae_median': statistics.median(maes),
        'rmse_mean': statistics.mean(rmses),
        'direction': statistics.mean(dirs),
        'coverage': statistics.mean(covs),
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_mae': max(maes),
    }

# Analyze BTCUSDT backtest
print("Loading BTCUSDT backtests...")
ch_btc = load_backtest('backtest_results.json')
tt_btc = load_backtest('backtest_toto.json')

ch_stats = analyze(ch_btc, 'CHRONOS-2 (BTCUSDT)')
tt_stats = analyze(tt_btc, 'TOTO 22M (BTCUSDT)')

# Comparison
print()
print("=" * 70)
print("=== FINAL COMPARISON (BTCUSDT) ===")
print("=" * 70)
print(f"{'Metric':<22} {'Chronos-2':>15} {'Toto 22M':>15} {'Winner':>10}")
print("-" * 70)
for metric, label in [
    ('mae_mean', 'Avg MAE'),
    ('mae_median', 'Median MAE'),
    ('rmse_mean', 'Avg RMSE'),
    ('direction', 'Direction %'),
    ('coverage', 'Coverage %'),
    ('var_95', 'VaR 95%'),
    ('cvar_95', 'CVaR 95%'),
    ('max_mae', 'Max MAE'),
]:
    ch_v = ch_stats[metric]
    tt_v = tt_stats[metric]
    winner = 'Chronos' if ch_v < tt_v else 'Toto'
    if metric in ['direction', 'coverage']:
        winner = 'Chronos' if ch_v > tt_v else 'Toto'
    print(f"{label:<22} {ch_v:>15.2f} {tt_v:>15.2f} {winner:>10}")

print()
# Verdict
ch_score = sum([
    1 if ch_stats['mae_mean'] < tt_stats['mae_mean'] else 0,
    1 if ch_stats['direction'] > tt_stats['direction'] else 0,
    1 if ch_stats['coverage'] > tt_stats['coverage'] else 0,
    1 if ch_stats['cvar_95'] < tt_stats['cvar_95'] else 0,
])
tt_score = 4 - ch_score
print(f"Overall score: Chronos-2 {ch_score}/4, Toto 22M {tt_score}/4")
if ch_score > tt_score:
    print("=> Winner: CHRONOS-2")
elif tt_score > ch_score:
    print("=> Winner: TOTO 22M")
else:
    print("=> TIE")

# Confidence check
print()
print("=" * 70)
print("=== USABILITY CHECK ===")
print("=" * 70)
def check_usable(stats, name):
    print(f"\n{name}:")
    print(f"  MAE < 5% of price?     {'YES' if stats['mae_median'] < 5000 else 'NO (crypto)'} (${stats['mae_median']:.0f})")
    print(f"  Direction >= 55%?      {'YES' if stats['direction'] >= 55 else 'NO (random)'} ({stats['direction']:.1f}%)")
    print(f"  Coverage 75-85%?       {'YES' if 75 <= stats['coverage'] <= 85 else 'NO'} ({stats['coverage']:.1f}%)")
    usable = (
        stats['direction'] >= 55 and
        75 <= stats['coverage'] <= 85
    )
    print(f"  => Usable for trading? {'YES' if usable else 'NO - probabilistic only'}")

check_usable(ch_stats, 'Chronos-2')
check_usable(tt_stats, 'Toto 22M')

print()
print("=" * 70)
print("=== RECOMMENDATION ===")
print("=" * 70)
if ch_stats['direction'] < 55 and tt_stats['direction'] < 55:
    print("Direction ~50% => Use ONLY for probabilistic forecasts")
    print("  - Risk management (VaR, CVaR)")
    print("  - Volatility estimation")
    print("  - Confidence intervals")
    print("  NOT for directional trading (long/short)")
