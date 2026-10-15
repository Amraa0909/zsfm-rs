#!/usr/bin/env python3
"""
Probability analysis: Chronos-2 + Toto-2 → 10 candles
- UP/DOWN probability per day
- Probability of price reaching levels
- Cumulative probability
- Confidence intervals (50%, 80%, 95%)
"""
import json, subprocess, sys, math
from scipy import stats
import numpy as np

def run_model(model, req):
    if model == 'chronos':
        args = ['./target/release/zsfm', 'chronos', 'infer',
                '--gguf', 'gguf/chronos-q8.gguf',
                '--config', 'models/amazon__chronos-2/config.json',
                '--group-ids', ','.join(['0']*len(req['context'][0]))]
    else:
        args = ['./target/release/zsfm', 'toto', 'infer',
                '--gguf', 'gguf/toto-22m-q8.gguf',
                '--config', 'models/Datadog__Toto-2.0-22m/config.json']
    proc = subprocess.run(args, input=json.dumps(req),
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"{model}: {proc.stderr[-300:]}")
    return json.loads(proc.stdout)['choices'][0]['forecast']['variates']

def prob_from_quantiles(quantiles, level):
    """Interpolate probability of value being below/above given quantile levels"""
    q_sorted = sorted([(float(k), v) for k, v in quantiles.items()])
    return q_sorted

def pct_between(q10, q90, med, true):
    """Approximate probability using normal distribution assumption"""
    # Estimate std from 80% interval
    sigma = (q90 - q10) / (2 * 1.2816)  # z_0.90 = 1.2816
    if sigma <= 0:
        return 50.0
    # P(X > true) using normal CDF
    z = (true - med) / sigma
    return (1 - stats.norm.cdf(z)) * 100

def pct_up_down(prev_close, med, q10, q90):
    """P(UP) = P(close > prev_close)"""
    sigma = (q90 - q10) / (2 * 1.2816)
    if sigma <= 0:
        return 50.0
    z = (prev_close - med) / sigma
    p_up = (1 - stats.norm.cdf(z)) * 100  # P(X > prev)
    return p_up

# Load data
with open('gold_ohlcv.json') as f:
    req_data = json.load(f)

variates_ctx = req_data['context'][0]
n_var = len(variates_ctx)
names = ['Open', 'High', 'Low', 'Close', 'Volume'][:n_var]
n_hor = req_data['horizon']

last_close = variates_ctx[3][-1]
print(f"\nОдоогийн Close: ${last_close:.2f}")
print(f"Context: {len(variates_ctx[0])} өдөр, Horizon: {n_hor} өдөр")

# Run both models
print("\nRunning Chronos-2...", file=sys.stderr)
ch_v = run_model('chronos', req_data)
print("Running Toto-2 22M...", file=sys.stderr)
tt_v = run_model('toto', req_data)

# Get Close variate
ch_close = ch_v[3]
tt_close = tt_v[3]
ch_q = ch_close['quantiles']
tt_q = tt_close['quantiles']
ch_med = ch_close['point']
tt_med = tt_close['point']

# ============ PROBABILITY TABLE ============
print()
print("=" * 110)
print(f"=== MAGADLALYN SHINJILGEE — XAUUSD (10 өдөр) ===")
print(f"=== Суурь: Close = ${last_close:.2f} ===")
print("=" * 110)
print()

# 1. UP/DOWN probability per day
print("1. UP/DOWN МАГАДЛАЛ (өмнөх өдрийн Close-той харьцуулан)")
print("-" * 110)
print(f"{'Day':<6} {'Chronos Med':>13} {'Chronos UP%':>13} {'Toto Med':>12} {'Toto UP%':>12} {'Consensus':>12}")
print("-" * 110)

ch_probs, tt_probs = [], []
prev_ch = last_close
prev_tt = last_close
for i in range(n_hor):
    ch_p = pct_up_down(prev_ch, ch_med[i],
                       float(ch_q['0.10'][i]), float(ch_q['0.90'][i]))
    tt_p = pct_up_down(prev_tt, tt_med[i],
                       float(tt_q['0.10'][i]), float(tt_q['0.90'][i]))
    ch_probs.append(ch_p)
    tt_probs.append(tt_p)
    consensus = (ch_p + tt_p) / 2
    marker = ""
    if consensus > 60: marker = " ↑↑"
    elif consensus > 55: marker = " ↑"
    elif consensus < 40: marker = " ↓↓"
    elif consensus < 45: marker = " ↓"
    print(f"D+{i+1:<3} {ch_med[i]:>13.2f} {ch_p:>12.1f}% {tt_med[i]:>12.2f} {tt_p:>11.1f}% {consensus:>11.1f}%{marker}")
    prev_ch = ch_med[i]
    prev_tt = tt_med[i]

# 2. Confidence intervals
print()
print("2. ИТГЭЛТЭЙ ИНТЕРВАЛУУД (Chronos-2 / Toto 22M)")
print("-" * 110)
print(f"{'Day':<6} {'Chronos 80%':<32} {'Toto 80%':<32}")
print("-" * 110)
for i in range(n_hor):
    ch_lo = float(ch_q['0.10'][i]); ch_hi = float(ch_q['0.90'][i])
    tt_lo = float(tt_q['0.10'][i]); tt_hi = float(tt_q['0.90'][i])
    print(f"D+{i+1:<3} [{ch_lo:8.2f} .. {ch_hi:8.2f}]  [{tt_lo:8.2f} .. {tt_hi:8.2f}]")

print()
print("=== 95% ИНТЕРВАЛ (Chronos-2, 0.01-0.99) ===")
print("-" * 110)
print(f"{'Day':<6} {'Low (q0.01)':>14} {'Median':>12} {'High (q0.99)':>14} {'Width':>10}")
print("-" * 110)
for i in range(n_hor):
    if '0.01' in ch_q and '0.99' in ch_q:
        lo = float(ch_q['0.01'][i])
        hi = float(ch_q['0.99'][i])
        print(f"D+{i+1:<3} {lo:>14.2f} {ch_med[i]:>12.2f} {hi:>14.2f} {hi-lo:>10.2f}")

# 3. Probability of specific levels
print()
print("3. ТҮВШИН ХҮРЭХ МАГАДЛАЛ (Chronos-2, D+10)")
print("-" * 110)

d10 = n_hor - 1
ch_med_10 = ch_med[d10]
# Get quantiles for D+10
q_levels = sorted([(float(k), float(v[d10])) for k, v in ch_q.items()])
# Build CDF
print(f"Суурь Close: ${last_close:.2f}")
print(f"D+10 Median: ${ch_med_10:.2f}")
print()
print(f"{'Level':<12} {'P(Close > Level)':>20} {'P(Close < Level)':>20}")
print("-" * 60)

levels = [
    last_close - 200,
    last_close - 100,
    last_close - 50,
    last_close,
    last_close + 50,
    last_close + 100,
    last_close + 200,
    ch_med_10,
]

# Estimate sigma from 80% interval
q10_10 = float(ch_q['0.10'][d10])
q90_10 = float(ch_q['0.90'][d10])
sigma = (q90_10 - q10_10) / (2 * 1.2816)

for lvl in levels:
    z = (lvl - ch_med_10) / sigma if sigma > 0 else 0
    p_above = (1 - stats.norm.cdf(z)) * 100
    p_below = stats.norm.cdf(z) * 100
    marker = ""
    if lvl == last_close: marker = " ← одоогийн"
    elif lvl == ch_med_10: marker = " ← median"
    print(f"${lvl:<11.2f} {p_above:>19.1f}% {p_below:>19.1f}%{marker}")

# 4. Expected move
print()
print("4. ХҮЛЭЭГДЭЖ БУЙ ХӨДӨЛГӨӨН (D+10)")
print("-" * 110)
ch_move = ch_med_10 - last_close
ch_move_pct = ch_move / last_close * 100
tt_move = tt_med[d10] - last_close
tt_move_pct = tt_move / last_close * 100

print(f"{'Model':<12} {'D+10 Median':>15} {'Move ($)':>15} {'Move (%)':>15}")
print("-" * 60)
print(f"{'Chronos-2':<12} {ch_med_10:>15.2f} {ch_move:>+15.2f} {ch_move_pct:>+14.2f}%")
print(f"{'Toto 22M':<12} {tt_med[d10]:>15.2f} {tt_move:>+15.2f} {tt_move_pct:>+14.2f}%")

# 5. Cumulative probability
print()
print("5. ХУРИМТЛАЛТ МАГАДЛАЛ (Chronos-2, D+10)")
print("-" * 110)
print("P(Close_D+10 < X):")
for lvl in levels:
    z = (lvl - ch_med_10) / sigma if sigma > 0 else 0
    p = stats.norm.cdf(z) * 100
    bar = '█' * int(p / 5)
    print(f"  < ${lvl:8.2f}: {p:5.1f}%  {bar}")

# 6. Risk metrics
print()
print("6. ЭРСДЭЛИЙН ХЭМЖЭЭ")
print("-" * 110)
print(f"{'Metric':<30} {'Chronos-2':>15} {'Toto 22M':>15}")
print("-" * 60)
print(f"{'D+10 Median ($)':<30} {ch_med_10:>15.2f} {tt_med[d10]:>15.2f}")
print(f"{'D+10 q0.10 ($)':<30} {q10_10:>15.2f} {float(tt_q['0.10'][d10]):>15.2f}")
print(f"{'D+10 q0.90 ($)':<30} {q90_10:>15.2f} {float(tt_q['0.90'][d10]):>15.2f}")
print(f"{'80% width ($)':<30} {q90_10-q10_10:>15.2f} {float(tt_q['0.90'][d10])-float(tt_q['0.10'][d10]):>15.2f}")
print(f"{'80% width (%)':<30} {(q90_10-q10_10)/last_close*100:>14.2f}% {(float(tt_q['0.90'][d10])-float(tt_q['0.10'][d10]))/last_close*100:>14.2f}%")

# Save results
out = {
    'last_close': last_close,
    'chronos': {
        'median': ch_med,
        'up_prob': ch_probs,
        'quantiles': {k: list(v) for k, v in ch_q.items()},
    },
    'toto': {
        'median': tt_med,
        'up_prob': tt_probs,
        'quantiles': {k: list(v) for k, v in tt_q.items()},
    },
}
with open('probability_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: probability_results.json", file=sys.stderr)
