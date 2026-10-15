import json, subprocess, sys

with open('gold_ohlc.json') as f:
    req_data = json.load(f)

variates = req_data['context'][0]
n_var = len(variates)
names = ['Open','High','Low','Close']

# Context-ийн сүүлийн 3 өдөр
print()
print("=" * 100)
print("=== CONTEXT (сүүлийн 3 өдөр, Volume-гүй) ===")
print("=" * 100)
print('         ' + '  '.join(f'Day-{i}' for i in range(2, -1, -1)))
for i, name in enumerate(names):
    last3 = variates[i][-3:]
    print(f'{name:6s}: ' + '  '.join(f'{x:10.2f}' for x in last3))

# Run Chronos-2
print("\nRunning Chronos-2...", file=sys.stderr)
ch = subprocess.run(
    ['./target/release/zsfm', 'chronos', 'infer',
     '--gguf', 'gguf/chronos-q8.gguf',
     '--config', 'models/amazon__chronos-2/config.json',
     '--group-ids', '0,0,0,0'],
    input=json.dumps(req_data), capture_output=True, text=True, timeout=600)
if ch.returncode != 0:
    print("Chronos ERROR:", ch.stderr[-500:]); sys.exit(1)
ch_d = json.loads(ch.stdout)

# Run Toto-2
print("Running Toto-2 22M...", file=sys.stderr)
toto_req = {'context': req_data['context'], 'horizon': req_data['horizon']}
tt = subprocess.run(
    ['./target/release/zsfm', 'toto', 'infer',
     '--gguf', 'gguf/toto-22m-q8.gguf',
     '--config', 'models/Datadog__Toto-2.0-22m/config.json'],
    input=json.dumps(toto_req), capture_output=True, text=True, timeout=600)
if tt.returncode != 0:
    print("Toto ERROR:", tt.stderr[-500:]); sys.exit(1)
tt_d = json.loads(tt.stdout)

ch_v = ch_d['choices'][0]['forecast']['variates']
tt_v = tt_d['choices'][0]['forecast']['variates']
n_hor = len(ch_v[0]['point'])

# Forecast comparison
print()
print("=" * 110)
print(f"=== XAUUSD FORECAST — OHLC ONLY (4 variate, {n_hor} days) ===")
print("=" * 110)
print(f"{'Variate':<7} {'Model':<10} " + ' '.join(f'D+{i+1:>5}' for i in range(n_hor)))
print("-" * 110)
for i, name in enumerate(names):
    print(f"{name:<7} {'Chronos':<10} " + ' '.join(f'{x:>6.2f}' for x in ch_v[i]['point']))
    print(f"{'':<7} {'Toto':<10} " + ' '.join(f'{x:>6.2f}' for x in tt_v[i]['point']))
    print()

# Model agreement
print("=" * 100)
print("=== MODEL AGREEMENT (Chronos vs Toto) ===")
print("=" * 100)
print(f"{'Variate':<9} {'Avg Diff':>12} {'Max Diff':>12} {'% Diff':>10}")
print("-" * 100)
for i, name in enumerate(names):
    ch_p = ch_v[i]['point']; tt_p = tt_v[i]['point']
    diffs = [abs(a-b) for a,b in zip(ch_p, tt_p)]
    avg = sum(diffs)/len(diffs); mx = max(diffs)
    ctx_last = variates[i][-1]
    pct = avg/ctx_last*100
    print(f"{name:<9} {avg:>12.2f} {mx:>12.2f} {pct:>9.2f}%")

# OHLC logic
print()
print("=" * 100)
print("=== OHLC LOGIC CHECK ===")
print("=" * 100)
def check(variates, n):
    o = variates[0]['point']; h = variates[1]['point']
    l = variates[2]['point']; c = variates[3]['point']
    hf = [max(o[i], h[i], c[i]) for i in range(n)]
    lf = [min(o[i], l[i], c[i]) for i in range(n)]
    fails = sum(1 for i in range(n) if not (hf[i] >= o[i] and hf[i] >= c[i] and lf[i] <= o[i] and lf[i] <= c[i] and hf[i] >= lf[i]))
    return n-fails, n
ch_p, ch_t = check(ch_v, n_hor)
tt_p, tt_t = check(tt_v, n_hor)
print(f"Chronos-2: {ch_p}/{ch_t} PASS")
print(f"Toto 22M:  {tt_p}/{tt_t} PASS")

# 80% interval
print()
print("=" * 100)
print("=== 80% CONFIDENCE INTERVAL (Close) ===")
print("=" * 100)
print(f"{'Day':<5} {'Chronos-2 80%':<28} {'Toto 22M 80%':<28}")
print("-" * 100)
ch_cq = ch_v[3]['quantiles']; tt_cq = tt_v[3]['quantiles']
for i in range(n_hor):
    ch_lo = ch_cq['0.10'][i]; ch_hi = ch_cq['0.90'][i]
    tt_lo = tt_cq['0.10'][i]; tt_hi = tt_cq['0.90'][i]
    print(f"D+{i+1:<3} [{ch_lo:8.2f} .. {ch_hi:8.2f}]  [{tt_lo:8.2f} .. {tt_hi:8.2f}]")
