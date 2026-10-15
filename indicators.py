import json, subprocess, sys, math

with open('gold_ohlcv.json') as f:
    req_data = json.load(f)
variates_ctx = req_data['context'][0]
n_var = len(variates_ctx)
names = ['Open','High','Low','Close','Volume'][:n_var]

req = {'context': [variates_ctx], 'horizon': 10, 'group_ids': [0]*n_var}
print("Running Chronos-2...", file=sys.stderr)
ch_proc = subprocess.run(
    ['./target/release/zsfm', 'chronos', 'infer',
     '--gguf', 'gguf/chronos-f16.gguf',
     '--config', 'models/amazon__chronos-2/config.json',
     '--group-ids', ','.join(['0']*n_var)],
    input=json.dumps(req), capture_output=True, text=True, timeout=600
)
ch_data = json.loads(ch_proc.stdout)

print("Running Toto-2...", file=sys.stderr)
toto_proc = subprocess.run(
    ['./target/release/zsfm', 'toto', 'infer',
     '--gguf', 'gguf/toto-22m-q8.gguf',
     '--config', 'models/Datadog__Toto-2.0-22m/config.json'],
    input=json.dumps({'context': [variates_ctx], 'horizon': 10}),
    capture_output=True, text=True, timeout=600
)
toto_data = json.loads(toto_proc.stdout)

def build_series(ctx, fc):
    return {
        'O': ctx[0] + fc[0]['point'],
        'H': ctx[1] + fc[1]['point'],
        'L': ctx[2] + fc[2]['point'],
        'C': ctx[3] + fc[3]['point'],
        'V': (ctx[4] if n_var >= 5 else [0]*len(ctx[0])) + (fc[4]['point'] if n_var >= 5 else [0]*len(fc[0]['point'])),
    }

ch_series = build_series(variates_ctx, ch_data['choices'][0]['forecast']['variates'])
toto_series = build_series(variates_ctx, toto_data['choices'][0]['forecast']['variates'])
N_CTX = len(variates_ctx[0])

def sma(d, n): return [None]*n + [sum(d[i-n+1:i+1])/n for i in range(n-1, len(d))]
def ema(d, n):
    k = 2/(n+1); out = [None]*(n-1) + [sum(d[:n])/n]
    for i in range(n, len(d)): out.append(d[i]*k + out[-1]*(1-k))
    return out
def rsi(d, n=14):
    deltas = [d[i]-d[i-1] for i in range(1, len(d))]
    g = [max(x,0) for x in deltas]; l = [max(-x,0) for x in deltas]
    if len(g) < n: return [None]*len(d)
    ag = sum(g[:n])/n; al = sum(l[:n])/n; out = [None]*n
    for i in range(n, len(g)):
        ag = (ag*(n-1)+g[i])/n; al = (al*(n-1)+l[i])/n
        out.append(100.0 if al==0 else 100-100/(1+ag/al))
    return [None] + out
def macd(d):
    ef = ema(d, 12); es = ema(d, 26)
    ml = [None if (a is None or b is None) else a-b for a,b in zip(ef,es)]
    valid = [x for x in ml if x is not None]
    if len(valid) < 9: return ml, [None]*len(d), [None]*len(d)
    sl = [None]*(len(ml)-len(valid)) + ema(valid, 9)
    hist = [None if (a is None or b is None) else a-b for a,b in zip(ml, sl)]
    return ml, sl, hist
def boll(d, n=20, k=2):
    mid = sma(d, n); up=[None]*len(d); lo=[None]*len(d)
    for i in range(n-1, len(d)):
        w = d[i-n+1:i+1]; m = sum(w)/n
        sd = math.sqrt(sum((x-m)**2 for x in w)/n)
        up[i] = m+k*sd; lo[i] = m-k*sd
    return up, mid, lo
def atr(h, l, c, n=14):
    tr = [h[0]-l[0]] + [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
    return sma(tr, n)
def stoch(h, l, c, n=14, d=3):
    kl = [None]*(n-1)
    for i in range(n-1, len(c)):
        hh = max(h[i-n+1:i+1]); ll = min(l[i-n+1:i+1])
        kl.append(100*(c[i]-ll)/(hh-ll) if hh != ll else 50)
    dl = [None]*(len(kl)-d) + [sum(kl[i-d+1:i+1])/d for i in range(len(kl)-d, len(kl))] if len(kl) >= d else [None]*len(kl)
    return kl, dl

def detect_patterns(s):
    O,H,L,C = s['O'],s['H'],s['L'],s['C']
    p = [None]*len(C)
    for i in range(1, len(C)):
        body = abs(C[i]-O[i]); rng = H[i]-L[i]
        if rng == 0: p[i] = 'Flat'; continue
        us = H[i]-max(O[i],C[i]); ls = min(O[i],C[i])-L[i]
        bp = body/rng
        if bp < 0.1: p[i] = 'Doji'
        elif ls > 2*body and us < body and body > 0: p[i] = 'Hammer'
        elif us > 2*body and ls < body and body > 0: p[i] = 'ShootingStar'
        elif C[i] > O[i] and C[i-1] < O[i-1] and C[i] > O[i-1] and O[i] < C[i-1]: p[i] = 'BullEngulf'
        elif C[i] < O[i] and C[i-1] > O[i-1] and C[i] < O[i-1] and O[i] > C[i-1]: p[i] = 'BearEngulf'
        elif bp > 0.9: p[i] = 'Marubozu'
        elif bp < 0.3: p[i] = 'SpinTop'
        else: p[i] = '-'
    for i in range(2, len(C)):
        if all(C[j] > O[j] for j in range(i-2, i+1)) and all(C[j] > C[j-1] for j in range(i-1, i+1)): p[i] = '3WhiteSold'
        elif all(C[j] < O[j] for j in range(i-2, i+1)) and all(C[j] < C[j-1] for j in range(i-1, i+1)): p[i] = '3BlackCrows'
    return p

def compute(s):
    c,h,l = s['C'],s['H'],s['L']
    return {
        'RSI': rsi(c), 'SMA5': sma(c,5), 'SMA20': sma(c,20),
        'BB_up': boll(c)[0], 'BB_mid': boll(c)[1], 'BB_low': boll(c)[2],
        'MACD': macd(c)[0], 'MACD_sig': macd(c)[1],
        'ATR': atr(h,l,c), 'Stoch_K': stoch(h,l,c)[0],
    }

ch_ind = compute(ch_series); toto_ind = compute(toto_series)
ch_pat = detect_patterns(ch_series); toto_pat = detect_patterns(toto_series)

def fmt(x, d=2): return f'{x:.{d}f}' if x is not None else '-'

for model, s, ind, pat in [('CHRONOS-2', ch_series, ch_ind, ch_pat), ('TOTO 2.0 22M', toto_series, toto_ind, toto_pat)]:
    print()
    print("=" * 100)
    print(f"=== {model} ===")
    print("=" * 100)
    print(f"{'Day':<6} {'Close':>9} {'RSI':>7} {'SMA5':>9} {'SMA20':>9} {'MACD':>9} {'MACDsig':>9} {'ATR':>8} {'StochK':>8} {'Pattern':<15}")
    print("-" * 100)
    for i in range(N_CTX, N_CTX + 10):
        print(f"D+{i-N_CTX+1:<3} {s['C'][i]:>9.2f} {fmt(ind['RSI'][i],1):>7} {fmt(ind['SMA5'][i]):>9} {fmt(ind['SMA20'][i]):>9} "
              f"{fmt(ind['MACD'][i]):>9} {fmt(ind['MACD_sig'][i]):>9} {fmt(ind['ATR'][i]):>8} {fmt(ind['Stoch_K'][i],1):>8} {(pat[i] or '-'):<15}")

print()
print("=" * 100)
print("=== PATTERN AGREEMENT (Chronos vs Toto) ===")
print("=" * 100)
print(f"{'Day':<6} {'Chronos-2':<18} {'Toto 22M':<18} {'Agree':<6}")
print("-" * 100)
for i in range(N_CTX, N_CTX + 10):
    cp = ch_pat[i] or '-'; tp = toto_pat[i] or '-'
    print(f"D+{i-N_CTX+1:<3} {cp:<18} {tp:<18} {'YES' if cp == tp else 'no':<6}")

print()
print("=" * 100)
print("=== SIGNAL SUMMARY ===")
print("=" * 100)
for model, s, ind in [('Chronos-2', ch_series, ch_ind), ('Toto 22M', toto_series, toto_ind)]:
    li = len(s['C'])-1
    r = ind['RSI'][li]; m = ind['MACD'][li]; ms = ind['MACD_sig'][li]
    sk = ind['Stoch_K'][li]; bu = ind['BB_up'][li]; bl = ind['BB_low'][li]
    c = s['C'][li]
    print(f"\n{model}:")
    if r is not None:
        print(f"  RSI: {r:.1f} {'(Overbought)' if r>70 else '(Oversold)' if r<30 else '(Neutral)'}")
    if m is not None and ms is not None:
        print(f"  MACD: {'Bullish' if m>ms else 'Bearish'} cross ({m-ms:+.2f})")
    if sk is not None:
        print(f"  Stoch: {sk:.1f} {'(Overbought)' if sk>80 else '(Oversold)' if sk<20 else '(Neutral)'}")
    if bu is not None and bl is not None:
        pos = 'Above upper' if c>bu else 'Below lower' if c<bl else 'Inside'
        print(f"  BB: {pos} (close={c:.2f}, [{bl:.2f}..{bu:.2f}])")
