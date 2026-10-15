import json, subprocess, sys, math, csv, time

print("Loading data...", file=sys.stderr)
gold_rows = list(csv.DictReader(open('gold_raw.csv'), delimiter=';'))
gold_rows = [r for r in gold_rows if r.get('Close') and r['Close'] not in ('', 'N/D')]
gold_o = [float(r['Open']) for r in gold_rows]
gold_h = [float(r['High']) for r in gold_rows]
gold_l = [float(r['Low']) for r in gold_rows]
gold_c = [float(r['Close']) for r in gold_rows]
gold_v = [float(r['Volume']) for r in gold_rows]
print(f"Gold: {len(gold_c)} rows", file=sys.stderr)

btc_raw = json.load(open('btc_200d.json'))
btc_o = [float(c[1]) for c in btc_raw]; btc_h = [float(c[2]) for c in btc_raw]
btc_l = [float(c[3]) for c in btc_raw]; btc_c = [float(c[4]) for c in btc_raw]
btc_v = [float(c[5]) for c in btc_raw]
print(f"BTC: {len(btc_c)} rows", file=sys.stderr)

N_CTX, N_HOR = 32, 10

# ========== SECTION 1: BACKTEST ==========
print()
print("=" * 100)
print("=== SECTION 1: XAUUSD BACKTEST (150 rolling windows) ===")
print("=" * 100)

def run_backtest(model, close_data, n_tests=150):
    contexts, truths, ctx_lasts = [], [], []
    for w in range(n_tests):
        ctx = close_data[w:w+N_CTX]
        true = close_data[w+N_CTX:w+N_CTX+N_HOR]
        contexts.append([ctx]); truths.append(true)
        ctx_lasts.append(close_data[w+N_CTX-1])
    all_res = []
    BATCH = 50
    t0 = time.time()
    for bs in range(0, n_tests, BATCH):
        be = min(bs+BATCH, n_tests)
        req = {'context': contexts[bs:be], 'horizon': N_HOR}
        if model == 'chronos':
            req['group_ids'] = [0]
            args = ['./target/release/zsfm','chronos','infer',
                    '--gguf','gguf/chronos-f16.gguf',
                    '--config','models/amazon__chronos-2/config.json',
                    '--group-ids','0']
        else:
            args = ['./target/release/zsfm','toto','infer',
                    '--gguf','gguf/toto-22m-q8.gguf',
                    '--config','models/Datadog__Toto-2.0-22m/config.json']
        proc = subprocess.run(args, input=json.dumps(req),
                              capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            print(f"{model} err: {proc.stderr[-200:]}", file=sys.stderr); continue
        d = json.loads(proc.stdout)
        for j, ch in enumerate(d['choices']):
            p = ch['forecast']['point']
            q10 = ch['forecast']['quantiles'].get('0.10', p)
            q90 = ch['forecast']['quantiles'].get('0.90', p)
            t = truths[bs+j]; cl = ctx_lasts[bs+j]; n = len(p)
            mae = sum(abs(a-b) for a,b in zip(p,t))/n
            rmse = math.sqrt(sum((a-b)**2 for a,b in zip(p,t))/n)
            prev = cl; dirs = 0
            for a,b in zip(p,t):
                dt = 1 if b>prev else -1; dp = 1 if a>prev else -1
                if dt==dp: dirs+=1
                prev = b
            cov = sum(1 for b,lo,hi in zip(t,q10,q90) if lo<=b<=hi)/n*100
            all_res.append({'mae':mae,'dir':dirs/n*100,'cov':cov})
        print(f"  batch {bs//BATCH+1}/{(n_tests+BATCH-1)//BATCH}", file=sys.stderr)
    return all_res, time.time()-t0

ch_bt, ch_t = run_backtest('chronos', gold_c)
toto_bt, toto_t = run_backtest('toto', gold_c)

print(f"{'Metric':<22} {'Chronos-2':>15} {'Toto 22M':>15} {'Diff':>12}")
print("-" * 68)
ch_mae = sum(r['mae'] for r in ch_bt)/len(ch_bt)
toto_mae = sum(r['mae'] for r in toto_bt)/len(toto_bt)
ch_med = sorted(r['mae'] for r in ch_bt)[len(ch_bt)//2]
toto_med = sorted(r['mae'] for r in toto_bt)[len(toto_bt)//2]
ch_dir = sum(r['dir'] for r in ch_bt)/len(ch_bt)
toto_dir = sum(r['dir'] for r in toto_bt)/len(toto_bt)
ch_cov = sum(r['cov'] for r in ch_bt)/len(ch_bt)
toto_cov = sum(r['cov'] for r in toto_bt)/len(toto_bt)
print(f"{'Avg MAE ($)':<22} {ch_mae:>15,.2f} {toto_mae:>15,.2f} {toto_mae-ch_mae:>+12,.2f}")
print(f"{'Median MAE ($)':<22} {ch_med:>15,.2f} {toto_med:>15,.2f} {toto_med-ch_med:>+12,.2f}")
print(f"{'Direction (%)':<22} {ch_dir:>15.1f} {toto_dir:>15.1f} {toto_dir-ch_dir:>+12.1f}")
print(f"{'Coverage (%)':<22} {ch_cov:>15.1f} {toto_cov:>15.1f} {toto_cov-ch_cov:>+12.1f}")
print(f"{'Time (s)':<22} {ch_t:>15.1f} {toto_t:>15.1f} {toto_t-ch_t:>+12.1f}")

# ========== SECTION 2: BTC + GOLD 2 GROUPS ==========
print()
print("=" * 100)
print("=== SECTION 2: BTC + GOLD (10 variates, 2 groups) ===")
print("=" * 100)

ctx_gold = [gold_o[-32:], gold_h[-32:], gold_l[-32:], gold_c[-32:], gold_v[-32:]]
ctx_btc = [btc_o[-32:], btc_h[-32:], btc_l[-32:], btc_c[-32:], btc_v[-32:]]
ctx_all = ctx_gold + ctx_btc

req_2g = {'context': [ctx_all], 'horizon': 10, 'group_ids': [0,0,0,0,0,1,1,1,1,1]}
print("Running Chronos-2 with 2 groups...", file=sys.stderr)
proc = subprocess.run(
    ['./target/release/zsfm','chronos','infer',
     '--gguf','gguf/chronos-f16.gguf',
     '--config','models/amazon__chronos-2/config.json',
     '--group-ids','0,0,0,0,0,1,1,1,1,1'],
    input=json.dumps(req_2g), capture_output=True, text=True, timeout=900)
if proc.returncode != 0:
    print("ERROR:", proc.stderr[-300:])
else:
    d = json.loads(proc.stdout)
    v10 = d['choices'][0]['forecast']['variates']
    names = ['Open','High','Low','Close','Volume']
    print(f"{'Asset':<7} {'Variate':<7} {'D+1':>11} {'D+5':>11} {'D+10':>11}")
    print("-" * 50)
    for i, v in enumerate(v10):
        asset = 'GOLD' if i < 5 else 'BTC'
        vi = i % 5
        p = v['point']
        if names[vi] == 'Volume':
            print(f"{asset:<7} {names[vi]:<7} {p[0]:>11.0f} {p[4]:>11.0f} {p[9]:>11.0f}")
        else:
            print(f"{asset:<7} {names[vi]:<7} {p[0]:>11.2f} {p[4]:>11.2f} {p[9]:>11.2f}")

# ========== SECTION 3: BOLLINGER BANDS BREAKOUT ==========
print()
print("=" * 100)
print("=== SECTION 3: BOLLINGER BANDS BREAKOUT ===")
print("=" * 100)

req_gold = {'context': [ctx_gold], 'horizon': 10, 'group_ids': [0,0,0,0,0]}
proc_g = subprocess.run(
    ['./target/release/zsfm','chronos','infer',
     '--gguf','gguf/chronos-f16.gguf',
     '--config','models/amazon__chronos-2/config.json',
     '--group-ids','0,0,0,0,0'],
    input=json.dumps(req_gold), capture_output=True, text=True, timeout=600)
gold_fc = json.loads(proc_g.stdout)['choices'][0]['forecast']['variates']
gold_full = gold_c + gold_fc[3]['point']

def boll(d, n=20, k=2):
    up=[None]*len(d); mid=[None]*len(d); lo=[None]*len(d)
    for i in range(n-1, len(d)):
        w=d[i-n+1:i+1]; m=sum(w)/n
        sd=math.sqrt(sum((x-m)**2 for x in w)/n)
        up[i]=m+k*sd; lo[i]=m-k*sd; mid[i]=m
    return up, mid, lo

bb_up, bb_mid, bb_lo = boll(gold_full, 20, 2)
print(f"{'Day':<6} {'Close':>9} {'BB_low':>9} {'BB_mid':>9} {'BB_up':>9} {'Signal':<15}")
print("-" * 75)
for i in range(len(gold_c), len(gold_full)):
    day = i - len(gold_c) + 1
    c = gold_full[i]
    sig = '-'
    if bb_up[i] and c > bb_up[i]: sig = 'SELL (upper)'
    elif bb_lo[i] and c < bb_lo[i]: sig = 'BUY (lower)'
    print(f"D+{day:<3} {c:>9.2f} {bb_lo[i]:>9.2f} {bb_mid[i]:>9.2f} {bb_up[i]:>9.2f} {sig:<15}")

# ========== SECTION 4: RSI DIVERGENCE ==========
print()
print("=" * 100)
print("=== SECTION 4: RSI DIVERGENCE (last 20 ctx + 10 fc) ===")
print("=" * 100)

def rsi_full(d, n=14):
    deltas = [d[i]-d[i-1] for i in range(1,len(d))]
    g=[max(x,0) for x in deltas]; l=[max(-x,0) for x in deltas]
    if len(g)<n: return [None]*len(d)
    ag=sum(g[:n])/n; al=sum(l[:n])/n; out=[None]*n
    for i in range(n, len(g)):
        ag=(ag*(n-1)+g[i])/n; al=(al*(n-1)+l[i])/n
        out.append(100.0 if al==0 else 100-100/(1+ag/al))
    return [None]+out

rsi_full_v = rsi_full(gold_full, 14)
start = len(gold_c) - 20
print(f"{'Day':<8} {'Close':>9} {'RSI':>7} {'Price5':<8} {'RSI5':<8} {'Divergence':<15}")
print("-" * 65)
for i in range(start, len(gold_full)):
    if i >= len(rsi_full_v) or rsi_full_v[i] is None: continue
    tag = f"D+{i-len(gold_c)+1}" if i >= len(gold_c) else f"Day-{len(gold_c)-i}"
    if i >= start+5 and rsi_full_v[i-5] is not None:
        pu = gold_full[i] > gold_full[i-5]
        ru = rsi_full_v[i] > rsi_full_v[i-5]
        if pu and not ru: div = 'BEARISH DIV'
        elif not pu and ru: div = 'BULLISH DIV'
        else: div = '-'
    else: div = '-'
    p5 = 'UP' if i>=5 and gold_full[i]>gold_full[i-5] else 'DOWN'
    r5 = 'UP' if i>=5 and rsi_full_v[i]>rsi_full_v[i-5] else 'DOWN'
    print(f"{tag:<8} {gold_full[i]:>9.2f} {rsi_full_v[i]:>7.1f} {p5:<8} {r5:<8} {div:<15}")

# ========== SECTION 5: MACD HISTOGRAM ==========
print()
print("=" * 100)
print("=== SECTION 5: MACD HISTOGRAM (momentum) ===")
print("=" * 100)

def ema_s(d, n):
    k=2/(n+1); out=[None]*(n-1)+[sum(d[:n])/n]
    for i in range(n, len(d)): out.append(d[i]*k+out[-1]*(1-k))
    return out

def macd_s(d):
    ef=ema_s(d,12); es=ema_s(d,26)
    ml=[None if (a is None or b is None) else a-b for a,b in zip(ef,es)]
    valid=[x for x in ml if x is not None]
    if len(valid)<9: return ml,[None]*len(d),[None]*len(d)
    sv=ema_s(valid,9)
    sl=[None]*(len(ml)-len(valid))+sv
    h=[None if (a is None or b is None) else a-b for a,b in zip(ml,sl)]
    return ml,sl,h

ml,ms,mh = macd_s(gold_full)
print(f"{'Day':<8} {'Close':>9} {'MACD':>9} {'Signal':>9} {'Hist':>9} {'Momentum':<15}")
print("-" * 65)
for i in range(len(gold_c)-15, len(gold_full)):
    if i >= len(mh) or mh[i] is None: continue
    tag = f"D+{i-len(gold_c)+1}" if i >= len(gold_c) else f"Day-{len(gold_c)-i}"
    if i>=1 and mh[i-1] is not None:
        if mh[i] > mh[i-1] and mh[i] > 0: mom='Bull strong'
        elif mh[i] > mh[i-1] and mh[i] <= 0: mom='Bull weak'
        elif mh[i] < mh[i-1] and mh[i] > 0: mom='Bear weak'
        else: mom='Bear strong'
    else: mom='-'
    print(f"{tag:<8} {gold_full[i]:>9.2f} {ml[i]:>9.2f} {ms[i]:>9.2f} {mh[i]:>9.2f} {mom:<15}")

print()
print("=" * 100)
print("=== ALL SECTIONS DONE ===")
print("=" * 100)
