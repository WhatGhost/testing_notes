import re, ast, itertools
rows=[]
for line in open('/tmp/extend_eval.log'):
    m=re.match(r"g(\d+) hkv(\d+) bs(\d+) ext(\d+) pre(\d+) (\{.*\}) best", line)
    if not m: continue
    g,hkv,bs,ext,pre=map(int,m.groups()[:5]); d=ast.literal_eval(m.group(6))
    rows.append((g,hkv,bs,ext,pre,{ast.literal_eval(k)[0]:v for k,v in d.items()}))
def cdiv(a,b): return -(-a//b)
def p(g,hkv,bs,ext,bm): return bs*hkv*cdiv(ext, bm//g)
def evaluate(choose, show=False):
    worst=1; tot=0; bad=[]
    for g,hkv,bs,ext,pre,d in rows:
        b=choose(g,hkv,bs,ext,pre)
        r=d[b]/min(d.values()); tot+=r
        worst=max(worst,r)
        if r>1.12: bad.append((g,hkv,bs,ext,pre,b,round(r,2)))
    if show:
        for x in bad: print("   ", x)
    return tot/len(rows), worst, len(bad)
best=[]
# A: thresholds on programs(128), separate for prefix / no prefix
for tp, tn in itertools.product([8,16,24,32,48,64],[48,64,96,128,192,256]):
    f=lambda g,hkv,bs,ext,pre: 128 if p(g,hkv,bs,ext,128) >= (tp if pre else tn) else 16
    best.append((evaluate(f), "A", tp, tn))
# B: total KV work of small tiles: programs16 * (pre + ext/2) >= X
for X in [2**k for k in range(16,24)]:
    for xs in (1, 1.5, 2, 3):
        X2 = int(X*xs)
        f=lambda g,hkv,bs,ext,pre,X2=X2: 128 if p(g,hkv,bs,ext,16)*(pre+ext//2) >= X2 else 16
        best.append((evaluate(f), "B", X2))
# C: programs128 * (pre + ext) >= Y
for Y in [2**k for k in range(12,22)]:
    for ys in (1,1.5,2,3):
        Y2=int(Y*ys)
        f=lambda g,hkv,bs,ext,pre,Y2=Y2: 128 if p(g,hkv,bs,ext,128)*(pre+ext) >= Y2 else 16
        best.append((evaluate(f), "C", Y2))
# D: programs128 >= T1 and work >= Z
for T1 in (8,16,24,32):
    for Z in [2**k for k in range(16,24)]:
        for zs in (1,1.5,2,3):
            Z2=int(Z*zs)
            f=lambda g,hkv,bs,ext,pre,T1=T1,Z2=Z2: 128 if (p(g,hkv,bs,ext,128) >= T1 and p(g,hkv,bs,ext,16)*(pre+ext//2) >= Z2) else 16
            best.append((evaluate(f), "D", T1, Z2))
best.sort(key=lambda x:(x[0][1], x[0][0]))
for x in best[:8]: print(x)
print("by avg:")
for x in sorted(best,key=lambda x:(x[0][0],x[0][1]))[:8]: print(x)
print("--- B 2^20 bad cases")
f=lambda g,hkv,bs,ext,pre: 128 if p(g,hkv,bs,ext,16)*(pre+ext//2) >= 1<<20 else 16
print(evaluate(f, show=True))
print("--- B 2^20 with 64 when no prefix")
f=lambda g,hkv,bs,ext,pre: (128 if pre else 64) if p(g,hkv,bs,ext,16)*(pre+ext//2) >= 1<<20 else 16
print(evaluate(f, show=True))
