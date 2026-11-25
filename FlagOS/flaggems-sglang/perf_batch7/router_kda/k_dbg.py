import importlib, importlib.util, os, sys, torch
W = open("/tmp/fgs-b.path").read().strip()
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gen = load("gen_kda", os.path.join(W, "src/flaggems_sglang/ops/fused_recurrent_kda.py"))
new = load("new_kda", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py"))
bm = importlib.import_module("benchmark.test_fused_recurrent_kda")
all_kw = [next(iter(bm._input_fn(s, torch.bfloat16, "cuda"))) for s in list(bm.SHAPES) + list(bm.MORE_SHAPES)]
kw = all_kw[0]
o1, s1 = gen.fused_recurrent_kda(**kw)
o2, s2 = new.fused_recurrent_kda(**kw)
o3, s3 = gen.fused_recurrent_kda(**kw)
for name, (a, b) in {"gen vs new out": (o1, o2), "gen vs new state": (s1, s2), "gen vs gen out": (o1, o3), "gen vs gen state": (s1, s3)}.items():
    va = a.view(torch.int16) if a.element_size() == 2 else a.view(torch.int32)
    vb = b.view(torch.int16) if b.element_size() == 2 else b.view(torch.int32)
    d = (va != vb)
    print(f"{name}: bit diffs {int(d.sum())}, value diffs {int((a != b).sum())}, nan a {int(torch.isnan(a.float()).sum())} b {int(torch.isnan(b.float()).sum())}", flush=True)
    if d.any():
        idx = d.nonzero()[:3].tolist()
        print("   first diffs at", idx, [(a[tuple(i)].item(), b[tuple(i)].item()) for i in idx])
