import importlib, importlib.util, os, sys, torch
W = open("/tmp/fgs-b.path").read().strip()
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gen = load("gen_kda", os.path.join(W, "src/flaggems_sglang/ops/fused_recurrent_kda.py"))
new = load("new_kda", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_recurrent_kda.py"))
bm = importlib.import_module("benchmark.test_fused_recurrent_kda")
kw = next(iter(bm._input_fn((1, 64), torch.bfloat16, "cuda")))
o0, s0 = gen.fused_recurrent_kda(**kw)
o0b, s0b = gen.fused_recurrent_kda(**kw)
print("generic run-to-run identical:", torch.equal(o0, o0b) and torch.equal(s0, s0b))
for cfg in ((1, 1), (1, 4), (2, 1), (4, 1), (8, 4), (16, 4), (16, 1), (32, 8)):
    new._config = lambda b, h, vd, c=cfg: c
    o, s = new.fused_recurrent_kda(**kw)
    do = (o.float() - o0.float()).abs(); ds = (s - s0).abs()
    print(f"cfg {cfg}: out mismatches {(o != o0).sum().item():6d}/{o.numel()} max {do.max().item():.2e} | state mismatches {(s != s0).sum().item():7d}/{s.numel()} max {ds.max().item():.2e}", flush=True)
print("--- bitwise")
def bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)
for cfg in ((1, 1), (1, 4), (2, 1), (4, 1), (8, 4), (16, 4), (16, 1), (32, 8)):
    new._config = lambda b, h, vd, c=cfg: c
    o, s = new.fused_recurrent_kda(**kw)
    bo = (bits(o) != bits(o0)); bs = (bits(s) != bits(s0))
    zo = int(((o == 0) & bo).sum()); zs = int(((s == 0) & bs).sum())
    print(f"cfg {cfg}: out bit diffs {int(bo.sum())} (all zeros: {zo}) | state bit diffs {int(bs.sum())} (all zeros: {zs})", flush=True)
