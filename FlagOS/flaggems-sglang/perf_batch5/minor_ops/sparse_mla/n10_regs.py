import importlib.util, os, sys, torch
T = open("/tmp/fgs-f8.path").read().strip()
sys.path.insert(0, T)
spec = importlib.util.spec_from_file_location("fused", "/tmp/n10_smla_fused.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
bm = importlib.import_module("benchmark.test_triton_sparse_mla_fwd")
kw = next(iter(bm._input_fn((16, 512), torch.float8_e4m3fn, "cuda")))
orig = mod._attend
captured = {}
class Spy:
    def __getitem__(self, grid):
        def run(*a, **k):
            ck = orig[grid](*a, **k)
            captured["k"] = ck
            return ck
        return run
mod._attend = Spy()
for cfg in ((32, 4, 1, 8), (64, 4, 1, 8), (64, 8, 1, 8), (32, 2, 1, 8), (16, 1, 1, 8)):
    mod._attend_config = lambda s, h, t, c=cfg: c
    mod.triton_sparse_mla_fwd(**kw)
    ck = captured["k"]
    md = ck.metadata
    asm = ck.asm.get("amdgcn", "")
    scratch = [l for l in asm.splitlines() if "scratch" in l.lower() and ("size" in l.lower() or "private_segment" in l.lower())][:3]
    print(cfg, "n_regs", ck.n_regs, "n_spills", ck.n_spills, "shared", md.shared, scratch)
