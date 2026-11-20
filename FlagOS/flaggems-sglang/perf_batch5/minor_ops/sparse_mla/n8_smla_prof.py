import importlib.util, os, torch
from torch.profiler import profile, ProfilerActivity
T = open("/tmp/fgs-f8.path").read().strip()
spec = importlib.util.spec_from_file_location("gen_smla", os.path.join(T, "src/flaggems_sglang/ops/triton_sparse_mla_fwd.py"))
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
import sys
sys.path.insert(0, T)
bm = importlib.import_module("benchmark.test_triton_sparse_mla_fwd")
for shape in ((128, 2048), (128, 512), (16, 2048), (1, 2048), (128, 128)):
    kw = next(iter(bm._input_fn(shape, torch.float8_e4m3fn, "cuda")))
    for _ in range(3):
        gen.triton_sparse_mla_fwd(**kw)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            gen.triton_sparse_mla_fwd(**kw)
        torch.cuda.synchronize()
    tot = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            tot[e.name] = tot.get(e.name, 0) + e.device_time_total / 10
    print(f"shape {shape}: q {tuple(kw['q_nope'].shape)} kv {tuple(kw['kv'].shape)} idx {tuple(kw['indices'].shape)} | " + ", ".join(f"{k[:22]} {v:.1f}us" for k, v in sorted(tot.items(), key=lambda x: -x[1])), flush=True)
