"""fused_moe on gfx1100: SGLang 0.5.11 Triton path (as the plugin runs it on
AMD: default config, FlagGems align and silu_and_mul) vs FlagGems-sglang master
(generic op) vs the new rdna3 op. Same process, same inputs."""

import importlib.util
import json
import os
import sys

import torch
from triton.testing import do_bench

W = open("/tmp/fgs-n2.path").read().strip()
M = open("/tmp/fgs-master.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
import flaggems_sglang  # noqa: E402,F401
from flaggems_sglang.reference import get_reference  # noqa: E402

assert flaggems_sglang.__file__.startswith(W), flaggems_sglang.__file__


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sgl = load("sgl_moe", "/tmp/sglang_new/standalone/sgl_moe.py")
master = load("master_fused_moe", os.path.join(M, "src/flaggems_sglang/ops/fused_moe.py"))
new = load("new_fused_moe", os.path.join(W, "src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/fused_moe.py"))
ref = get_reference("fused_moe")
only = sys.argv[1] if len(sys.argv) > 1 else "all"


def bench(fn):
    return do_bench(fn, warmup=5, rep=50, return_mode="median") * 1e3


rows = []
models = {"official": (2048, 1408, 64, 8), "35B": (2048, 128, 256, 8)}
for model, (H, I, E, topk) in models.items():
    if only != "all" and only != model:
        continue
    gen = torch.Generator(device="cuda").manual_seed(0)
    w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
        tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
        tw, ti = tw.float().contiguous(), ti.to(torch.int32).contiguous()
        to = (tw, ti, None)
        err = {}
        if T <= 512:
            want = ref(x, w1, w2, to, "silu", 1.0).float()
            for name, out in (("sglang", sgl.fused_experts(x, w1, w2, tw, ti)),
                              ("master", master.fused_moe(x, w1, w2, to)),
                              ("new", new.fused_moe(x, w1, w2, to))):
                err[name] = (out.float() - want).abs().max().item()
        t_sgl = bench(lambda: sgl.fused_experts(x, w1, w2, tw, ti))
        t_master = bench(lambda: master.fused_moe(x, w1, w2, to))
        t_new = bench(lambda: new.fused_moe(x, w1, w2, to))
        rows.append(dict(model=model, T=T, sglang=t_sgl, master=t_master, new=t_new, err=err))
        es = " ".join(f"{k}={v:.2e}" for k, v in err.items())
        print(f"{model:8s} T={T:5d} | sglang {t_sgl:9.1f} | master {t_master:9.1f} | new {t_new:9.1f} | "
              f"new/sglang {t_sgl / t_new:5.2f}x | new/master {t_master / t_new:5.2f}x | {es}", flush=True)
json.dump(rows, open(f"/tmp/n3_three_way_{only}.json", "w"), indent=1)
