import importlib.util, os, sys, torch
from triton.testing import do_bench
N = open("/tmp/fgs-n1.path").read().strip(); M = open("/tmp/fgs-master.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
import flaggems_sglang
def load(root, name, alias):
    pkg = f"{root}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops"
    spec = importlib.util.spec_from_file_location(f"{alias}.{name}", f"{pkg}/{name}.py", submodule_search_locations=None)
    # load _decode_splitkv first under the alias package so relative imports resolve
    import types
    if alias not in sys.modules:
        p = types.ModuleType(alias); p.__path__ = [pkg]; sys.modules[alias] = p
    m = importlib.util.module_from_spec(spec); sys.modules[f"{alias}.{name}"] = m; spec.loader.exec_module(m); return m
for alias, root in (("m_ops", M), ("n_ops", N)):
    load(root, "_decode_splitkv", alias)
mods = {tag: (load(root, "decode_attention", alias), load(root, "decode_grouped_attention", alias)) for tag, alias, root in (("master", "m_ops", M), ("new", "n_ops", N))}
torch.manual_seed(0)
for (bs, hq, hkv, d, seq) in ((2, 128, 8, 512, 128), (2, 16, 1, 512, 128), (8, 32, 8, 512, 1024), (4, 64, 4, 256, 2048)):
    pool = bs * seq
    q = torch.randn(bs, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pool, hkv, d, device="cuda", dtype=torch.bfloat16); v = torch.randn(pool, hkv, d, device="cuda", dtype=torch.bfloat16)
    indptr = torch.arange(0, (bs + 1) * seq, seq, device="cuda", dtype=torch.int32); idx = torch.arange(pool, device="cuda", dtype=torch.int32)
    for i, name in enumerate(("decode_attention", "decode_grouped_attention")):
        fm, fn = getattr(mods["master"][i], name), getattr(mods["new"][i], name)
        same = torch.equal(fm(q, k, v, indptr, idx, d ** -0.5), fn(q, k, v, indptr, idx, d ** -0.5))
        tm = min(do_bench(lambda: fm(q, k, v, indptr, idx, d ** -0.5), warmup=10, rep=100, return_mode="median") for _ in range(2)) * 1e3
        tn = min(do_bench(lambda: fn(q, k, v, indptr, idx, d ** -0.5), warmup=10, rep=100, return_mode="median") for _ in range(2)) * 1e3
        print(f"{name:25s} bs={bs} hq={hq} hkv={hkv} d={d} seq={seq} | master {tm:7.1f} new {tn:7.1f} ({tm/tn:.2f}x) bitwise={same}")
