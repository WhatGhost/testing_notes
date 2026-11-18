import importlib.util, os, sys, time, types, torch
N = open("/tmp/fgs-n1.path").read().strip(); M = open("/tmp/fgs-master.path").read().strip()
sys.path.insert(0, os.path.join(N, "src"))
import flaggems_sglang
def load(root, name, alias):
    pkg = f"{root}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops"
    if alias not in sys.modules:
        p = types.ModuleType(alias); p.__path__ = [pkg]; sys.modules[alias] = p
    spec = importlib.util.spec_from_file_location(f"{alias}.{name}", f"{pkg}/{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[f"{alias}.{name}"] = m; spec.loader.exec_module(m); return m
mods = {}
for tag, alias, root in (("master", "m_ops", M), ("new", "n_ops", N)):
    load(root, "_decode_splitkv", alias)
    mods[tag] = load(root, "decode_attention", alias)
q = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(20, 4, 64, device="cuda", dtype=torch.bfloat16); v = torch.randn(20, 4, 64, device="cuda", dtype=torch.bfloat16)
indptr = torch.tensor([0, 10, 20], device="cuda", dtype=torch.int32); idx = torch.arange(20, device="cuda", dtype=torch.int32)
for tag in ("master", "new", "master", "new"):
    f = mods[tag].decode_attention
    for _ in range(200): f(q, k, v, indptr, idx, 0.125)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2000): f(q, k, v, indptr, idx, 0.125)
    t1 = time.perf_counter(); torch.cuda.synchronize()
    print(tag, "host us/call %.2f" % ((t1 - t0) / 2000 * 1e6))
sk = sys.modules["n_ops._decode_splitkv"]
t0 = time.perf_counter()
for _ in range(20000): sk.needs_int64(k, v)
print("needs_int64 us/call %.2f" % ((time.perf_counter() - t0) / 20000 * 1e6))
sk.needs_int64 = lambda k, v, ks, vs: False
f = mods["new"].decode_attention
for name in ("new, needs_int64 stubbed",):
    for _ in range(200): f(q, k, v, indptr, idx, 0.125)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(2000): f(q, k, v, indptr, idx, 0.125)
    t1 = time.perf_counter(); torch.cuda.synchronize()
    print(name, "host us/call %.2f" % ((t1 - t0) / 2000 * 1e6))
def fast_int64(k_buffer, v_buffer):
    ks, vs = k_buffer.stride(), v_buffer.stride()
    kn, vn = k_buffer.shape, v_buffer.shape
    return max(kn[0] * ks[0], vn[0] * vs[0], kn[1] * ks[1], vn[1] * vs[1]) >= 1 << 31
t0 = time.perf_counter()
for _ in range(20000): fast_int64(k, v)
print("shape*stride check us/call %.2f" % ((time.perf_counter() - t0) / 20000 * 1e6))
t0 = time.perf_counter()
for _ in range(20000): max(k.numel(), v.numel()) >= 1 << 30
print("numel check us/call %.2f" % ((time.perf_counter() - t0) / 20000 * 1e6))
