import importlib, os, statistics, sys
import torch
W = open("/tmp/fgs-n2.path").read().strip()
sys.path.insert(0, os.path.join(W, "src"))
r = importlib.import_module("flaggems_sglang.runtime.backend._amd.rdna3.ops.fused_moe")
read_buf = torch.ones(512 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
def timed(fn, reps=100, warmup=10):
    for _ in range(warmup):
        torch.sum(read_buf, dtype=torch.float32); fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        torch.sum(read_buf, dtype=torch.float32)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)
H, I, E, topk = 2048, 128, 256, 8
gen = torch.Generator(device="cuda").manual_seed(0)
w1 = (torch.randn(E, 2 * I, H, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, H, I, device="cuda", generator=gen) * 0.05).to(torch.bfloat16)
tiny = torch.empty(1, device="cuda")
t_launch = timed(lambda: tiny.add_(1))
print(f"empty kernel (launch floor, read flush): {t_launch:.1f} us")
for T in (1, 2, 4, 8, 16):
    x = torch.randn(T, H, device="cuda", generator=gen).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(T, E, device="cuda", generator=gen), -1), topk, dim=-1)
    to = (tw.float().contiguous(), ti.to(torch.int32).contiguous(), None)
    n_exp = to[1].unique().numel()
    nbytes = n_exp * 3 * I * H * 2
    t = timed(lambda: r.fused_moe(x, w1, w2, to))
    print(f"35B T={T:2d}: fused_moe {t:6.1f} us, unique-expert weights {nbytes / 1e6:5.1f} MB -> {nbytes / t / 1e3:4.0f} GB/s", flush=True)
