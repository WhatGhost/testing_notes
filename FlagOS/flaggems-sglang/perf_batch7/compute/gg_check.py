"""rdna3 grouped_gemm vs the reference: unit-test cases plus edge cases
(ragged N / K, many empty experts, unassigned head and tail rows)."""

import importlib
import importlib.util
import os
import sys

import torch

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gg = importlib.import_module("flaggems_sglang.ops.grouped_gemm")
from flaggems_sglang.reference.grouped_gemm import reference  # noqa: E402

spec = importlib.util.spec_from_file_location("gg_c", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/grouped_gemm.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
meta = importlib.import_module("benchmark.test_grouped_gemm")._metadata


def case(m, n, k, e, head=0, tail=0, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", generator=g).to(dtype)
    b = (0.1 * torch.randn(e, n, k, device="cuda", generator=g)).to(dtype)
    ids = torch.sort(torch.randint(0, e, (m - head - tail,), device="cuda", generator=g, dtype=torch.int32))[0]
    counts, offs, boffs, sched = meta(ids.contiguous(), e)
    offs = (offs + head).to(torch.int32)
    return a, b, e, counts, offs, boffs, sched


cases = [("test (128,256,512,4)", case(128, 256, 512, 4)), ("test (1024,512,1024,8)", case(1024, 512, 1024, 8)),
         ("test (333,256,256,16)", case(333, 256, 256, 16))]
cases += [(f"bench ({m},512,2048,{e})", case(m, 512, 2048, e)) for m in (128, 1024, 4096) for e in (8, 32)]
cases += [("ragged N=200 K=300", case(1000, 200, 300, 8)), ("many empty experts (600 rows, 256 experts)", case(600, 512, 2048, 256)),
          ("head 10 / tail 7 unassigned rows", case(1200, 256, 512, 8, head=10, tail=7)),
          ("fp16", case(777, 384, 1024, 16, dtype=torch.float16)), ("single expert", case(600, 512, 2048, 1)),
          ("large rows per expert, K=768", case(4096, 2048, 768, 8)), ("large rows per expert, K=2048", case(4096, 512, 2048, 8)),
          ("N=1536 K=2048 E=128", case(4096, 1536, 2048, 128)), ("N=2048 K=768 E=128", case(4096, 2048, 768, 128)),
          ("512 rows, N=1536 E=128", case(512, 1536, 2048, 128)), ("4096 rows, N=512 E=128", case(4096, 512, 2048, 128)),
          ("512 rows, E=32", case(512, 512, 2048, 32)), ("768 rows, E=32", case(768, 512, 2048, 32)),
          ("600 rows, N=256 K=300 E=32", case(600, 256, 300, 32))]
bad = 0
for name, args in cases:
    ref = reference(*args)
    out = new.grouped_gemm(*args)
    g = gg.grouped_gemm(*args)
    err, err_g = (out.float() - ref.float()).abs().max().item(), (g.float() - ref.float()).abs().max().item()
    ulp = (out.view(torch.int16) != ref.view(torch.int16)).float().mean().item()
    ulp_g = (g.view(torch.int16) != ref.view(torch.int16)).float().mean().item()
    ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2) and not torch.isnan(out).any().item()
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:42s} max err generic {err_g:.1e} new {err:.1e} | outputs off the reference "
          f"generic {ulp_g:.2%} new {ulp:.2%}", flush=True)
print(f"{len(cases) - bad}/{len(cases)} within the unit-test tolerance")
