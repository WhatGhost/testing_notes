"""Registers, spills and program counts of the generic and rdna3 kkt kernels
on the official 1024-token shape."""

import importlib
import importlib.util
import os
import sys

import torch

W = open("/tmp/fgs-b.path").read().strip()
sys.path[:0] = [W, os.path.join(W, "src")]
gmod = importlib.import_module("flaggems_sglang.ops.chunk_kda_scaled_dot_kkt")
spec = importlib.util.spec_from_file_location("kkt_r", f"{W}/src/flaggems_sglang/runtime/backend/_amd/rdna3/ops/chunk_kda_scaled_dot_kkt.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
bm = importlib.import_module("benchmark.test_chunk_kda_scaled_dot_kkt")

args = next(iter(bm._input_fn((1, 1024, 8, 128), torch.bfloat16, "cuda")))
gmod.chunk_kda_scaled_dot_kkt(*args)
new.chunk_kda_scaled_dot_kkt(*args)
torch.cuda.synchronize()


def report(name, fn):
    for dev_cache in fn.device_caches.values():
        for kern in dev_cache[0].values():
            kern._init_handles()
            md = kern.metadata
            print(f"{name}: regs {kern.n_regs} spills {kern.n_spills} warps {md.num_warps} shared {md.shared}")


report("generic tile_compute", gmod.tile_compute)
report("rdna3 _kkt_kernel", new._kkt_kernel)
