"""E6: Performance — forward+backward cost vs PyTorch, scaling, memory.

(a) Chains of length L in {16, 64, 256, 1024, 4096}: micrograd vs scalar
    float64 PyTorch tensors (same DAG shape), best-of-N wall time.
(b) A 2000-op random fan-out DAG: build + backward time.
(c) Memory: tracemalloc peak for the 4096 chain (micrograd only; PyTorch
    allocations happen outside Python's allocator and tracemalloc cannot see
    them — noted honestly).
(d) MLP training throughput on the lecture dataset.
"""

from __future__ import annotations

import random
import time
import tracemalloc

import common  # noqa: F401
from micrograd_verified import graphgen
from micrograd_verified.engine import Value
from micrograd_verified.nn import MLP

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

LENGTHS = (16, 64, 256, 1024, 4096)


def micro_chain(L):
    x = Value(1.000001)
    w = Value(0.9999)
    b = Value(0.01)
    s = x
    for _ in range(L // 2):
        s = s * w + b
    s.backward()
    return s.data


def torch_chain(L):
    x = torch.tensor(1.000001, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(0.9999, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    s = x
    for _ in range(L // 2):
        s = s * w + b
    s.backward()
    return s.item()


def main():
    out = {"has_torch": HAVE_TORCH, "torch_version": torch.__version__ if HAVE_TORCH else None,
           "chains": {}}

    for L in LENGTHS:
        repeats = 3 if L >= 4096 else 5
        tm = common.best_of(lambda L=L: micro_chain(L), repeats)
        entry = {"L": L, "micro_s": tm, "micro_ns_per_node": tm / L * 1e9}
        if HAVE_TORCH:
            tt = common.best_of(lambda L=L: torch_chain(L), repeats)
            entry.update({"torch_s": tt, "torch_ns_per_node": tt / L * 1e9,
                          "micro_over_torch": tm / tt})
        out["chains"][str(L)] = entry
        msg = f"E6 chain L={L:5d}: micro {tm*1e3:8.3f} ms ({entry['micro_ns_per_node']:7.1f} ns/node)"
        if HAVE_TORCH:
            msg += f" | torch {entry['torch_s']*1e3:8.3f} ms ({entry['torch_ns_per_node']:7.1f} ns/node) | ratio {entry['micro_over_torch']:.1f}x"
        print(msg)

    # random DAG 2000 ops
    rng = random.Random(2026)
    inputs = graphgen.make_inputs(rng, 4)
    spec = graphgen.random_spec(rng, 2000, 4)

    def micro_dag():
        xs = [Value(v.data) for v in inputs]
        out_v, _ = graphgen.build_from_spec(spec, xs)
        out_v.backward()

    t = common.best_of(micro_dag, 3)
    out["random_dag_2000_ops"] = {"seconds": t, "us_per_op": t / 2000 * 1e6}
    print(f"E6 random DAG 2000 ops (build+fwd+bwd): {t*1e3:.1f} ms ({t/2000*1e6:.1f} us/op)")

    # memory: chain 4096
    tracemalloc.start()
    micro_chain(4096)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out["memory_chain_4096"] = {
        "peak_bytes": peak,
        "current_bytes": current,
        "bytes_per_node_peak": peak / 4096,
        "note": "tracemalloc sees Python objects only; PyTorch's C++ allocator is invisible to it",
    }
    print(f"E6 tracemalloc peak (chain 4096): {peak/1024:.0f} KiB ({peak/4096:.0f} B/node)")

    # MLP training throughput on lecture dataset
    data = [([2.0, 3.0, -1.0], 1.0), ([3.0, -1.0, 0.5], -1.0),
            ([0.5, 1.0, 1.0], -1.0), ([1.0, 1.0, -1.0], 1.0)]
    mlp = MLP(3, [4, 4, 1], seed=0)

    def train_steps(n=50):
        for _ in range(n):
            loss = sum(((mlp(x) - y) ** 2 for x, y in data), Value(0.0))
            mlp.zero_grad()
            loss.backward()
            for p in mlp.parameters():
                p.data -= 0.05 * p.grad

    t = common.best_of(train_steps, 3)
    out["training_throughput"] = {
        "steps": 50,
        "seconds": t,
        "steps_per_s": 50 / t,
        "samples_per_s": 50 * 4 / t,
        "n_params": len(mlp.parameters()),
    }
    print(f"E6 MLP training: {50/t:.0f} steps/s ({50*4/t:.0f} samples/s, {len(mlp.parameters())} params)")

    common.save_result("e6_benchmarks", out)
    return out


if __name__ == "__main__":
    main()
