"""E3: The aliasing bug — assignment vs accumulation in backprop.

Naive scalar-autograd implementations write ``self.grad = out.grad`` instead
of ``self.grad += out.grad`` during the backward pass. This is invisible
whenever every node is consumed at most once, and silently wrong whenever a
value fans out. We quantify:

1. deterministic cases (a+a, diamonds, triple use),
2. over 100 random DAGs: how often the buggy pass differs from the verified
   engine, and by how much,
3. how often the buggy gradients still *pass* a PyTorch-tolerance gradient
   check (i.e. bugs that hide under loose tolerances).
"""

from __future__ import annotations

import math
import random

import common  # noqa: F401
from scalar_autograd_verified import graphgen
from scalar_autograd_verified.engine import Value
from scalar_autograd_verified.gradcheck import central_difference, gradcheck

N_GRAPHS = 100
N_OPS = 15
N_INPUTS = 3
SEED = 424242


class BuggyValue:
    """Same DAG API, but backward uses assignment (the classic bug)."""

    __slots__ = ("data", "grad", "_children", "_backward")

    def __init__(self, data):
        self.data = float(data)
        self.grad = 0.0
        self._children = []
        self._backward = lambda: None

    def __add__(self, other):
        out = BuggyValue(self.data + other.data)
        out._children = [self, other]

        def _bw():
            self.grad = out.grad
            other.grad = out.grad

        out._backward = _bw
        return out

    def __mul__(self, other):
        out = BuggyValue(self.data * other.data)
        out._children = [self, other]

        def _bw():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _bw
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = BuggyValue(t)
        out._children = [self]

        def _bw():
            self.grad = (1 - t * t) * out.grad

        out._backward = _bw
        return out

    def exp(self):
        out = BuggyValue(math.exp(self.data))
        out._children = [self]

        def _bw():
            self.grad = out.data * out.grad

        out._backward = _bw
        return out

    def relu(self):
        out = BuggyValue(max(0.0, self.data))
        out._children = [self]

        def _bw():
            self.grad = (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _bw
        return out

    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = BuggyValue(s)
        out._children = [self]

        def _bw():
            self.grad = s * (1 - s) * out.grad

        out._backward = _bw
        return out

    def __pow__(self, p):
        assert p == 2
        out = BuggyValue(self.data**2)
        out._children = [self]

        def _bw():
            self.grad = 2 * self.data * out.grad

        out._backward = _bw
        return out

    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for c in node._children:
                stack.append((c, False))
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def build_buggy(spec, vals):
    pool = [BuggyValue(v) for v in vals]
    for kind, i, j in spec:
        if j is None:
            if kind == "tanh":
                out = pool[i].tanh()
            elif kind == "exp":
                out = pool[i].exp()
            elif kind == "relu":
                out = pool[i].relu()
            elif kind == "sigmoid":
                out = pool[i].sigmoid()
            elif kind == "square":
                out = pool[i] ** 2
            else:
                raise ValueError(kind)
        else:
            out = (pool[i] + pool[j]) if kind == "+" else (pool[i] * pool[j])
        if abs(out.data) > 50.0:
            out = out.tanh()
        pool.append(out)
    return pool[-1], pool


def rel(a, b):
    return abs(a - b) / max(1e-8, abs(a), abs(b))


def main():
    out = {"config": {"n_graphs": N_GRAPHS, "n_ops": N_OPS, "n_inputs": N_INPUTS, "seed": SEED}}

    # --- deterministic cases
    a = Value(3.0)
    (a + a).backward()
    a2 = BuggyValue(3.0)
    (a2 + a2).backward()
    out["case_self_add"] = {"correct": a.grad, "buggy": a2.grad, "fd": 2.0}

    x = Value(0.4)
    y = Value(-0.9)
    (x * y + x * y).backward()
    bx, by = BuggyValue(0.4), BuggyValue(-0.9)
    (bx * by + bx * by).backward()
    out["case_repeated_product"] = {
        "correct": [x.grad, y.grad],
        "buggy_repeated_sum": [bx.grad, by.grad],
        "fd": [2 * -0.9, 2 * 0.4],
    }

    # --- random DAGs
    rng = random.Random(SEED)
    n_diff = 0
    n_caught = 0
    worst = 0.0
    rel_diffs = []
    for _ in range(N_GRAPHS):
        vals = [rng.uniform(-1, 1) for _ in range(N_INPUTS)]
        spec = graphgen.random_spec(rng, N_OPS, N_INPUTS)

        xs = [Value(v) for v in vals]
        fn = lambda: graphgen.build_from_spec(spec, xs)[0]  # noqa: E731
        res = gradcheck(fn, xs)  # sets xs[i].grad
        correct = [v.grad for v in xs]

        buggy_out, buggy_pool = build_buggy(spec, vals)
        buggy_out.backward()
        buggy = [buggy_pool[i].grad for i in range(N_INPUTS)]

        d = max(rel(c, b) for c, b in zip(correct, buggy))
        rel_diffs.append(d)
        if d > 1e-6:
            n_diff += 1
        worst = max(worst, d)

        # would the buggy gradients pass the same PyTorch-default check?
        caught = False
        for i in range(N_INPUTS):
            numeric = central_difference(fn, xs, i, 1e-6)
            if abs(buggy[i] - numeric) > 1e-5 + 1e-3 * abs(numeric):
                caught = True
        n_caught += 1 if caught else 0

    rel_diffs_sorted = sorted(rel_diffs)
    out["random_dags"] = {
        "n": N_GRAPHS,
        "graphs_with_observable_mismatch_gt_1e-6": n_diff,
        "mismatch_fraction": n_diff / N_GRAPHS,
        "worst_rel_diff": worst,
        "median_rel_diff": rel_diffs_sorted[N_GRAPHS // 2],
        "fraction_caught_by_gradcheck": n_caught / N_GRAPHS,
    }
    common.save_result("e3_aliasing_bug", out)
    print(
        "E3: self-add correct=%s buggy=%s | mismatch %.0f%% of DAGs, worst rel %.2e, gradcheck catches %.0f%%"
        % (
            out["case_self_add"]["correct"],
            out["case_self_add"]["buggy"],
            100 * out["random_dags"]["mismatch_fraction"],
            worst,
            100 * out["random_dags"]["fraction_caught_by_gradcheck"],
        )
    )
    return out


if __name__ == "__main__":
    main()
