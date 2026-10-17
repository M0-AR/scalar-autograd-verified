"""Cross-check scalar-autograd-verified against PyTorch autograd (oracle tests)."""

import random
import sys

import pytest

sys.path.insert(0, ".")

torch = pytest.importorskip("torch")

from scalar_autograd_verified.engine import Value
from scalar_autograd_verified.nn import MLP


def _rel(a, b, floor=1e-12):
    return abs(a - b) / max(floor, abs(a), abs(b))


def test_expression_matches_torch():
    vals = [0.7, -1.3, 2.1]
    tx = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in vals]
    mx = [Value(v) for v in vals]

    def build(seq, tanh, sigmoid, relu, exp):
        a, b, c = seq
        return (
            (a * b + tanh(a)).sigmoid() if sigmoid else (a * b + tanh(a))
        ) + (a * a) * b + exp(a) * 0.1 + relu(b) - c * 0.5

    tf = build(tx, torch.tanh, None, torch.relu, torch.exp)
    mf = build(mx, lambda v: v.tanh(), None, lambda v: v.relu(), lambda v: v.exp())
    tf.backward()
    mf.backward()

    assert _rel(tf.item(), mf.data) < 1e-12
    for i in range(3):
        assert _rel(tx[i].grad.item(), mx[i].grad) < 1e-10, f"input {i}"


def _torch_replica(mlp):
    return [
        torch.tensor(p.data, dtype=torch.float64, requires_grad=True)
        for p in mlp.parameters()
    ]


def _torch_forward(params, x, sizes):
    idx = 0
    h = x
    for li in range(1, len(sizes)):
        nin, nout = sizes[li - 1], sizes[li]
        outs = []
        for _ in range(nout):
            acc = params[idx + nin]
            for k in range(nin):
                acc = acc + params[idx + k] * h[k]
            outs.append(torch.tanh(acc))
            idx += nin + 1
        h = outs
    return h[0]


def test_mlp_gradients_match_torch():
    rng = random.Random(3)
    sizes = [2, 3, 1]
    mlp = MLP(2, sizes[1:], seed=42)
    x = [0.4, -0.9]
    t = 1.0

    y = mlp(x)
    loss = (y - t) ** 2
    mlp.zero_grad()
    loss.backward()

    tp = _torch_replica(mlp)
    tx = [torch.tensor(v, dtype=torch.float64) for v in x]
    tloss = (_torch_forward(tp, tx, sizes) - t) ** 2
    tloss.backward()

    assert _rel(loss.data, tloss.item()) < 1e-12
    for p, q in zip(mlp.parameters(), tp):
        assert _rel(p.grad, q.grad.item()) < 1e-9


def test_sgd_trajectories_match_torch():
    rng = random.Random(5)
    data = [
        ([2.0, 3.0, -1.0], 1.0),
        ([3.0, -1.0, 0.5], -1.0),
        ([0.5, 1.0, 1.0], -1.0),
        ([1.0, 1.0, -1.0], 1.0),
    ]
    mlp = MLP(3, [4, 4, 1], seed=7)
    tp = _torch_replica(mlp)
    sizes = [3, 4, 4, 1]
    lr, steps = 0.05, 30

    for _ in range(steps):
        loss = sum(((mlp(x) - t) ** 2 for x, t in data), Value(0.0))
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= lr * p.grad

        tloss = sum((_torch_forward(tp, [torch.tensor(v, dtype=torch.float64) for v in x], sizes) - t) ** 2 for x, t in data)
        for q in tp:
            q.grad = None
        tloss.backward()
        with torch.no_grad():
            for q in tp:
                q -= lr * q.grad

    assert _rel(loss.data, tloss.item()) < 1e-6
