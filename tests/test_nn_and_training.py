"""Tests for the nn library and end-to-end training behavior."""

import sys

sys.path.insert(0, ".")

from micrograd_verified.engine import Value
from micrograd_verified.nn import MLP


def test_parameter_count_lecture_mlp():
    mlp = MLP(3, [4, 4, 1], seed=1)
    # 4*(3+1) + 4*(4+1) + 1*(4+1) = 16 + 20 + 5 = 41
    assert len(mlp.parameters()) == 41


def test_seeded_init_is_deterministic():
    a = MLP(3, [4, 4, 1], seed=11)
    b = MLP(3, [4, 4, 1], seed=11)
    assert [p.data for p in a.parameters()] == [p.data for p in b.parameters()]
    c = MLP(3, [4, 4, 1], seed=12)
    assert [p.data for p in a.parameters()] != [p.data for p in c.parameters()]


def test_zero_grad_clears_all():
    mlp = MLP(2, [3, 1], seed=2)
    out = mlp([0.5, -0.5])
    out.backward()
    assert any(p.grad != 0 for p in mlp.parameters())
    mlp.zero_grad()
    assert all(p.grad == 0 for p in mlp.parameters())


def test_training_reduces_loss_and_fits_lecture_dataset():
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    mlp = MLP(3, [4, 4, 1], seed=42)

    def loss_fn():
        return sum(((mlp(x) - y) ** 2 for x, y in zip(xs, ys)), Value(0.0))

    initial = loss_fn().data
    for _ in range(100):
        loss = loss_fn()
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.05 * p.grad
    final = loss_fn().data

    assert final < initial * 0.1
    assert final < 0.1
    preds = [mlp(x).data for x in xs]
    assert all((p > 0) == (y > 0) for p, y in zip(preds, ys))
