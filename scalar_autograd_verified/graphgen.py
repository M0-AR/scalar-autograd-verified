"""Random differentiable program generator (bounded expression DAGs).

Two-phase design so graphs can be *replayed* many times (finite-difference
gradient checking evaluates the expression O(2n) times):

1. ``random_spec(rng, n_ops, n_inputs)`` samples a program: a list of
   ``(op, i, j)`` tuples indexing a growing pool of nodes.
2. ``build_from_spec(spec, inputs)`` evaluates it with concrete :class:`Value`
   leaves, recreating the identical DAG.

Values are kept bounded by squashing any node with |data| > 50 through tanh,
so finite-difference references stay numerically meaningful.
"""

from __future__ import annotations

import random

from .engine import Value

UNARY_OPS = ("tanh", "exp", "relu", "sigmoid", "square")
BINARY_OPS = ("+", "*")


def _apply_unary(v: Value, kind: str) -> Value:
    if kind == "tanh":
        return v.tanh()
    if kind == "exp":
        return v.exp()
    if kind == "relu":
        return v.relu()
    if kind == "sigmoid":
        return v.sigmoid()
    if kind == "square":
        return v**2
    raise ValueError(kind)


def random_spec(rng: random.Random, n_ops: int, n_inputs: int):
    """Sample an op sequence. Every step indexes the current pool."""
    spec = []
    pool_size = n_inputs
    for _ in range(n_ops):
        if rng.random() < 0.4 or pool_size < 2:
            spec.append((rng.choice(UNARY_OPS), rng.randrange(pool_size), None))
        else:
            spec.append(
                (
                    rng.choice(BINARY_OPS),
                    rng.randrange(pool_size),
                    rng.randrange(pool_size),
                )
            )
        pool_size += 1
    return spec


def build_from_spec(spec, inputs):
    """Replay a spec; returns (output, pool) with pool[0:len(inputs)] == inputs."""
    pool = list(inputs)
    for kind, i, j in spec:
        if j is None:
            out = _apply_unary(pool[i], kind)
        else:
            out = (pool[i] + pool[j]) if kind == "+" else (pool[i] * pool[j])
        if abs(out.data) > 50.0:
            out = out.tanh()
        pool.append(out)
    return pool[-1], pool


def make_inputs(rng: random.Random, n: int, lo: float = -1.0, hi: float = 1.0):
    return [Value(rng.uniform(lo, hi), label=f"x{i}") for i in range(n)]


def random_expression(rng: random.Random, n_ops: int, inputs: list[Value]):
    """One-shot convenience: sample a spec and build it immediately."""
    spec = random_spec(rng, n_ops, len(inputs))
    return build_from_spec(spec, inputs)
