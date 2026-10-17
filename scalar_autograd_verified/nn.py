"""A minimal neural-network library on top of the scalar autograd engine.

Mirrors the PyTorch module API: every module exposes ``parameters()`` and
``zero_grad()``; layers are called with ``__call__``. Deterministic when a
``seed`` is provided (a single shared ``random.Random`` streams through all
layers, so parameter counts and order fully determine initialization).
"""

from __future__ import annotations

import random

from .engine import Value

__all__ = ["Module", "Neuron", "Layer", "MLP"]


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    """y = tanh(sum_i w_i * x_i + b) (linear if ``nonlin=False``)."""

    def __init__(self, nin: int, nonlin: bool = True, rng: random.Random | None = None):
        rng = rng or random
        self.w = [Value(rng.uniform(-1, 1), label=f"w{i}") for i in range(nin)]
        self.b = Value(0.0, label="b")
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin: int, nout: int, nonlin: bool = True, rng=None):
        self.neurons = [Neuron(nin, nonlin=nonlin, rng=rng) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Multi-layer perceptron: ``nouts`` gives the width of each layer."""

    def __init__(self, nin: int, nouts: list[int], seed: int | None = None):
        rng = random.Random(seed) if seed is not None else random
        szs = [nin] + list(nouts)
        self.layers = [Layer(szs[i], szs[i + 1], rng=rng) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
