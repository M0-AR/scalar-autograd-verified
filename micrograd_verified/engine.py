"""Scalar reverse-mode automatic differentiation engine.

A verified re-derivation of the autograd core popularized by karpathy/micrograd
(MIT). Every node is a scalar wrapped in a :class:`Value`; arithmetic builds a
dynamic DAG; :meth:`Value.backward` seeds dL/dL = 1 and runs the chain rule in
reverse topological order.

Two deliberate deviations from the reference implementation, each studied
empirically in experiments/ (see README, Sections 3 and 5):

1. ``_prev`` is stored as an insertion-ordered, identity-deduplicated tuple by
   default (``MICROGRAD_PREV_MODE=ordered``) instead of a ``set``. The ``set``
   variant remains available (``MICROGRAD_PREV_MODE=set``) to measure how
   hash-order-dependent iteration affects last-ulp reproducibility (E5).
2. Topological ordering is iterative (explicit stack) rather than recursive, so
   arbitrarily deep graphs back-propagate without hitting Python's recursion
   limit (E6 scales chains up to 4096 nodes).

All gradients are accumulated with ``+=`` into buffers initialized at 0.0,
which is what makes graphs with shared sub-expressions (fan-out > 1) correct —
the classic bug documented in E3.
"""

from __future__ import annotations

import math
import os

__all__ = ["Value"]


def _dedup_ordered(children) -> tuple:
    """Deduplicate children by object identity, preserving insertion order."""
    seen: set[int] = set()
    out = []
    for c in children:
        if id(c) not in seen:
            seen.add(id(c))
            out.append(c)
    return tuple(out)


class Value:
    """A scalar node of a differentiable DAG.

    Attributes:
        data: the scalar value (float64).
        grad: derivative of the final output w.r.t. this node (initialized 0.0).
        label: optional human-readable name (used by visualizations).
    """

    _prev_mode = os.environ.get("MICROGRAD_PREV_MODE", "ordered")

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, _children=(), _op: str = "", label: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None  # leaves define no chain-rule step
        if Value._prev_mode == "set":
            self._prev = set(_children)
        else:
            self._prev = _dedup_ordered(_children)
        self._op = _op
        self.label = label

    # ------------------------------------------------------------------ utils
    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def zero_grad(self) -> None:
        self.grad = 0.0

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __truediv__(self, other):
        return self * (other**-1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) / self

    # ---------------------------------------------------------- nonlinearities
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            z = math.exp(x)
            s = z / (1.0 + z)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ autograd
    def _topo(self) -> list["Value"]:
        """Iterative post-order DFS; children appear before parents."""
        topo: list[Value] = []
        visited: set[Value] = set()
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
            for child in node._prev:
                stack.append((child, False))
        return topo

    def backward(self) -> None:
        """Seed dL/dL = 1 and accumulate gradients over the whole DAG."""
        topo = self._topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
