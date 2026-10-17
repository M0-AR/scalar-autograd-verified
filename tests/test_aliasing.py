"""Aliasing tests: shared sub-expressions, fan-out, and the classic += bug.

Backprop pass must *accumulate* gradients (``grad += ...``). If a variable is
used more than once, assignment semantics silently returns wrong gradients.
These tests pin the correct behavior and demonstrate that a buggy assignment
engine both (a) disagrees with analytic ground truth and (b) is caught by a
finite-difference check.
"""

import sys

sys.path.insert(0, ".")

from micrograd_verified.engine import Value
from micrograd_verified.gradcheck import central_difference


def test_self_add():
    a = Value(3.0)
    b = a + a
    b.backward()
    assert b.data == 6.0
    assert a.grad == 2.0  # d(a+a)/da = 2, NOT 1


def test_shared_subexpression_diamond():
    a, b = Value(2.0), Value(-1.0)
    c = a + b
    d = c * c  # c used twice through the same node
    d.backward()
    assert a.grad == 2.0  # dd/da = 2c = 2
    assert b.grad == 2.0


def test_triple_use():
    a = Value(1.5)
    y = a * a * a + a
    y.backward()
    # dy/da = 3a^2 + 1
    assert abs(a.grad - (3 * 1.5**2 + 1)) < 1e-12


def test_reused_across_branches():
    a, b = Value(0.5), Value(-2.0)
    u = a * b
    v = u + a
    w = u * v  # u appears in both branches
    w.backward()
    # analytic: w = u*(u+a), u=ab;  dw/da = (2u + a)*b + u
    u_val = 0.5 * -2.0
    expected_a = (2 * u_val + 0.5) * (-2.0) + u_val
    expected_b = (2 * u_val + 0.5) * (0.5)
    assert abs(a.grad - expected_a) < 1e-12
    assert abs(b.grad - expected_b) < 1e-12


class _BuggyValue:
    """Minimal engine with the classic bug: ``grad =`` instead of ``grad +=``."""

    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._children = []
        self._backward = lambda: None

    def __add__(self, other):
        out = _BuggyValue(self.data + other.data)
        out._children = [self, other]

        def _backward():
            self.grad = out.grad  # BUG: assignment, not accumulation
            other.grad = out.grad

        out._backward = _backward
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


def test_buggy_engine_is_wrong_and_gradcheck_catches_it():
    # Correct engine
    a = Value(3.0)
    (a + a).backward()
    correct = a.grad

    # Buggy engine
    b = _BuggyValue(3.0)
    (b + b).backward()
    buggy = b.grad

    assert correct == 2.0
    assert buggy == 1.0
    assert buggy != correct  # the bug is observable

    # Finite differences agree with the correct engine, catching the bug.
    x = Value(3.0)
    fd = central_difference(lambda: x + x, [x], 0, eps=1e-6)
    assert abs(fd - correct) < 1e-6
    assert abs(fd - buggy) > 0.5


def test_fanout_direct_gradcheck():
    from micrograd_verified.gradcheck import gradcheck

    a = Value(0.7)
    res = gradcheck(lambda: a + a * a + a * a * a, [a])
    assert res["passed"]
