"""Unit tests for the core engine: ops, known gradients, operator protocol."""

import math
import sys

import pytest

sys.path.insert(0, ".")

from scalar_autograd_verified.engine import Value


def test_forward_values():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    assert c.data == -2.0
    assert d.data == 0.0  # -8 + 8


def test_known_gradients_simple():
    a = Value(3.0)
    b = Value(-2.0)
    c = a + b
    d = a * b
    e = c * d
    e.backward()
    # c = a+b, d = a*b, e = c*d
    # de/da = d*(dc/da) + c*(dd/da) = d + c*b = -6 + 1*(-2) = -8
    # de/db = d + c*a = -6 + 1*3 = -3
    assert e.grad == 1.0
    assert a.grad == -8.0
    assert b.grad == -3.0


def test_quadratic_derivative():
    # f(x) = 3x^2 - 4x + 5, f'(3) = 14
    x = Value(3.0)
    f = 3 * x**2 - 4 * x + 5
    f.backward()
    assert f.data == 20.0
    assert x.grad == 14.0


def test_reflected_operators():
    a = Value(2.0)
    assert (2 + a).data == 4.0
    assert (2 * a).data == 4.0
    assert (2 - a).data == 0.0
    assert (2 / a).data == 1.0
    assert (a - 2).data == 0.0
    assert (a / 2).data == 1.0


def test_division_and_pow_grads():
    a = Value(4.0)
    b = Value(2.0)
    c = a / b
    c.backward()
    assert c.data == 2.0
    assert a.grad == 0.5
    assert abs(b.grad - (-1.0)) < 1e-12  # dc/db = -a/b^2 = -1


def test_nonlinearities_values_and_grads():
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    assert abs(x.grad - (1 - math.tanh(0.5) ** 2)) < 1e-12

    x = Value(1.3)
    y = x.exp()
    y.backward()
    assert abs(x.grad - math.exp(1.3)) < 1e-9

    x = Value(-0.7)
    y = x.relu()
    y.backward()
    assert y.data == 0.0 and x.grad == 0.0

    x = Value(1.1)
    y = x.sigmoid()
    y.backward()
    s = 1 / (1 + math.exp(-1.1))
    assert abs(x.grad - s * (1 - s)) < 1e-12


def test_backward_on_intermediate_node_seeds_that_node():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    unrelated = Value(5.0)
    c.backward()  # seeds dL/dL=1 at c, propagates to a and b only
    assert c.grad == 1.0
    assert a.grad == 3.0 and b.grad == 2.0
    assert unrelated.grad == 0.0


def test_accumulation_requires_zeroing():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    c.backward()  # same as forgetting zero_grad in a training loop
    assert a.grad == 6.0  # 3 + 3 -- documented, matches reference engine


def test_deep_chain_no_recursion_error():
    x = Value(1.0)
    s = x
    for _ in range(10_000):
        s = s * 1.0001
    s.backward()
    assert abs(x.grad - 1.0001**10_000) / x.grad < 1e-9


def test_slots_and_repr():
    a = Value(1.5, label="a")
    assert "Value(data=1.5" in repr(a)
    assert a.label == "a"
    with pytest.raises(AttributeError):
        a.new_attribute = 1  # __slots__ enforced
