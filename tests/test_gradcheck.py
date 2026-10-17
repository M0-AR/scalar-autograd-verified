"""Numerical gradient-check tests (the core verification methodology)."""

import random
import sys

sys.path.insert(0, ".")

from micrograd_verified import graphgen
from micrograd_verified.engine import Value
from micrograd_verified.gradcheck import gradcheck


def test_gradcheck_quadratic():
    x = Value(3.0)
    res = gradcheck(lambda: 3 * x**2 - 4 * x + 5, [x])
    assert res["passed"]
    assert res["max_abs_error"] < 1e-7


def test_gradcheck_neuron_expression():
    x1, x2, w1, w2, b = [Value(v) for v in (2.0, 0.0, -3.0, 1.0, 6.8813735870195432)]
    res = gradcheck(lambda: (x1 * w1 + x2 * w2 + b).tanh(), [x1, x2, w1, w2, b])
    assert res["passed"]
    by_val = {round(p["data"], 6): p for p in res["per_input"]}
    assert abs(by_val[2.0]["analytic"] - (-1.5)) < 1e-12
    assert abs(by_val[0.0]["analytic"] - 0.5) < 1e-12


def test_gradcheck_fanout_graph():
    a, b = Value(0.3), Value(-0.8)
    res = gradcheck(lambda: (a + b) * (a + b) + (a * b).tanh(), [a, b])
    assert res["passed"]
    assert res["max_rel_error"] < 1e-6


def test_gradcheck_random_graphs_batch():
    rng = random.Random(1234)
    failures = 0
    worst = 0.0
    for _ in range(40):
        inputs = graphgen.make_inputs(rng, 3)
        spec = graphgen.random_spec(rng, n_ops=12, n_inputs=3)

        def fn(inputs=inputs, spec=spec):
            out, _ = graphgen.build_from_spec(spec, inputs)
            return out

        res = gradcheck(fn, inputs)
        failures += 0 if res["passed"] else 1
        worst = max(worst, res["max_rel_error"])
    assert failures == 0, f"{failures}/40 random graphs failed gradcheck"
    assert worst < 1e-5, f"worst relative error too high: {worst}"


def test_gradcheck_deep_graph():
    rng = random.Random(99)
    inputs = graphgen.make_inputs(rng, 2)
    spec = graphgen.random_spec(rng, n_ops=30, n_inputs=2)

    def fn():
        out, _ = graphgen.build_from_spec(spec, inputs)
        return out

    res = gradcheck(fn, inputs)
    assert res["passed"], res
