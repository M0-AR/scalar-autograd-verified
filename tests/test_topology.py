"""Topological-order tests: correctness of ordering, dedup, and depth safety."""

import random
import sys

sys.path.insert(0, ".")

from micrograd_verified import graphgen
from micrograd_verified.engine import Value


def test_reversed_topo_is_valid_topological_order():
    rng = random.Random(7)
    for _ in range(20):
        inputs = graphgen.make_inputs(rng, 3)
        spec = graphgen.random_spec(rng, n_ops=20, n_inputs=3)
        out, pool = graphgen.build_from_spec(spec, inputs)
        topo = out._topo()
        pos = {id(n): i for i, n in enumerate(topo)}
        assert len(pos) == len(topo)  # no duplicates in topo
        for node in topo:
            for child in node._prev:
                assert pos[id(child)] < pos[id(node)], "child after parent in topo"


def test_shared_children_appear_once():
    a, b = Value(1.0), Value(2.0)
    c = a + b
    d = c * c
    topo = d._topo()
    ids = [id(n) for n in topo]
    assert len(ids) == len(set(ids))
    assert ids.count(id(c)) == 1


def test_operand_order_does_not_change_gradients():
    # a + b vs b + a and a * b vs b * a are compiled to identical semantics
    rng = random.Random(11)
    vals = [rng.uniform(-1, 1) for _ in range(3)]
    g = []
    for swapped in (False, True):
        xs = [Value(v) for v in vals]
        a, b, c = xs
        e = (b + a if swapped else a + b) * (b * a if swapped else a * b) + c
        e.backward()
        g.append([x.grad for x in xs])
    assert g[0] == g[1]  # bit-for-bit equal


def test_rebuild_is_bitwise_deterministic():
    rng = random.Random(21)
    vals = [0.31, -0.77, 0.19]
    spec = graphgen.random_spec(rng, n_ops=25, n_inputs=3)
    runs = []
    for _ in range(2):
        xs = [Value(v) for v in vals]
        out, _ = graphgen.build_from_spec(spec, xs)
        out.backward()
        runs.append([x.grad for x in xs])
    assert runs[0] == runs[1]


def test_very_deep_chain_no_recursion_error():
    depth = 50_000
    x = Value(1.0)
    s = x
    for _ in range(depth):
        s = s + 0.0
    s.backward()
    assert x.grad == 1.0
