"""Determinism probe: rebuild a fixed batch of graphs and emit grad hashes.

Run as a subprocess by experiments/e5_determinism.py with specific
PYTHONHASHSEED and MICROGRAD_PREV_MODE values. Prints one JSON line:
per-graph sha256 hashes, per-graph grads, within-process rebuild check.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from micrograd_verified import graphgen  # noqa: E402
from micrograd_verified.engine import Value  # noqa: E402

SEED = 77
N_GRAPHS = 60
N_OPS = 50
N_INPUTS = 4


def one_graph(rng):
    vals = [rng.uniform(-1, 1) for _ in range(N_INPUTS)]
    spec = graphgen.random_spec(rng, N_OPS, N_INPUTS)
    xs = [Value(v) for v in vals]
    out, _ = graphgen.build_from_spec(spec, xs)
    out.backward()
    return [x.grad for x in xs] + [out.data]


def main():
    rng = random.Random(SEED)
    graphs = []
    t0 = time.perf_counter()
    for _ in range(N_GRAPHS):
        graphs.append(one_graph(rng))
    elapsed = time.perf_counter() - t0

    # within-process rebuild control: same spec, fresh Values, same process
    rng2 = random.Random(SEED)
    rebuild_ok = True
    for _ in range(N_GRAPHS):
        vals = [rng2.uniform(-1, 1) for _ in range(N_INPUTS)]
        spec = graphgen.random_spec(rng2, N_OPS, N_INPUTS)
        g1 = _build_and_grad(spec, vals)
        g2 = _build_and_grad(spec, vals)
        if g1 != g2:
            rebuild_ok = False
            break

    hashes = []
    grads = []
    for g in graphs:
        s = "|".join("%.17g" % v for v in g)
        hashes.append(hashlib.sha256(s.encode()).hexdigest()[:16])
        grads.append([float(v) for v in g])

    print(json.dumps({
        "mode": os.environ.get("MICROGRAD_PREV_MODE", "ordered"),
        "hashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "hashes": hashes,
        "grads": grads,
        "elapsed_s": elapsed,
        "within_process_rebuild_identical": rebuild_ok,
    }))


def _build_and_grad(spec, vals):
    xs = [Value(v) for v in vals]
    out, _ = graphgen.build_from_spec(spec, xs)
    out.backward()
    return [x.grad for x in xs] + [out.data]


if __name__ == "__main__":
    main()
