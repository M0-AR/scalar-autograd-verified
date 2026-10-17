"""E5: Reproducibility across processes — ordered tuple vs hash-ordered set.

The reference engine stores ``_prev`` as a Python ``set``; its iteration order
depends on object hashes (memory addresses), so the order in which the chain
rule accumulates into a shared node can vary from process to process, changing
results in the last ulps. Our default stores an insertion-ordered deduplicated
tuple instead.

Method: run scripts/prev_mode_probe.py in 6 fresh subprocesses:
    {ordered, set} x PYTHONHASHSEED {0, 1, 2}
Each run rebuilds 60 identical fan-out-heavy random graphs (graph spec seeded
by a constant) and reports grad hashes and values. We then compare:
  * across runs of the same mode  -> cross-process reproducibility
  * within a process (rebuild)    -> intra-process reproducibility
  * set vs ordered (pairwise)     -> max ulp deviation from accumulation order
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import common  # noqa: F401

PROBE = os.path.join(common.ROOT, "scripts", "prev_mode_probe.py")
MODES = ("ordered", "set")
HASH_SEEDS = ("0", "1", "2")


def run_probe(mode: str, hashseed: str):
    env = dict(os.environ)
    env["MICROGRAD_PREV_MODE"] = mode
    env["PYTHONHASHSEED"] = hashseed
    t0 = __import__("time").perf_counter()
    proc = subprocess.run(
        [sys.executable, PROBE], env=env, capture_output=True, text=True, cwd=common.ROOT
    )
    wall = __import__("time").perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"probe failed ({mode}, seed {hashseed}):\n{proc.stderr}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["wall_s"] = wall
    return payload


def ulp_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / math.ulp(scale)


def main():
    runs = {mode: [run_probe(mode, hs) for hs in HASH_SEEDS] for mode in MODES}

    out = {"config": {"n_graphs": 60, "n_ops": 50, "hashseeds": list(HASH_SEEDS)}, "modes": {}}
    for mode, rs in runs.items():
        agg = [tuple(r["hashes"]) for r in rs]
        distinct = len(set(agg))

        max_ulp = 0.0
        n_graphs_diff = 0
        for g in range(len(rs[0]["grads"])):
            any_diff = False
            for r in rs[1:]:
                for a, b in zip(rs[0]["grads"][g], r["grads"][g]):
                    d = ulp_diff(a, b)
                    if d > 0:
                        any_diff = True
                    max_ulp = max(max_ulp, d)
            n_graphs_diff += 1 if any_diff else 0

        out["modes"][mode] = {
            "distinct_aggregate_hashes_across_3_processes": distinct,
            "cross_process_bitwise_reproducible": distinct == 1,
            "max_ulp_diff_across_processes": max_ulp,
            "graphs_with_any_ulp_diff": n_graphs_diff,
            "n_graphs": len(rs[0]["grads"]),
            "within_process_rebuild_identical": all(
                r["within_process_rebuild_identical"] for r in rs
            ),
            "elapsed_s_median": sorted(r["elapsed_s"] for r in rs)[1],
            "probe_wall_s_median": sorted(r["wall_s"] for r in rs)[1],
        }

    # pairwise set-vs-ordered deviation
    ordered0 = runs["ordered"][0]["grads"]
    max_cross = 0.0
    for r in runs["set"]:
        for go, gs in zip(ordered0, r["grads"]):
            for a, b in zip(go, gs):
                max_cross = max(max_cross, ulp_diff(a, b))
    out["set_vs_ordered_max_ulp"] = max_cross

    common.save_result("e5_determinism", out)
    for mode in MODES:
        m = out["modes"][mode]
        print(
            "E5 %-8s: distinct hashes %d/3, cross-process reproducible=%s, max ulp %.1f, graphs differing %d/60"
            % (mode, m["distinct_aggregate_hashes_across_3_processes"],
               m["cross_process_bitwise_reproducible"], m["max_ulp_diff_across_processes"],
               m["graphs_with_any_ulp_diff"])
        )
    print(f"E5 set-vs-ordered max ulp deviation: {max_cross:.1f}")
    return out


if __name__ == "__main__":
    main()
