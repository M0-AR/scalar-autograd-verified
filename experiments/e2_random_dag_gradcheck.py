"""E2: Gradient-check random DAGs at scale.

200 random expression graphs over 4 leaves, bucketed by operation count
({5, 10, 20, 40}). Each graph is checked with the PyTorch-default criterion
(central differences, h=1e-6, atol=1e-5, rtol=1e-3). Reports pass rates,
error distributions per depth bucket, and the growth rate of the worst
relative error with program depth.
"""

from __future__ import annotations

import random
import time

import common  # noqa: F401
from scalar_autograd_verified import graphgen
from scalar_autograd_verified.gradcheck import fit_loglog_slope, gradcheck

N_PER_BUCKET = 50
BUCKETS = (5, 10, 20, 40)
N_INPUTS = 4
BASE_SEED = 20261017


def percentile(xs, q):
    ys = sorted(xs)
    if not ys:
        return float("nan")
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def main():
    rng = random.Random(BASE_SEED)
    out = {"buckets": {}, "config": {"n_per_bucket": N_PER_BUCKET, "n_inputs": N_INPUTS}}
    all_pass = 0
    all_n = 0

    for n_ops in BUCKETS:
        rel, times = [], []
        n_pass = 0
        for _ in range(N_PER_BUCKET):
            inputs = graphgen.make_inputs(rng, N_INPUTS)
            spec = graphgen.random_spec(rng, n_ops=n_ops, n_inputs=N_INPUTS)

            def fn(inputs=inputs, spec=spec):
                out_v, _ = graphgen.build_from_spec(spec, inputs)
                return out_v

            t0 = time.perf_counter()
            res = gradcheck(fn, inputs)
            times.append(time.perf_counter() - t0)

            n_pass += 1 if res["passed"] else 0
            rel.append(res["max_rel_error"])
        stats = {
            "n": N_PER_BUCKET,
            "n_ops": n_ops,
            "pass_rate": n_pass / N_PER_BUCKET,
            "max_rel_error_median": percentile(rel, 0.5),
            "max_rel_error_p90": percentile(rel, 0.9),
            "max_rel_error_worst": max(rel),
            "seconds_median": percentile(times, 0.5),
        }
        out["buckets"][str(n_ops)] = stats
        all_pass += n_pass
        all_n += N_PER_BUCKET
        print(f"E2 n_ops={n_ops}: pass {n_pass}/{N_PER_BUCKET}, median rel err {stats['max_rel_error_median']:.2e}")

    out["overall_pass_rate"] = all_pass / all_n
    xs = [s["n_ops"] for s in out["buckets"].values()]
    ys = [s["max_rel_error_median"] for s in out["buckets"].values()]
    slope, r2 = fit_loglog_slope(xs, ys)
    out["error_growth_vs_depth"] = {
        "slope": slope,
        "r2": r2,
        "note": "log-log slope of median max-relative-error vs op count",
    }
    common.save_result("e2_random_dag_gradcheck", out)
    print(f"E2 overall pass rate: {out['overall_pass_rate']:.3f}; growth slope {slope:.3f} (r2 {r2:.3f})")
    return out


if __name__ == "__main__":
    main()
