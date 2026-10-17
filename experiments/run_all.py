"""Run all experiments E1-E6 in order, then build results/RESULTS.md."""

from __future__ import annotations

import time

import common  # noqa: F401


def main():
    import e1_finite_difference
    import e2_random_dag_gradcheck
    import e3_aliasing_bug
    import e4_market_data
    import e5_determinism
    import e6_benchmarks
    from micrograd_verified import report

    t0 = time.perf_counter()
    for mod in (
        e1_finite_difference,
        e2_random_dag_gradcheck,
        e3_aliasing_bug,
        e4_market_data,
        e5_determinism,
        e6_benchmarks,
    ):
        name = mod.__name__
        print(f"\n=== {name} ===")
        mod.main()
    summary = report.main()
    print(f"\nAll experiments completed in {time.perf_counter() - t0:.1f}s")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
