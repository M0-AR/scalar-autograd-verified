"""Numerical gradient checking utilities.

Implements the standard two-sided finite-difference check used by PyTorch
(``torch.autograd.gradcheck``) and JAX (``check_grads``): a pass requires

    |analytic - numeric| <= atol + rtol * |numeric|

with defaults eps=1e-6, atol=1e-5, rtol=1e-3, matching PyTorch 2.x.

Also provides the raw error instrumentation used by experiment E1 to study the
truncation-vs-roundoff trade-off: forward and central differences, an epsilon
sweep, and log-log slope fitting.
"""

from __future__ import annotations

import math

from .engine import Value

__all__ = [
    "forward_difference",
    "central_difference",
    "gradcheck",
    "max_abs_error",
    "max_rel_error",
    "epsilon_sweep",
    "fit_loglog_slope",
]


def forward_difference(fn, inputs, i: int, eps: float = 1e-6) -> float:
    """(f(x + eps e_i) - f(x)) / eps using the current values of ``inputs``."""
    v = inputs[i]
    orig = v.data
    v.data = orig + eps
    f_plus = fn().data
    v.data = orig
    return (f_plus - fn().data) / eps


def central_difference(fn, inputs, i: int, eps: float = 1e-6) -> float:
    """(f(x + eps e_i) - f(x - eps e_i)) / (2 eps)."""
    v = inputs[i]
    orig = v.data
    v.data = orig + eps
    f_plus = fn().data
    v.data = orig - eps
    f_minus = fn().data
    v.data = orig
    return (f_plus - f_minus) / (2.0 * eps)


def analytic_grads(fn, inputs) -> float:
    """Run one backward pass and return the output value and input gradients."""
    for v in inputs:
        v.grad = 0.0
    out = fn()
    out.backward()
    return out.data, [v.grad for v in inputs]


def max_abs_error(analytic, numeric) -> float:
    return max(abs(a - n) for a, n in zip(analytic, numeric))


def max_rel_error(analytic, numeric, floor: float = 1e-8) -> float:
    err = 0.0
    for a, n in zip(analytic, numeric):
        denom = max(floor, abs(a), abs(n))
        err = max(err, abs(a - n) / denom)
    return err


def gradcheck(
    fn,
    inputs,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    scheme: str = "central",
):
    """Check analytic gradients against finite differences.

    Returns a dict with per-input analytic/numeric gradients and pass flags,
    plus aggregate pass/fail under the PyTorch criterion.
    """
    diff = central_difference if scheme == "central" else forward_difference
    fval, analytic = analytic_grads(fn, inputs)
    numeric = [diff(fn, inputs, i, eps) for i in range(len(inputs))]

    per_input = []
    for v, a, n in zip(inputs, analytic, numeric):
        slack = atol + rtol * abs(n)
        per_input.append(
            {
                "data": v.data,
                "analytic": a,
                "numeric": n,
                "abs_error": abs(a - n),
                "rel_error": abs(a - n) / max(1e-8, abs(a), abs(n)),
                "tolerance": slack,
                "passed": abs(a - n) <= slack,
            }
        )
    return {
        "output": fval,
        "eps": eps,
        "scheme": scheme,
        "atol": atol,
        "rtol": rtol,
        "per_input": per_input,
        "passed": all(p["passed"] for p in per_input),
        "max_abs_error": max_abs_error(analytic, numeric),
        "max_rel_error": max_rel_error(analytic, numeric),
    }


def epsilon_sweep(f_scalar, x0: float, exact: float, epsilons, schemes=("forward", "central")):
    """Absolute error of finite-difference estimates of f'(x0) over a sweep of eps.

    ``f_scalar`` is a plain float->float function; ``exact`` the analytic slope.
    """
    results = {}
    for scheme in schemes:
        errs = []
        for eps in epsilons:
            if scheme == "forward":
                est = (f_scalar(x0 + eps) - f_scalar(x0)) / eps
            else:
                est = (f_scalar(x0 + eps) - f_scalar(x0 - eps)) / (2.0 * eps)
            errs.append(abs(est - exact))
        results[scheme] = errs
    return results


def fit_loglog_slope(xs, ys) -> tuple[float, float]:
    """Least-squares slope/intercept of log10(y) vs log10(x); returns (slope, r2)."""
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ly)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(lx, ly))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, r2
