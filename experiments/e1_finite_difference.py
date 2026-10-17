"""E1: Finite-difference error laws — truncation vs roundoff.

Measure the forward and central difference error for f'(x) over 61 epsilons
spanning 1e-16..1e-1 and fit power laws in the two asymptotic regimes:

    E_forward(h) ~ a1*h + b/h        (slope +1 truncation, -1 roundoff)
    E_central(h) ~ a2*h^2 + b/h      (slope +2 truncation, -1 roundoff)

Predictions are compared against closed-form optima derived from the analytic
derivatives of f and IEEE-754 double precision eps_m = 2^-52.

f(x) = exp(sin x) at x0 = 1.0.
  f'  = cos(x) e^{sin x}
  f'' = (cos^2 x - sin x) e^{sin x}
  f'''= (cos^3 x - 3 sin x cos x - cos x) e^{sin x}
"""

from __future__ import annotations

import math

import common  # noqa: F401  (sets sys.path)
from micrograd_verified.gradcheck import epsilon_sweep, fit_loglog_slope

X0 = 1.0
EPS_M = 2.0**-52


def f(x):
    return math.exp(math.sin(x))


def f1(x):
    return math.cos(x) * math.exp(math.sin(x))


def f2(x):
    c, s, e = math.cos(x), math.sin(x), math.exp(math.sin(x))
    return (c * c - s) * e


def f3(x):
    c, s, e = math.cos(x), math.sin(x), math.exp(math.sin(x))
    return (c**3 - 3 * c * s - c) * e


def main():
    n = 61
    lo, hi = -16.0, -1.0
    epsilons = [10 ** (lo + (hi - lo) * i / (n - 1)) for i in range(n)]

    exact = f1(X0)
    sweep = epsilon_sweep(f, X0, exact, epsilons)

    out = {
        "f": "exp(sin(x))",
        "x0": X0,
        "exact_slope": exact,
        "derivatives": {"f": f(X0), "f1": f1(X0), "f2": f2(X0), "f3": f3(X0)},
        "eps_machine": EPS_M,
        "epsilons": epsilons,
        "forward": {"errors": sweep["forward"]},
        "central": {"errors": sweep["central"]},
    }

    def seg_slope(errs, lo_e, hi_e):
        xs = [e for e in epsilons if lo_e <= e <= hi_e]
        ys = [err for e, err in zip(epsilons, errs) if lo_e <= e <= hi_e]
        slope, r2 = fit_loglog_slope(xs, ys)
        return {"lo": lo_e, "hi": hi_e, "slope": slope, "r2": r2, "n": len(xs)}

    for scheme, trunc_lo in (("forward", 1e-5), ("central", 1e-4)):
        errs = out[scheme]["errors"]
        out[scheme]["fit_roundoff"] = seg_slope(errs, 1e-16, 1e-12)
        out[scheme]["fit_truncation"] = seg_slope(errs, trunc_lo, 1e-1)
        i_opt = min(range(n), key=lambda i: errs[i])
        out[scheme]["measured_optimal"] = {"eps": epsilons[i_opt], "error": errs[i_opt]}

    # Closed-form optima.
    #   forward:  E = (|f2|/2) h + (2 eps_m |f|)/h  -> h* = sqrt(b/a)
    #   central:  E = (|f3|/6) h^2 + (2 eps_m (|f|+|x0 f1|))/h -> h* = (b/(2a))^{1/3}
    a1 = abs(f2(X0)) / 2.0
    b1 = 2 * EPS_M * abs(f(X0))
    a2 = abs(f3(X0)) / 6.0
    b2 = 2 * EPS_M * (abs(f(X0)) + abs(X0) * abs(f1(X0)))
    h1 = math.sqrt(b1 / a1)
    h2 = (b2 / (2 * a2)) ** (1.0 / 3.0)
    out["predicted_optimal"] = {
        "forward": {"eps": h1, "error": a1 * h1 + b1 / h1, "a": a1, "b": b1},
        "central": {"eps": h2, "error": a2 * h2**2 + b2 / h2, "a": a2, "b": b2},
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.loglog(epsilons, out["forward"]["errors"], "o-", ms=3, label="forward E(h)")
        ax.loglog(epsilons, out["central"]["errors"], "s-", ms=3, label="central E(h)")
        hh = [10 ** (lo + (hi - lo) * i / 200) for i in range(201)]
        ax.loglog(hh, [a1 * h + b1 / h for h in hh], "k--", lw=1, label=r"$a_1 h + b_1/h$")
        ax.loglog(hh, [a2 * h * h + b2 / h for h in hh], "k:", lw=1, label=r"$a_2 h^2 + b_2/h$")
        ax.axvline(h1, color="tab:blue", alpha=0.3)
        ax.axvline(h2, color="tab:orange", alpha=0.3)
        ax.set_xlabel(r"step size $h$")
        ax.set_ylabel("abs. error of f'(x0)")
        ax.set_title(r"$f(x)=e^{\sin x}$, $x_0=1$: FD error laws")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        fig.savefig(str(common.FIGDIR + "/e1_finite_difference.png"), dpi=150)
        out["figure"] = "results/figures/e1_finite_difference.png"
    except Exception as e:  # pragma: no cover
        out["figure_error"] = repr(e)

    common.save_result("e1_finite_difference", out)
    fwd, cen = out["forward"], out["central"]
    print(
        "E1: forward optimal h=%.3g (pred %.3g), central optimal h=%.3g (pred %.3g)"
        % (
            fwd["measured_optimal"]["eps"],
            h1,
            cen["measured_optimal"]["eps"],
            h2,
        )
    )
    print(
        "E1 slopes: fwd trunc %.2f (r2=%.4f), fwd roundoff %.2f; cen trunc %.2f (r2=%.4f), cen roundoff %.2f"
        % (
            fwd["fit_truncation"]["slope"],
            fwd["fit_truncation"]["r2"],
            fwd["fit_roundoff"]["slope"],
            cen["fit_truncation"]["slope"],
            cen["fit_truncation"]["r2"],
            cen["fit_roundoff"]["slope"],
        )
    )
    return out


if __name__ == "__main__":
    main()
