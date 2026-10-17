"""E4: Verification against real market data (frozen live snapshots).

For each dataset (BTC/USD, ETH/USD, EUR/USD, EUR/JPY daily closes fetched
2026-10-17; see data/SOURCES.md):

1. Validate the loaded snapshot against the recorded fetch-time values.
2. Build a leak-free supervised task: features = lagged log-returns + rolling
   volatility known at t-1; target = sign of the t-to-t+1 return.
3. Train a small MLP with the scalar engine (chronological 80/20 split;
   train-only feature standardization).
4. Gradient integrity: finite-difference gradient check of the training loss
   on real rows w.r.t. every parameter.
5. Honest evaluation: test accuracy vs majority baseline, AUC, binomial test.
   (This is a methodology demonstration, not investment advice; the expected
   outcome is close to the majority baseline.)
"""

from __future__ import annotations

import random

import common  # noqa: F401
from micrograd_verified.data import (
    chronological_split,
    load_btc_prices,
    load_eth_prices,
    load_fx_series,
    make_supervised,
    standardize,
)
from micrograd_verified.engine import Value
from micrograd_verified.gradcheck import gradcheck
from micrograd_verified.metrics import accuracy, binomial_z_test, majority_baseline, roc_auc
from micrograd_verified.nn import MLP

STEPS = 150
BATCH = 64
LR = 0.05
N_LAGS = 3
VOL_WINDOW = 5
GRADCHECK_ROWS = 8

VALIDATION = {
    "btc_usd": {"n": 365, "last": 81417.97, "tol": 0.01},
    "eth_usd": {"n": 365, "last": 2414.12, "tol": 0.01},
    "eur_usd": {"n": 522, "last": 1.1253, "tol": 0.005},
    "eur_jpy": {"n": 522, "last": 177.82, "tol": 0.005},
}


def run_dataset(name, prices, seed):
    ref = VALIDATION[name]
    validation = {
        "n_points": len(prices),
        "first_day": str(prices[0][0]),
        "last_day": str(prices[-1][0]),
        "last_close": prices[-1][1],
        "expected_n": ref["n"],
        "n_ok": len(prices) == ref["n"],
    }
    if ref["last"] is not None:
        rel = abs(prices[-1][1] - ref["last"]) / abs(ref["last"])
        validation.update({"expected_last": ref["last"], "rel_diff": rel, "value_ok": rel <= ref["tol"]})

    feats, targets, _days = make_supervised(prices, n_lags=N_LAGS, vol_window=VOL_WINDOW)
    tr_x, tr_y, te_x, te_y = chronological_split(feats, targets, 0.8)
    tr_x_s, te_x_s = standardize(tr_x, te_x)

    mlp = MLP(len(tr_x[0]), [8, 8, 1], seed=seed)

    def train_once():
        model = MLP(len(tr_x[0]), [8, 8, 1], seed=seed)
        rng = random.Random(seed)
        losses = []
        for _ in range(STEPS):
            batch = rng.sample(range(len(tr_x_s)), min(BATCH, len(tr_x_s)))
            loss = sum(((model(tr_x_s[i]) - tr_y[i]) ** 2 for i in batch), Value(0.0))
            loss = loss * (1.0 / len(batch))
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= LR * p.grad
            losses.append(loss.data)
        return model, losses

    mlp, losses = train_once()
    _mlp2, losses2 = train_once()
    determinism = {"loss_first": losses[0], "loss_last": losses[-1], "rerun_loss_last": losses2[-1],
                   "bitwise_identical": losses[-1] == losses2[-1]}

    # --- gradient integrity on real rows (all parameters, FD ground truth)
    params = mlp.parameters()

    def loss_fn(params=params, rows=GRADCHECK_ROWS):
        return sum(((mlp(tr_x_s[i]) - tr_y[i]) ** 2 for i in range(rows)), Value(0.0))

    gc = gradcheck(loss_fn, params, eps=1e-6)
    grad_integrity = {
        "n_params": len(params),
        "rows": GRADCHECK_ROWS,
        "passed": gc["passed"],
        "max_abs_error": gc["max_abs_error"],
        "max_rel_error": gc["max_rel_error"],
        "worst_param": max(gc["per_input"], key=lambda d: d["rel_error"])["data"],
    }

    # --- evaluation
    te_pred = [mlp(x).data for x in te_x_s]
    tr_pred = [mlp(x).data for x in tr_x_s]
    acc_test = accuracy(te_pred, te_y)
    n_test = len(te_y)
    hits = sum(1 for p, t in zip(te_pred, te_y) if (p > 0) == (t > 0))
    z, pval = binomial_z_test(hits, n_test, 0.5)
    evaluation = {
        "n_train": len(tr_y),
        "n_test": n_test,
        "train_accuracy": accuracy(tr_pred, tr_y),
        "test_accuracy": acc_test,
        "majority_baseline_test": majority_baseline(te_y),
        "test_auc": roc_auc(te_pred, te_y),
        "binomial_z": z,
        "binomial_p": pval,
        "positive_rate_test": sum(1 for t in te_y if t > 0) / n_test,
    }

    return {
        "validation": validation,
        "determinism": determinism,
        "grad_integrity": grad_integrity,
        "evaluation": evaluation,
        "loss_curve_sample": losses[::15],
    }


def main():
    datasets = {
        "btc_usd": load_btc_prices(),
        "eth_usd": load_eth_prices(),
        "eur_usd": load_fx_series("USD"),
        "eur_jpy": load_fx_series("JPY"),
    }
    out = {"config": {"steps": STEPS, "batch": BATCH, "lr": LR, "n_lags": N_LAGS,
                      "vol_window": VOL_WINDOW, "model": [8, 8, 1]},
           "datasets": {}}
    for i, (name, prices) in enumerate(datasets.items()):
        res = run_dataset(name, prices, seed=100 + i)
        out["datasets"][name] = res
        v, e, g = res["validation"], res["evaluation"], res["grad_integrity"]
        print(
            "E4 %-8s n=%d last=%.5g | gradcheck=%s (rel %.1e) | test acc %.3f vs baseline %.3f (p=%.3f)"
            % (name, v["n_points"], v["last_close"], g["passed"], g["max_rel_error"],
               e["test_accuracy"], e["majority_baseline_test"], e["binomial_p"])
        )
        assert v["n_ok"], f"{name}: snapshot length mismatch"
        if "value_ok" in v:
            assert v["value_ok"], f"{name}: snapshot value drift vs recorded live fetch"
        assert g["passed"], f"{name}: gradient check failed on real data"
        assert res["determinism"]["bitwise_identical"], f"{name}: training not deterministic"
    common.save_result("e4_market_data", out)
    return out


if __name__ == "__main__":
    main()
