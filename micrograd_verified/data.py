"""Real-market data loaders and supervised-task construction.

Datasets are the frozen live snapshots in ``data/`` (see data/SOURCES.md),
fetched from CoinGecko and the ECB (via Frankfurter) on 2026-10-17.

Task construction is deliberately leak-free:
* features at time ``t`` use information up to and including ``t-1`` only;
* the target is the sign of the return from ``t-1`` to ``t``;
* splits are chronological (train = earliest 80%, test = latest 20%);
* feature standardization uses *train* statistics only.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

__all__ = [
    "load_btc_prices",
    "load_eth_prices",
    "load_fx_series",
    "log_returns",
    "make_supervised",
    "chronological_split",
    "standardize",
]


def _read_json(name: str) -> dict:
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def _coingecko_prices(fname: str):
    raw = _read_json(fname)["prices"]
    out = []
    for ts_ms, price in raw:
        day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
        out.append((day, float(price)))
    # CoinGecko may return two points for the same calendar day; keep the last.
    dedup = {}
    for day, price in out:
        dedup[day] = price
    return sorted(dedup.items())


def load_btc_prices():
    return _coingecko_prices("btc_usd_daily_raw.json")


def load_eth_prices():
    return _coingecko_prices("eth_usd_daily_raw.json")


def load_fx_series(quote: str = "USD"):
    """ECB daily reference rates, base EUR. ``quote`` in {"USD", "JPY"}."""
    rates = _read_json("fx_eur_daily_raw.json")["rates"]
    days = sorted(rates)
    return [(datetime.fromisoformat(d).date(), float(rates[d][quote])) for d in days]


def log_returns(values: list[float]) -> list[float]:
    """r_t = ln(p_t / p_{t-1}) with r_0 = 0.0 (padding to keep alignment)."""
    out = [0.0]
    for a, b in zip(values, values[1:]):
        out.append(math.log(b / a))
    return out


def make_supervised(prices, n_lags: int = 3, vol_window: int = 5):
    """Build (features, target, dates) with features known at t-1 and target at t.

    features: [r_{t-1}, ..., r_{t-n_lags}, rolling_std(vol_window) of returns up to t-1]
    target:   +1.0 if r_t > 0 else -1.0
    """
    days = [d for d, _ in prices]
    closes = [p for _, p in prices]
    rets = log_returns(closes)

    rows, targets, row_days = [], [], []
    for t in range(max(n_lags, vol_window) + 1, len(rets)):
        feats = [rets[t - k] for k in range(1, n_lags + 1)]
        window = rets[t - vol_window : t]
        mu = sum(window) / len(window)
        vol = math.sqrt(sum((w - mu) ** 2 for w in window) / len(window))
        feats.append(vol)
        rows.append(feats)
        targets.append(1.0 if rets[t] > 0 else -1.0)
        row_days.append(days[t])
    return rows, targets, row_days


def chronological_split(features, targets, train_frac: float = 0.8):
    n_train = int(len(features) * train_frac)
    return (
        features[:n_train],
        targets[:n_train],
        features[n_train:],
        targets[n_train:],
    )


def standardize(train_rows, *other_rows):
    """Standardize columns using train mean/std (std floored at 1e-12)."""
    n_cols = len(train_rows[0])
    means = [sum(row[j] for row in train_rows) / len(train_rows) for j in range(n_cols)]
    stds = []
    for j in range(n_cols):
        mu = means[j]
        var = sum((row[j] - mu) ** 2 for row in train_rows) / len(train_rows)
        stds.append(max(math.sqrt(var), 1e-12))

    def scale(rows):
        return [[(row[j] - means[j]) / stds[j] for j in range(n_cols)] for row in rows]

    return (scale(train_rows),) + tuple(scale(r) for r in other_rows)
