"""Re-fetch the live market snapshots into data/*.json.

Sources (keyless):
  * CoinGecko  /api/v3/coins/{id}/market_chart?vs_currency=usd&days=365&interval=daily
  * Frankfurter (ECB)  /2024-10-01..2026-10-16?base=EUR

Usage: python scripts/fetch_data.py
The repository ships with the frozen 2026-10-17 snapshots; run this to refresh.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "scalar-autograd-verified/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def fetch_coingecko(coin_id: str, fname: str, days: int = 365):
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days={days}")
    payload = get_json(url)
    out = {"source": "coingecko", "coin": coin_id, "vs": "usd", "days": days,
           "fetched_utc": date.today().isoformat(), "prices": payload["prices"]}
    path = os.path.join(DATA, fname)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[ok] {path}: {len(payload['prices'])} points")


def fetch_frankfurter(fname: str, start="2024-10-01", end=None):
    end = end or date.today().isoformat()
    url = f"https://api.frankfurter.dev/v1/{start}..{end}?base=EUR&symbols=USD,JPY"
    payload = get_json(url)
    out = {"source": "frankfurter (ECB reference rates)", "base": "EUR",
           "fetched_utc": date.today().isoformat(), "rates": payload["rates"]}
    path = os.path.join(DATA, fname)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[ok] {path}: {len(payload['rates'])} days")


def main():
    os.makedirs(DATA, exist_ok=True)
    fetch_coingecko("bitcoin", "btc_usd_daily_raw.json")
    fetch_coingecko("ethereum", "eth_usd_daily_raw.json")
    fetch_frankfurter("fx_eur_daily_raw.json")
    print("Note: update data/SOURCES.md validation values after a refresh.")


if __name__ == "__main__":
    main()
