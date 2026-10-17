# Data provenance

All datasets in this directory are real, public, and redistributable snapshots
fetched on **2026-10-17 (UTC)**. Re-fetch at any time with `python scripts/fetch_data.py`.

| File | Source | Endpoint | Range | Rows |
|---|---|---|---|---|
| `btc_usd_daily_raw.json` | CoinGecko (keyless public API) | `api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365&interval=daily` | 2025-10-18 → 2026-10-17 | 366 daily closes |
| `eth_usd_daily_raw.json` | CoinGecko (keyless public API) | same, `coins/ethereum` | 2025-10-18 → 2026-10-17 | 366 daily closes |
| `fx_eur_daily_raw.json` | European Central Bank via Frankfurter | `api.frankfurter.app/2024-10-01..2026-10-17?base=EUR&symbols=USD,JPY` | 2024-10-01 → 2026-10-16 | 522 business days |

## Live cross-validation (2026-10-17)

Snapshot endpoints were verified against independent live sources at build time:

| Series | Snapshot (last) | Live source | Value | Agreement |
|---|---|---|---|---|
| BTC/USD | 81,417.97 | openresearch MCP (CoinGecko) | 81,418.00 | to the dollar |
| EUR/USD | 1.1253 (2026-10-16) | openresearch MCP (ECB) | 1.1253 | exact |

## License / terms

- CoinGecko public API data: free for non-commercial use, attribution required.
- ECB reference rates via Frankfurter: public domain (ECB data reuse policy).

No personally identifying information. These snapshots are inputs to a
*methodological* verification study — not investment research.
