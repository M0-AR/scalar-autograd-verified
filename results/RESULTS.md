# scalar-autograd-verified — experiment results
_Generated 2026-10-17T18:26:36.141310+00:00 · Python 3.12.13 · numpy 2.2.6 · torch 2.13.0+cu130_
All numbers below were produced by `make experiments` (`python experiments/run_all.py`). Raw data: `results/*.json`. Figures: `results/figures/`.
## E1 — Finite-difference error laws (f = exp(sin x), x0 = 1)
| scheme | truncation slope (r²) | roundoff slope (r²) | measured h* | meas. err(h*) | predicted h* |
|---|---|---|---|---|---|
| forward | 1.01 (1.0) | -0.89 (0.92) | 1.00e-08 | 9.45e-09 | 4.02e-08 |
| central | 2.0 (1.0) | -0.89 (0.77) | 3.16e-06 | 1.07e-11 | 1.06e-05 |

![E1](figures/e1_finite_difference.png)

## E2 — 200 random DAGs under PyTorch-default gradcheck
| ops | pass rate | median max-rel-err | p90 | worst |
|---|---|---|---|---|
| 5 | 100.0% | 4.70e-11 | 1.40e-10 | 4.23e-08 |
| 10 | 100.0% | 5.77e-11 | 1.49e-10 | 2.14e-09 |
| 20 | 100.0% | 8.83e-11 | 5.22e-10 | 7.95e-07 |
| 40 | 100.0% | 1.36e-10 | 4.83e-10 | 4.78e-08 |

Overall pass rate **100.0%**; error growth vs depth: log-log slope 0.522 (r² 0.977).

## E3 — Aliasing bug (assignment vs accumulation)
- Deterministic case `y = a + a`: correct grad **2.0**, buggy engine **1.0** (FD confirms 2.0).
- Random DAGs: **64.0%** produced wrong gradients (worst relative deviation 1.99); gradcheck caught **64.0%** of buggy graphs at PyTorch tolerances.

## E4 — Live market data (frozen 2026-10-17 snapshots)
| dataset | n | last close | gradcheck | test acc | baseline | AUC | p |
|---|---|---|---|---|---|---|---|
| btc_usd | 365 | 8.14e+04 | pass (rel 5.81e-08) | 0.486 | 0.514 | 0.438 | 0.814 |
| eth_usd | 365 | 2414.12284 | pass (rel 3.30e-08) | 0.486 | 0.569 | 0.421 | 0.814 |
| eur_usd | 522 | 1.1253 | pass (rel 1.34e-08) | 0.558 | 0.577 | 0.48 | 0.239 |
| eur_jpy | 522 | 177.82 | pass (rel 1.82e-07) | 0.51 | 0.577 | 0.56 | 0.845 |

Snapshot values cross-validated against the recorded live fetch (see `data/SOURCES.md`); gradient integrity holds on all datasets. As expected for daily sign prediction, test accuracy is statistically indistinguishable from the majority baseline — this is a verification methodology demo, not a trading strategy.

## E5 — Cross-process reproducibility: ordered tuple vs `set`
| mode | distinct hashes (3 runs) | reproducible | max ulp diff | graphs differing |
|---|---|---|---|---|
| ordered | 1/3 | True | 0.0 | 0/60 |
| set | 2/3 | False | 1.0 | 4/60 |

`set`-vs-`ordered` max deviation: 1.0 ulp.

## E6 — Performance
| chain L | micrograd | torch (scalar) | ratio micro/torch |
|---|---|---|---|
| 16 | 0.014 ms (893.0 ns/node) | 0.078 ms (4886.0 ns/node) | 0.18× |
| 64 | 0.042 ms (654.0 ns/node) | 0.235 ms (3675.0 ns/node) | 0.18× |
| 256 | 0.205 ms (799.0 ns/node) | 0.883 ms (3450.0 ns/node) | 0.23× |
| 1024 | 0.834 ms (815.0 ns/node) | 3.509 ms (3426.0 ns/node) | 0.24× |
| 4096 | 3.789 ms (925.0 ns/node) | 15.645 ms (3820.0 ns/node) | 0.24× |

- Random 2000-op DAG (build + fwd + bwd): 1.6 ms (0.78 µs/op)
- Chain-4096 memory: 2350.0 KiB peak (588.0 B/node, tracemalloc)
- MLP training throughput: 2232.0 steps/s (8929.0 samples/s)

