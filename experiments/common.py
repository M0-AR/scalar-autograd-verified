"""Shared helpers for experiments: paths, env metadata, result saving, timing."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for mod in ("numpy", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = m.__version__
        except Exception:
            info[mod] = None
    return info


def save_result(name: str, payload: dict) -> str:
    payload = {"experiment": name, "env": env_info(), **payload}
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {path}")
    return path


def load_result(name: str):
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def best_of(fn, repeats: int = 5) -> float:
    """Best-of-N wall time in seconds (mitigates scheduler noise)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best
