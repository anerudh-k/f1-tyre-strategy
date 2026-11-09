"""Per-compound tyre degradation models."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class DegModel:
    compound: str
    kind: str              # "linear" or "quadratic"
    coefs: list            # [c0, c1, (c2)] for delta = c0 + c1*age + c2*age^2
    rmse: float
    n_laps: int
    max_age_seen: int
    cliff_age: float | None = None

    def predict(self, age):
        age = np.asarray(age, dtype=float)
        c = list(self.coefs) + [0.0] * (3 - len(self.coefs))
        return c[0] + c[1] * age + c[2] * age ** 2

    @property
    def deg_per_lap(self) -> float:
        """Average slope over a typical 20-lap stint (s/lap)."""
        return float((self.predict(20) - self.predict(1)) / 19)


def _cv_rmse(x, y, deg, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(x))
    errs = []
    for f in np.array_split(idx, k):
        tr = np.setdiff1d(idx, f)
        p = np.polyfit(x[tr], y[tr], deg)
        errs.append(np.mean((np.polyval(p, x[f]) - y[f]) ** 2))
    return float(np.sqrt(np.mean(errs)))


def find_cliff(age, delta, min_seg: int = 5):
    """Two-segment piecewise-linear fit; returns the breakpoint age if the
    second slope is clearly steeper (>2x) than the first, else None."""
    age, delta = np.asarray(age, float), np.asarray(delta, float)
    ages = np.unique(age)
    if len(ages) < 2 * min_seg:
        return None
    best = (np.inf, None, None)
    for bp in ages[min_seg:-min_seg]:
        X = np.column_stack([np.ones_like(age), age, np.clip(age - bp, 0, None)])
        coef, *_ = np.linalg.lstsq(X, delta, rcond=None)
        sse = np.sum((X @ coef - delta) ** 2)
        if sse < best[0]:
            best = (sse, bp, coef)
    _, bp, coef = best
    s1, s2 = coef[1], coef[1] + coef[2]
    return float(bp) if s2 > max(2 * s1, s1 + 0.05) else None


def fit_compound(df: pd.DataFrame, compound: str) -> DegModel:
    d = df[df["Compound"] == compound]
    x, y = d["TyreLife"].to_numpy(float), d["Delta"].to_numpy(float)
    lin, quad = _cv_rmse(x, y, 1), _cv_rmse(x, y, 2)
    # Prefer linear unless quadratic is meaningfully better (>2%) and curvature is positive.
    pq = np.polyfit(x, y, 2)
    use_quad = quad < 0.98 * lin and pq[0] > 0
    if use_quad:
        coefs, kind, rmse = [pq[2], pq[1], pq[0]], "quadratic", quad
    else:
        pl = np.polyfit(x, y, 1)
        coefs, kind, rmse = [pl[1], pl[0]], "linear", lin
    return DegModel(compound, kind, [float(c) for c in coefs], rmse, len(d),
                    int(x.max()), find_cliff(x, y))


def fit_all(df: pd.DataFrame, min_laps: int = 30) -> dict[str, DegModel]:
    out = {}
    for comp, g in df.groupby("Compound"):
        if len(g) >= min_laps:
            out[comp] = fit_compound(df, comp)
    return out


def summary_table(models: dict[str, DegModel]) -> pd.DataFrame:
    rows = [dict(Compound=m.compound, Model=m.kind, PaceOffset_s=round(m.predict(1).item(), 3),
                 Deg_s_per_lap=round(m.deg_per_lap, 4), CV_RMSE_s=round(m.rmse, 3),
                 Laps=m.n_laps, MaxAgeSeen=m.max_age_seen, CliffAge=m.cliff_age)
            for m in models.values()]
    return pd.DataFrame(rows).sort_values("PaceOffset_s").reset_index(drop=True)


def save_models(models: dict[str, DegModel], path: str, meta: dict | None = None):
    with open(path, "w") as f:
        json.dump({"meta": meta or {}, "models": {k: asdict(v) for k, v in models.items()}},
                  f, indent=2)


def load_models(path: str):
    with open(path) as f:
        raw = json.load(f)
    return {k: DegModel(**v) for k, v in raw["models"].items()}, raw.get("meta", {})
