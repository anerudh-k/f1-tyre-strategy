"""Pit strategy optimisation on top of the degradation models."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from .degradation import DegModel


@dataclass
class RaceSetup:
    total_laps: int
    pit_loss: float = 21.0
    pit_loss_sc: float = 11.0
    min_stint: int = 5
    stint_cap_factor: float = 1.25
    models: dict[str, DegModel] = field(default_factory=dict)

    def __post_init__(self):
        self._cum = {}
        for c, m in self.models.items():
            ages = np.arange(1, self.total_laps + 1)
            self._cum[c] = np.concatenate([[0.0], np.cumsum(m.predict(ages))])

    def max_stint(self, c: str) -> int:
        return int(min(self.total_laps, np.ceil(self.stint_cap_factor * self.models[c].max_age_seen)))

    def stint_time(self, c: str, n: int) -> float:
        return float(self._cum[c][n])


def _stints(pit_laps, total):
    edges = [0, *pit_laps, total]
    return [b - a for a, b in zip(edges[:-1], edges[1:])]


def race_time(setup: RaceSetup, compounds, pit_laps, sc_laps: set | None = None) -> float:
    """Total race time relative to baseline pace; pit_laps = last lap of each stint."""
    lens = _stints(pit_laps, setup.total_laps)
    t = sum(setup.stint_time(c, n) for c, n in zip(compounds, lens))
    for lap in pit_laps:
        t += setup.pit_loss_sc if sc_laps and lap in sc_laps else setup.pit_loss
    return t


def _valid(setup, compounds, lens):
    if len(set(compounds)) < 2:
        return False
    return all(setup.min_stint <= n <= setup.max_stint(c) for c, n in zip(compounds, lens))


def enumerate_strategies(setup: RaceSetup, max_stops: int = 2) -> pd.DataFrame:
    """Brute-force every 1- and 2-stop plan; returns them sorted by race time."""
    comps = list(setup.models)
    L = setup.total_laps
    rows = []
    for stops in range(1, max_stops + 1):
        for seq in product(comps, repeat=stops + 1):
            if len(set(seq)) < 2:
                continue
            if stops == 1:
                laps_iter = ((p,) for p in range(1, L))
            else:
                laps_iter = ((p1, p2) for p1 in range(1, L) for p2 in range(p1 + 1, L))
            for pits in laps_iter:
                lens = _stints(pits, L)
                if _valid(setup, seq, lens):
                    rows.append(dict(Stops=stops, Compounds="-".join(c[0] for c in seq),
                                     Seq=seq, PitLaps=pits, Stints=tuple(lens),
                                     RaceTime=race_time(setup, seq, pits)))
    df = pd.DataFrame(rows).sort_values("RaceTime").reset_index(drop=True)
    df["Gap"] = df["RaceTime"] - df["RaceTime"].iloc[0]
    return df


def best_per_sequence(strats: pd.DataFrame) -> pd.DataFrame:
    return strats.loc[strats.groupby("Compounds")["RaceTime"].idxmin()].sort_values(
        "RaceTime").reset_index(drop=True)


def sample_safety_cars(total_laps, rng, p_sc=0.5, dur=(3, 5)):
    """Return the set of SC laps for one simulated race (at most one SC)."""
    if rng.random() > p_sc:
        return set()
    start = int(rng.integers(2, total_laps - 3))
    n = int(rng.integers(dur[0], dur[1] + 1))
    return set(range(start, min(start + n, total_laps)))


def _react(setup, seq, pits, sc, window):
    """If a planned stop is within `window` laps of a safety car, move it to the first SC lap."""
    if not sc:
        return pits
    first = min(sc)
    new = list(pits)
    for i, p in enumerate(pits):
        if abs(p - first) <= window:
            cand = sorted(new[:i] + [first] + new[i + 1:])
            if len(set(cand)) == len(cand) and _valid(setup, seq, _stints(cand, setup.total_laps)):
                new = cand
            break
    return tuple(new)


def monte_carlo(setup: RaceSetup, candidates: pd.DataFrame, n_sims: int = 5000,
                p_sc: float = 0.5, window: int = 6, seed: int = 0, wear_sd: float = 0.0):
    """Simulate candidate plans under random safety cars and uncertain tyre wear."""
    rng = np.random.default_rng(seed)
    cand = candidates.reset_index(drop=True)
    times = np.empty((n_sims, len(cand)))
    comps = list(setup.models)
    for s in range(n_sims):
        sc = sample_safety_cars(setup.total_laps, rng, p_sc)
        shift = dict(zip(comps, rng.normal(0, wear_sd, len(comps)))) if wear_sd > 0 else None
        for j, r in cand.iterrows():
            pits = _react(setup, r.Seq, r.PitLaps, sc, window)
            times[s, j] = race_time(setup, r.Seq, pits, sc)
            if shift:
                times[s, j] += sum(shift[c] * n * (n + 1) / 2
                                   for c, n in zip(r.Seq, _stints(pits, setup.total_laps)))
    best = times <= times.min(1, keepdims=True) + 1e-9
    wins = (best / best.sum(1, keepdims=True)).mean(0)
    summ = cand[["Compounds", "PitLaps", "RaceTime"]].copy()
    summ = summ.rename(columns={"RaceTime": "NoSC_Time"})
    summ["MeanTime"] = times.mean(0)
    summ["P10"] = np.percentile(times, 10, axis=0)
    summ["P90"] = np.percentile(times, 90, axis=0)
    summ["WinRate"] = wins
    return summ, times
