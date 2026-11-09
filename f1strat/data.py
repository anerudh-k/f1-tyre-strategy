"""Loading and cleaning race lap data from FastF1."""
from __future__ import annotations

import numpy as np
import pandas as pd

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def load_race(year: int, gp: str | int, cache_dir: str = "data/cache"):
    """Load a race session with FastF1 and return (session, laps)."""
    import os
    import fastf1

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, gp, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    return session, session.laps


def clean_laps(laps: pd.DataFrame, outlier_pct: float = 1.07) -> pd.DataFrame:
    """Keep representative green-flag laps on dry tyres.

    Removes: lap 1, in/out laps, non-green laps (SC/VSC/yellow), laps FastF1
    marks as inaccurate, wet/inter tyres, and slow outliers (> outlier_pct of
    the driver's median, which catches traffic and mistakes).
    """
    df = pd.DataFrame(laps).copy()
    df = df[df["LapTime"].notna() & (df["LapNumber"] > 1)]
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    df = df[df["TrackStatus"].astype(str) == "1"]
    if "IsAccurate" in df:
        df = df[df["IsAccurate"].fillna(False).astype(bool)]
    df = df[df["Compound"].isin(DRY_COMPOUNDS)]

    df["LapTimeS"] = df["LapTime"].dt.total_seconds()
    med = df.groupby("Driver")["LapTimeS"].transform("median")
    df = df[df["LapTimeS"] < outlier_pct * med]

    df["TyreLife"] = df["TyreLife"].astype(float)
    cols = ["Driver", "Team", "LapNumber", "Stint", "Compound", "TyreLife", "LapTimeS"]
    return df[[c for c in cols if c in df]].reset_index(drop=True)


def fuel_correct(df: pd.DataFrame, total_laps: int, fuel_start_kg: float = 100.0,
                 sec_per_kg: float = 0.03) -> pd.DataFrame:
    """Remove the fuel-burn effect so lap-time change reflects tyre wear only.

    Assumes linear fuel burn from fuel_start_kg to ~0 at the flag. The
    defaults are typical public estimates; treat them as assumptions and
    run a sensitivity check.
    """
    df = df.copy()
    fuel_left = fuel_start_kg * (1 - (df["LapNumber"] - 1) / total_laps)
    df["FuelKg"] = fuel_left
    df["LapTimeCorr"] = df["LapTimeS"] - sec_per_kg * fuel_left
    # Per-driver baseline so drivers of different pace can be pooled.
    df["Delta"] = df["LapTimeCorr"] - df.groupby("Driver")["LapTimeCorr"].transform(
        lambda s: s.quantile(0.10))
    return df


def pit_loss_from_session(laps: pd.DataFrame) -> float:
    """Rough pit loss estimate: median (in-lap + out-lap) minus 2x normal lap.

    Falls back to NaN if not enough clean stops. Check against known values
    for the track before trusting it.
    """
    df = pd.DataFrame(laps).copy()
    df["LapTimeS"] = df["LapTime"].dt.total_seconds()
    green = df[(df["TrackStatus"].astype(str) == "1") & df["PitInTime"].isna()
               & df["PitOutTime"].isna()]
    normal = green.groupby("Driver")["LapTimeS"].median()

    losses = []
    for drv, g in df.sort_values("LapNumber").groupby("Driver"):
        g = g.set_index("LapNumber")
        for lap in g.index[g["PitInTime"].notna()]:
            if lap + 1 in g.index and drv in normal:
                pair = g.loc[[lap, lap + 1]]
                if (pair["TrackStatus"].astype(str) == "1").all():
                    losses.append(pair["LapTimeS"].sum() - 2 * normal[drv])
    losses = [x for x in losses if 10 < x < 40]
    return float(np.median(losses)) if len(losses) >= 3 else float("nan")


def make_synthetic_laps(total_laps: int = 57, n_drivers: int = 12, seed: int = 0) -> pd.DataFrame:
    """FastF1-shaped fake race for testing the pipeline offline."""
    rng = np.random.default_rng(seed)
    true = {"SOFT": (0.0, 0.085, 0.0030), "MEDIUM": (0.45, 0.050, 0.0010),
            "HARD": (0.85, 0.030, 0.0003)}
    plans = [[("MEDIUM", 20), ("HARD", None)], [("SOFT", 14), ("HARD", None)],
             [("MEDIUM", 18), ("HARD", 20), ("SOFT", None)], [("HARD", 28), ("MEDIUM", None)]]
    rows = []
    for d in range(n_drivers):
        pace = 92.0 + rng.normal(0, 0.4)
        plan = plans[d % len(plans)]
        lap, stint = 1, 1
        for i, (comp, length) in enumerate(plan):
            length = length or (total_laps - lap + 1)
            length = min(length, total_laps - lap + 1)
            c0, c1, c2 = true[comp]
            for age in range(1, length + 1):
                fuel = 100 * (1 - (lap - 1) / total_laps)
                t = pace + c0 + c1 * age + c2 * age ** 2 + 0.03 * fuel + rng.normal(0, 0.25)
                pit_in = age == length and i < len(plan) - 1
                pit_out = age == 1 and i > 0
                if pit_in:
                    t += 17
                if pit_out:
                    t += 5
                if lap == 1:
                    t += 6
                rows.append(dict(Driver=f"D{d:02d}", Team=f"T{d//2}", LapNumber=lap, Stint=stint,
                                 Compound=comp, TyreLife=float(age),
                                 LapTime=pd.to_timedelta(t, unit="s"),
                                 PitInTime=pd.Timedelta(1, "s") if pit_in else pd.NaT,
                                 PitOutTime=pd.Timedelta(1, "s") if pit_out else pd.NaT,
                                 TrackStatus="4" if 30 <= lap <= 33 else "1", IsAccurate=True))
                lap += 1
            stint += 1
    return pd.DataFrame(rows)
