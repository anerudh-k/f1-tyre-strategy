"""Loading and cleaning race lap data from OpenF1 (https://openf1.org).

F1's own live-timing server (used by FastF1) returns 403 to cloud machines
such as Google Colab, so laps, stints, pit stops and race-control messages are
pulled from the free OpenF1 API instead and reshaped into FastF1-style columns.
"""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
OPENF1 = "https://api.openf1.org/v1"


class NoDataError(RuntimeError):
    """OpenF1 has no data for this session (cancelled, not run yet, or not published)."""


def _get(endpoint: str, cache_dir: str, retries: int = 5, **params) -> list:
    """GET an OpenF1 endpoint with on-disk JSON caching and retry on rate limits.

    A 4xx answer (other than 429 rate-limit) means "no data for this query" and
    returns []. Empty answers are not cached, so they are retried next run.
    """
    import requests

    key = endpoint + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    path = os.path.join(cache_dir, key.replace("/", "-").replace(" ", "") + ".json")
    # The current season's race list keeps growing: refresh it if older than 12 h
    stale = (endpoint in ("sessions", "meetings")
             and int(params.get("year", 0)) >= time.gmtime().tm_year
             and os.path.exists(path) and time.time() - os.path.getmtime(path) > 12 * 3600)
    if os.path.exists(path) and not stale:
        with open(path) as f:
            return json.load(f)
    code = None
    for attempt in range(retries):
        r = requests.get(f"{OPENF1}/{endpoint}", params=params, timeout=60)
        code = r.status_code
        if code == 429 or code >= 500:
            time.sleep(2 * (attempt + 1))
            continue
        time.sleep(0.4)  # stay under the free-tier rate limit
        if 400 <= code < 500:
            return []
        data = r.json()
        if data:
            with open(path, "w") as f:
                json.dump(data, f)
        return data
    if stale and os.path.exists(path):      # can't refresh: fall back to the old list
        with open(path) as f:
            return json.load(f)
    raise RuntimeError(f"OpenF1 {endpoint} failed after {retries} tries (HTTP {code})")


def _get_optional(endpoint: str, cache_dir: str, **params) -> list:
    """Like _get, but a failure just means the extra info is missing."""
    try:
        return _get(endpoint, cache_dir, **params)
    except Exception:
        return []


def _norm(x) -> str:
    """Lower-case and strip accents so 'Sao Paulo' matches 'São Paulo'."""
    import unicodedata
    return unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()


def list_races(year: int, cache_dir: str = "data/cache", session_name: str = "Race") -> pd.DataFrame:
    """All sessions of one type in a season, in date order, with GP names."""
    os.makedirs(cache_dir, exist_ok=True)
    races = sorted(_get("sessions", cache_dir, year=year, session_name=session_name),
                   key=lambda s: s["date_start"])
    meetings = {m["meeting_key"]: m for m in _get("meetings", cache_dir, year=year)}
    rows = [dict(Year=year, Round=i + 1, session_key=s["session_key"],
                 EventName=meetings.get(s["meeting_key"], {}).get("meeting_name"),
                 Circuit=s.get("circuit_short_name"), Location=s.get("location"),
                 Country=s.get("country_name"), Date=s["date_start"][:10])
            for i, s in enumerate(races)]
    return pd.DataFrame(rows)


def _find_session(year: int, gp, cache_dir: str, session_name: str = "Race"):
    """Match a GP name ('Bahrain', 'Sao Paulo', 'Monza') or round number to a session."""
    races = sorted(_get("sessions", cache_dir, year=year, session_name=session_name),
                   key=lambda s: s["date_start"])
    if not races:
        raise ValueError(f"No '{session_name}' sessions found on OpenF1 for {year}")
    meetings = {m["meeting_key"]: m for m in _get("meetings", cache_dir, year=year)}
    if isinstance(gp, (int, np.integer)) or str(gp).isdigit():
        return races[int(gp) - 1], meetings
    q = _norm(gp)
    for s in races:
        m = meetings.get(s["meeting_key"], {})
        names = [s.get("country_name"), s.get("location"), s.get("circuit_short_name"),
                 m.get("meeting_name"), m.get("meeting_official_name")]
        if any(n and q in _norm(n) for n in names):
            return s, meetings
    opts = [meetings.get(s["meeting_key"], {}).get("meeting_name", s.get("location")) for s in races]
    raise ValueError(f"'{gp}' not found for {year}. Options: {opts}")


def _neutralised_periods(rc: pd.DataFrame) -> list:
    """(start, end, status) for safety car ('4'), red flag ('5') and VSC ('6').

    State machine over race-control messages:
      SC  : 'SAFETY CAR DEPLOYED'          -> 'SAFETY CAR IN THIS LAP' (+2 min) or red flag
      VSC : 'VIRTUAL SAFETY CAR DEPLOYED'  -> 'VIRTUAL SAFETY CAR ENDING' (+30 s)
      Red : flag RED / 'RED FLAG' / 'SUSPENDED' -> next 'SESSION STARTED' (+3 min, covers
            the restart lap)
    Anything still open is capped (SC 25 min, VSC 15 min, red 90 min) so a missing
    message can never wipe out the rest of the race.
    """
    if rc.empty:
        return []
    rc = rc.sort_values("date")
    msg = rc["message"].fillna("").astype(str).str.upper()
    flag = rc["flag"].fillna("").astype(str).str.upper() if "flag" in rc else pd.Series("", index=rc.index)
    cat = rc["category"].fillna("").astype(str) if "category" in rc else pd.Series("", index=rc.index)
    cap = {"4": pd.Timedelta(minutes=25), "6": pd.Timedelta(minutes=15), "5": pd.Timedelta(minutes=90)}
    out, open_ = [], {}

    def close(k, t):
        start = open_.pop(k)
        out.append((start, min(t, start + cap[k]), k))

    for t, m, f, c in zip(rc["date"], msg, flag, cat):
        is_red = f == "RED" or "RED FLAG" in m or "SUSPENDED" in m
        if is_red and "5" not in open_:
            for k in ("4", "6"):
                if k in open_:
                    close(k, t)
            open_["5"] = t
        elif "VIRTUAL SAFETY CAR DEPLOYED" in m:
            open_.setdefault("6", t)
        elif "SAFETY CAR DEPLOYED" in m:
            open_.setdefault("4", t)
        if "6" in open_ and "VIRTUAL SAFETY CAR ENDING" in m:
            close("6", t + pd.Timedelta(seconds=30))
        elif "4" in open_ and ("SAFETY CAR IN THIS LAP" in m or "SAFETY CAR ENDING" in m):
            close("4", t + pd.Timedelta(minutes=2))
        if "5" in open_ and t > open_["5"] and ("SESSION STARTED" in m or c == "SessionStatus"
                                                or "RACE RESUMED" in m):
            close("5", t + pd.Timedelta(minutes=3))
    for k in list(open_):
        close(k, open_[k] + cap[k])
    return out


def _weather_summary(wx: pd.DataFrame) -> dict:
    if wx.empty:
        return dict(TrackTemp=np.nan, AirTemp=np.nan, Humidity=np.nan, Rain=False)
    g = lambda c: float(pd.to_numeric(wx[c], errors="coerce").mean()) if c in wx else np.nan
    rain = bool(pd.to_numeric(wx.get("rainfall", pd.Series([0])), errors="coerce").fillna(0).gt(0).any())
    return dict(TrackTemp=g("track_temperature"), AirTemp=g("air_temperature"),
                Humidity=g("humidity"), Rain=rain)


def load_race(year: int, gp: str | int, cache_dir: str = "data/cache",
              session_name: str = "Race", with_position: bool = True):
    """Load a race (or sprint: session_name="Sprint") from OpenF1 and return (session, laps).

    `laps` has FastF1-style columns: Driver, Team, LapNumber, LapTime, Stint,
    Compound, TyreLife, PitInTime, PitOutTime, TrackStatus, IsAccurate, Position.
    Also keeps speed traps (I1/I2/ST speed) and sector times, plus Year/Circuit.
    `session.event["EventName"]` holds the GP name; `session.weather` holds
    race-average track/air temperature, humidity and whether it rained.
    """
    os.makedirs(cache_dir, exist_ok=True)
    sess, meetings = _find_session(year, gp, cache_dir, session_name)
    sk = sess["session_key"]

    if pd.Timestamp(sess["date_start"]) > pd.Timestamp.now(tz="UTC"):
        raise NoDataError("not run yet")
    laps = pd.DataFrame(_get("laps", cache_dir, session_key=sk))
    if laps.empty:
        raise NoDataError("no lap data on OpenF1 (race cancelled or not published)")
    stints = pd.DataFrame(_get("stints", cache_dir, session_key=sk))
    if stints.empty:
        raise NoDataError("no tyre-stint data on OpenF1")
    pits = pd.DataFrame(_get_optional("pit", cache_dir, session_key=sk))
    drivers = pd.DataFrame(_get_optional("drivers", cache_dir, session_key=sk))
    rc = pd.DataFrame(_get_optional("race_control", cache_dir, session_key=sk))
    pos = (pd.DataFrame(_get_optional("position", cache_dir, session_key=sk)) if with_position
           else pd.DataFrame())
    wx = pd.DataFrame(_get_optional("weather", cache_dir, session_key=sk))

    extra = {"i1_speed": "I1Speed", "i2_speed": "I2Speed", "st_speed": "STSpeed",
             "duration_sector_1": "S1", "duration_sector_2": "S2", "duration_sector_3": "S3"}
    keep = ["driver_number", "lap_number", "lap_duration", "date_start", "is_pit_out_lap"]
    df = laps[keep + [c for c in extra if c in laps]].rename(columns=extra).copy()
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True, format="ISO8601")
    df["LapTime"] = pd.to_timedelta(df["lap_duration"], unit="s")

    # Driver code / team
    if not drivers.empty:
        d = drivers.drop_duplicates("driver_number").set_index("driver_number")
        df["Driver"] = df["driver_number"].map(d["name_acronym"]).fillna(df["driver_number"].astype(str))
        df["Team"] = df["driver_number"].map(d["team_name"])
    else:   # driver list unavailable: use car numbers
        df["Driver"] = "#" + df["driver_number"].astype(str)
        df["Team"] = None

    # Stint, compound and tyre age from the stints table
    df["Stint"], df["Compound"], df["TyreLife"] = np.nan, None, np.nan
    for st in stints.itertuples():
        if pd.isna(st.lap_start):
            continue
        lap_end = st.lap_end if pd.notna(st.lap_end) else df["lap_number"].max()
        m = ((df["driver_number"] == st.driver_number)
             & df["lap_number"].between(st.lap_start, lap_end))
        df.loc[m, "Stint"] = st.stint_number
        df.loc[m, "Compound"] = (st.compound or "UNKNOWN").upper()
        age0 = st.tyre_age_at_start if pd.notna(st.tyre_age_at_start) else 0
        df.loc[m, "TyreLife"] = age0 + df.loc[m, "lap_number"] - st.lap_start + 1

    # Pit in-laps (OpenF1 lap_number = lap the car entered the pits) and out-laps
    df["PitInTime"] = pd.Series(pd.NaT, index=df.index, dtype="timedelta64[ns]")
    df["PitOutTime"] = pd.Series(pd.NaT, index=df.index, dtype="timedelta64[ns]")
    if not pits.empty:
        pin = set(zip(pits["driver_number"], pits["lap_number"]))
        m = [(a, b) in pin for a, b in zip(df["driver_number"], df["lap_number"])]
        df.loc[m, "PitInTime"] = pd.Timedelta(1, "s")
    out = df["is_pit_out_lap"].fillna(False).astype(bool)
    df.loc[out, "PitOutTime"] = pd.Timedelta(1, "s")
    # Some races have no pit records: the lap before an out-lap is the in-lap
    prev = df[out][["driver_number", "lap_number"]].assign(lap_number=lambda x: x.lap_number - 1)
    pin2 = set(zip(prev["driver_number"], prev["lap_number"]))
    m2 = [(a, b) in pin2 for a, b in zip(df["driver_number"], df["lap_number"])]
    df.loc[m2, "PitInTime"] = pd.Timedelta(1, "s")

    # Track status per lap from SC / VSC / red-flag periods
    df["TrackStatus"] = "1"
    if not rc.empty:
        rc["date"] = pd.to_datetime(rc["date"], utc=True, format="ISO8601")
        lap_end_t = df["date_start"] + df["LapTime"].fillna(pd.Timedelta(seconds=100))
        for start, end, status in _neutralised_periods(rc):
            hit = (df["date_start"] < end) & (lap_end_t > start)
            df.loc[hit, "TrackStatus"] = status

    df["IsAccurate"] = df["lap_duration"].notna() & df["date_start"].notna()

    # Final classified position (for comparing with actual strategies)
    if not pos.empty:
        final = pos.sort_values("date").groupby("driver_number")["position"].last()
        df["Position"] = df["driver_number"].map(final)

    df = df.rename(columns={"lap_number": "LapNumber"})
    df["Year"] = year
    df["Circuit"] = sess.get("circuit_short_name") or sess.get("location")
    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    meeting = meetings.get(sess["meeting_key"], {})
    session = SimpleNamespace(
        event={"EventName": meeting.get("meeting_name", f"{sess.get('country_name')} Grand Prix"),
               "Location": sess.get("location")},
        session_key=sk, info=sess, weather=_weather_summary(wx))
    return session, df


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
