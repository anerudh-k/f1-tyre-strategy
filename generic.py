"""Generic (multi-circuit) tyre degradation model.

One model is trained on every dry race OpenF1 has. It predicts the
fuel-corrected lap-time delta of a lap from tyre age, compound, race progress
(track evolution), weather and features that describe the circuit, so it can
produce degradation curves for any track, including one it was not trained on.

Pipeline:  build_dataset -> clean_dataset (outliers) -> compare_models
           (leave-circuits-out CV) -> fit_final -> predict_deg_models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .data import (DRY_COMPOUNDS, NoDataError, clean_laps, fuel_correct, list_races, load_race,
                   pit_loss_from_session)
from .degradation import DegModel

COMPOUND_CODE = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
TRACK_FEATURES = ["MedianLap", "STSpeed", "I1Speed", "I2Speed", "S1Share", "S2Share"]
WEATHER_FEATURES = ["TrackTemp", "AirTemp", "Humidity"]
FEATURES = ["TyreLife", "CompoundCode", "RaceProgress"] + TRACK_FEATURES + WEATHER_FEATURES
TARGET = "Delta"


# --------------------------------------------------------------------------- data
def _race_features(raw: pd.DataFrame) -> dict:
    """Circuit descriptors from green-flag, non-pit laps of one race."""
    g = raw[(raw["TrackStatus"].astype(str) == "1") & raw["PitInTime"].isna()
            & raw["PitOutTime"].isna() & raw["LapTime"].notna() & (raw["LapNumber"] > 1)]
    lt = g["LapTime"].dt.total_seconds()
    med = lambda c: float(pd.to_numeric(g[c], errors="coerce").median()) if c in g else np.nan
    out = dict(MedianLap=float(lt.median()), STSpeed=med("STSpeed"),
               I1Speed=med("I1Speed"), I2Speed=med("I2Speed"))
    if {"S1", "S2"} <= set(g.columns):
        out["S1Share"] = float((pd.to_numeric(g["S1"], errors="coerce") / lt).median())
        out["S2Share"] = float((pd.to_numeric(g["S2"], errors="coerce") / lt).median())
    else:
        out["S1Share"] = out["S2Share"] = np.nan
    return out


def build_dataset(years=(2023, 2024, 2025), cache_dir: str = "data/cache",
                  fuel_start_kg: float = 100.0, sec_per_kg: float = 0.03, verbose: bool = True):
    """Download and assemble every race in `years`.

    Returns (laps, races):
      laps  - one row per clean lap with target `Delta` and all features
      races - one row per race: data-quality numbers, track/weather features,
              pit-loss estimate, whether it rained or had a safety car
    """
    lap_frames, race_rows = [], []
    for year in years:
        for r in list_races(year, cache_dir).itertuples():
            tag = f"{year} R{r.Round:02d} {r.EventName}"
            try:
                sess, raw = load_race(year, r.Round, cache_dir, with_position=False)
            except NoDataError as e:   # cancelled / future / unpublished: expected, not an error
                race_rows.append(dict(Year=year, Round=r.Round, EventName=r.EventName,
                                      Circuit=r.Circuit, Status=f"skipped: {e}"))
                if verbose:
                    print(f"{tag:45s} skipped - {e}")
                continue
            except Exception as e:      # anything else is recorded, not fatal
                race_rows.append(dict(Year=year, Round=r.Round, EventName=r.EventName,
                                      Circuit=r.Circuit, Status=f"FAILED: {type(e).__name__}: {e}"[:150]))
                if verbose:
                    print(f"{tag:45s} FAILED - {type(e).__name__}: {e}")
                continue
            total = int(raw["LapNumber"].max())
            comps = raw["Compound"].value_counts().to_dict()
            wet_laps = int(raw["Compound"].isin(["INTERMEDIATE", "WET"]).sum())
            info = dict(RaceId=f"{year}_{r.Round:02d}_{raw['Circuit'].iloc[0]}",
                        Year=year, Round=r.Round, EventName=r.EventName,
                        Circuit=raw["Circuit"].iloc[0], TotalLaps=total,
                        RawLaps=len(raw), MissingLapTime=int(raw["LapTime"].isna().sum()),
                        SCLaps=int(raw.groupby("LapNumber")["TrackStatus"]
                                   .agg(lambda s: (s.astype(str) != "1").mean() > 0.5).sum()),
                        HadSC=bool((raw["TrackStatus"].astype(str) == "4").any()),
                        WetLaps=wet_laps, RainfallSeen=sess.weather["Rain"],
                        # wet race = more than 10% of laps on inters/wets (a few laps
                        # on inters at a dry race are just a gamble, not a wet race)
                        Rain=wet_laps > 0.10 * len(raw),
                        PitLoss=pit_loss_from_session(raw),
                        **{f"Laps_{c}": comps.get(c, 0) for c in DRY_COMPOUNDS},
                        **_race_features(raw),
                        **{k: sess.weather[k] for k in WEATHER_FEATURES})
            clean = clean_laps(raw)
            info["CleanLaps"] = len(clean)
            info["Status"] = "ok"
            race_rows.append(info)
            if verbose:
                print(f"{tag:45s} {len(raw):5d} raw -> {len(clean):5d} clean"
                      + ("  (wet)" if info["Rain"] else ""))
            if clean.empty:
                continue
            df = fuel_correct(clean, total, fuel_start_kg, sec_per_kg)
            df["RaceId"] = info["RaceId"]
            df["Year"], df["Circuit"], df["EventName"] = year, info["Circuit"], r.EventName
            df["RaceProgress"] = df["LapNumber"] / total
            df["CompoundCode"] = df["Compound"].map(COMPOUND_CODE)
            for k in TRACK_FEATURES + WEATHER_FEATURES + ["Rain"]:
                df[k] = info[k]
            lap_frames.append(df)
    laps = pd.concat(lap_frames, ignore_index=True) if lap_frames else pd.DataFrame()
    races = pd.DataFrame(race_rows)
    if not laps.empty:
        laps, races = _impute_track_features(laps, races, verbose)
    return laps, races


def _impute_track_features(laps, races, verbose=True):
    """A missing circuit descriptor (e.g. a broken speed trap) is filled with the
    same circuit's value from other years, so one bad sensor doesn't drop a race."""
    cols = TRACK_FEATURES + WEATHER_FEATURES
    ok = races["Status"] == "ok"
    for c in cols:
        miss = ok & races[c].isna()
        if miss.any():
            fill = races[ok].groupby("Circuit")[c].median()
            races.loc[miss, c] = races.loc[miss, "Circuit"].map(fill)
            if verbose:
                print(f"filled missing {c} for {races.loc[miss, 'RaceId'].tolist()}")
    m = races[ok].set_index("RaceId")
    for c in cols:
        laps[c] = laps["RaceId"].map(m[c])
    return laps, races


# --------------------------------------------------------------------- cleaning
def _mad_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    return 0.6745 * (x - med) / (mad if mad > 0 else 1e-9)


def stint_slopes(laps: pd.DataFrame, min_laps: int = 5) -> pd.DataFrame:
    """Per-stint linear deg slope (s/lap) - useful for EDA and stint-level outliers."""
    rows = []
    for (rid, drv, st), g in laps.groupby(["RaceId", "Driver", "Stint"]):
        if len(g) >= min_laps and g["TyreLife"].nunique() >= 3:
            slope = np.polyfit(g["TyreLife"], g[TARGET], 1)[0]
            rows.append(dict(RaceId=rid, Driver=drv, Stint=st, Circuit=g["Circuit"].iloc[0],
                             Compound=g["Compound"].iloc[0], Laps=len(g),
                             StartAge=g["TyreLife"].min(), Slope=slope))
    return pd.DataFrame(rows)


def clean_dataset(laps: pd.DataFrame, races: pd.DataFrame, z_max: float = 3.5,
                  max_slope: float = 0.4, min_race_laps: int = 200, age_pct: float = 0.99,
                  verbose: bool = True):
    """Remove outliers in layers and report what each layer dropped.

    1. wet races (rain or inter/wet tyres used)       - different physics
    2. races with too few clean laps                    - unreliable features
    3. stints shorter than 5 laps or with a non-physical slope
       (|slope| > max_slope s/lap)                      - bad stint/compound data
    4. lap residual outliers: robust z (MAD) of the residual from a per
       race/compound linear fit, |z| > z_max            - traffic, errors, mis-timed laps
    5. tyre ages beyond the 99th percentile per compound - too few points to learn
    """
    log, d = [], laps.copy()
    log.append(("start", len(d)))
    wet = races.loc[races["Rain"].eq(True), "RaceId"]
    d = d[~d["RaceId"].isin(set(wet))]
    log.append(("wet races", len(d)))
    n = d.groupby("RaceId")[TARGET].transform("size")
    d = d[n >= min_race_laps]
    log.append((f"races < {min_race_laps} laps", len(d)))

    # stint-level check first, so the lap filter can't hide a corrupted stint
    sl = stint_slopes(d)
    key = ["RaceId", "Driver", "Stint"]
    d = d.merge(sl[key], on=key, how="inner")            # drops stints < 5 laps
    log.append(("stints < 5 laps", len(d)))
    bad = sl.loc[sl["Slope"].abs() > max_slope, key]
    d = d.merge(bad.assign(_bad=1), on=key, how="left")
    d = d[d["_bad"].isna()].drop(columns="_bad").reset_index(drop=True)
    log.append((f"stint |slope|>{max_slope}", len(d)))

    def resid(g):
        if len(g) < 10:
            return pd.Series(0.0, index=g.index)
        p = np.polyfit(g["TyreLife"], g[TARGET], 1)
        return g[TARGET] - np.polyval(p, g["TyreLife"])
    r = d.groupby(["RaceId", "Compound"], group_keys=False)[["TyreLife", TARGET]].apply(resid)
    r = r.reindex(d.index)
    z = r.groupby(d["RaceId"]).transform(_mad_z)
    d = d[(z.abs() <= z_max).to_numpy()]
    log.append((f"lap residual |z|>{z_max}", len(d)))

    cap = d.groupby("Compound")["TyreLife"].transform(lambda s: s.quantile(age_pct))
    d = d[d["TyreLife"] <= cap]
    log.append((f"tyre age > p{int(age_pct*100)}", len(d)))
    d = d.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    log.append(("missing features", len(d)))

    rep = pd.DataFrame(log, columns=["step", "laps_left"])
    rep["removed"] = (-rep["laps_left"].diff()).fillna(0).astype(int)
    if verbose:
        print(rep.to_string(index=False))
    return d, rep


# ------------------------------------------------------- per-race tyre model
def fit_race(g: pd.DataFrame, min_laps: int = 40) -> dict | None:
    """Separate tyre wear from fuel burn + track evolution for ONE race.

        LapTime = driver pace + compound offset + LapEffect * LapNumber
                  + Wear_compound * TyreLife

    Within a stint, tyre age and lap number rise together, so they can't be
    told apart from one stint alone; stints starting at different laps make it
    possible. Fitted with robust (Huber) regression on raw, non-fuel-corrected
    lap times. Returns wear (s/lap) and its standard error per compound, the
    compound offsets vs MEDIUM (s) and the fuel+track LapEffect (s/lap).
    """
    comps = [c for c in DRY_COMPOUNDS if (g["Compound"] == c).sum() >= min_laps]
    g = g[g["Compound"].isin(comps)]
    if len(comps) < 2 or g["Driver"].nunique() < 5:
        return None
    ref = "MEDIUM" if "MEDIUM" in comps else comps[0]
    X = pd.concat([pd.get_dummies(g["Driver"], dtype=float),
                   pd.DataFrame({f"off_{c}": (g["Compound"] == c).astype(float)
                                 for c in comps if c != ref}, index=g.index),
                   pd.DataFrame({f"wear_{c}": ((g["Compound"] == c) * g["TyreLife"]).astype(float)
                                 for c in comps}, index=g.index),
                   g[["LapNumber"]].astype(float)], axis=1)
    A, y, w = X.to_numpy(), g["LapTimeS"].to_numpy(float), np.ones(len(g))
    for _ in range(8):                                   # Huber IRLS
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ beta
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        w = np.where(np.abs(r) <= 1.345 * s, 1.0, 1.345 * s / np.abs(r))
    Aw = A * np.sqrt(w)[:, None]
    cov = np.linalg.pinv(Aw.T @ Aw) * np.sum(w * r ** 2) / max(len(y) - A.shape[1], 1)
    b = dict(zip(X.columns, beta)); se = dict(zip(X.columns, np.sqrt(np.clip(np.diag(cov), 0, None))))
    out = dict(LapEffect=b["LapNumber"], LapEffectSE=se["LapNumber"], ResidMAD=s, NLaps=len(g), Ref=ref)
    for c in comps:
        out[f"Wear_{c}"], out[f"WearSE_{c}"] = b[f"wear_{c}"], se[f"wear_{c}"]
        out[f"Off_{c}"] = 0.0 if c == ref else b[f"off_{c}"]
        out[f"MaxAge_{c}"] = float(g.loc[g["Compound"] == c, "TyreLife"].quantile(0.95))
    if ref != "MEDIUM":                                  # no medium: offsets unknown vs MEDIUM
        for c in comps:
            out[f"Off_{c}"] = np.nan
    return out


def race_table(d: pd.DataFrame, races: pd.DataFrame, max_se: float = 0.015,
               verbose: bool = True) -> pd.DataFrame:
    """Fit every race and return one row per race x compound.

    Estimates that can't be trusted are dropped: wear uncertainty > max_se
    (usually everyone pitted on the same lap, so wear and lap number can't be
    separated) or physically impossible values.
    """
    info = races[races["Status"] == "ok"].set_index("RaceId")
    rows = []
    for rid, g in d.groupby("RaceId"):
        o = fit_race(g)
        if not o:
            continue
        for c in DRY_COMPOUNDS:
            if f"Wear_{c}" in o:
                rows.append(dict(RaceId=rid, Circuit=g["Circuit"].iloc[0], Year=int(g["Year"].iloc[0]),
                                 Round=int(info.loc[rid, "Round"]), Compound=c, Wear=o[f"Wear_{c}"],
                                 SE=o[f"WearSE_{c}"], Offset=o[f"Off_{c}"], MaxAge=o[f"MaxAge_{c}"],
                                 LapEffect=o["LapEffect"], ResidMAD=o["ResidMAD"]))
    L = pd.DataFrame(rows)
    n0 = len(L)
    keep = (L["SE"] < max_se) & L["LapEffect"].between(-0.2, 0.05) & L["Wear"].between(-0.05, 0.3)
    if verbose:
        bad = L.loc[~keep, ["RaceId", "Compound", "Wear", "SE"]].round(3)
        print(f"{n0} race x compound estimates, {int((~keep).sum())} dropped as unreliable")
        if len(bad):
            print(bad.to_string(index=False))
    L = L[keep].reset_index(drop=True)
    L["Order"] = L["Year"] * 100 + L["Round"]
    L["CompoundCode"] = L["Compound"].map(COMPOUND_CODE)
    L["Era2026"] = (L["Year"] >= 2026).astype(int)
    return L.join(info[TRACK_FEATURES + WEATHER_FEATURES], on="RaceId")


# ----------------------------------------------------------- history model
class HistoryModel:
    """Predicts a race's tyre numbers from earlier races only.

    For a circuit, compound and season:
      1. prior   = this season's average so far, shrunk toward the all-time
                   median (k_season pseudo-races) - captures yearly tyre changes
      2. predict = this circuit's past values, shrunk toward the prior
                   (k_circuit pseudo-races) - few past races -> trust the prior
    Used for wear (s/lap) and compound offset (s vs MEDIUM). Only a handful of
    numbers are estimated, so it cannot overfit.
    """

    def __init__(self, k_circuit: float = 3.0, k_season: float = 5.0):
        self.k_circuit, self.k_season = k_circuit, k_season

    def fit(self, L: pd.DataFrame):
        self.L_ = L.copy()
        return self

    def _one(self, A, circuit, year, compound, col):
        a = A[(A["Compound"] == compound)].dropna(subset=[col])
        if a.empty:
            return np.nan
        prior = a[col].median()
        s = a.loc[a["Year"] == year, col]
        prior = (s.sum() + self.k_season * prior) / (len(s) + self.k_season)
        h = a.loc[a["Circuit"] == circuit, col]
        return (h.sum() + self.k_circuit * prior) / (len(h) + self.k_circuit)

    def predict_one(self, circuit, year, compound, before=None, col="Wear"):
        A = self.L_ if before is None else self.L_[self.L_["Order"] < before]
        return self._one(A, circuit, year, compound, col)

    def predict(self, B: pd.DataFrame, before=None) -> np.ndarray:
        return np.array([self.predict_one(r.Circuit, r.Year, r.Compound, before) for r in B.itertuples()])


# ------------------------------------------------------- model comparison
MODEL_FEATURES = ["CompoundCode", "Era2026"] + TRACK_FEATURES + ["TrackTemp", "AirTemp"]


def candidate_models(random_state: int = 0) -> dict:
    """Race-level models compared against the history model. Tree models get
    conservative settings (shallow, big leaves) to limit overfitting."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    return {
        "Ridge (features + history)": make_pipeline(StandardScaler(), Ridge(20.0)),
        "Decision tree (features + history)": DecisionTreeRegressor(max_depth=3, min_samples_leaf=10,
                                                                    random_state=random_state),
        "Random forest (features + history)": RandomForestRegressor(
            300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=random_state),
        "Gradient boosting (features + history)": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10,
            l2_regularization=1.0, random_state=random_state),
    }


def _history_feature(A: pd.DataFrame, B: pd.DataFrame, hm: HistoryModel) -> np.ndarray:
    hm.fit(A)
    return hm.predict(B)


def _history_feature_oof(A: pd.DataFrame, hm: HistoryModel) -> np.ndarray:
    """History feature for training rows, each computed without its own race."""
    out = np.empty(len(A))
    for rid, idx in A.groupby("RaceId").indices.items():
        out[idx] = _history_feature(A[A["RaceId"] != rid], A.iloc[idx], hm)
    return out


def compare_models(L: pd.DataFrame, test_from: int = 202501, models: dict | None = None):
    """Forward-chaining validation: every race from `test_from` on is predicted
    using ONLY the races before it (exactly how the model would be used before
    a race weekend). Reports error on those races (Test), error on the data each
    model was trained on (Train) and the gap between them (overfitting).
    """
    from sklearn.base import clone
    models = models or candidate_models()
    L = L.reset_index(drop=True)
    names = ["Compound average", "Circuit history (HistoryModel)"] + list(models)
    pred = {n: np.full(len(L), np.nan) for n in names}
    train_err = {n: [] for n in names}
    hm = HistoryModel()
    for o in sorted(L.loc[L["Order"] >= test_from, "Order"].unique()):
        A, B = L[L["Order"] < o].reset_index(drop=True), L[L["Order"] == o]
        if len(A) < 30:
            continue
        cavg = A.groupby("Compound")["Wear"].median()
        pred["Compound average"][B.index] = B["Compound"].map(cavg)
        train_err["Compound average"].append(np.mean(np.abs(A["Compound"].map(cavg) - A["Wear"])))
        hA, hB = _history_feature_oof(A, hm), _history_feature(A, B, hm)
        pred["Circuit history (HistoryModel)"][B.index] = hB
        train_err["Circuit history (HistoryModel)"].append(np.mean(np.abs(hA - A["Wear"])))
        Xa, Xb = A[MODEL_FEATURES].assign(History=hA), B[MODEL_FEATURES].assign(History=hB)
        for n, est in models.items():
            m = clone(est).fit(Xa, A["Wear"])
            pred[n][B.index] = m.predict(Xb)
            train_err[n].append(np.mean(np.abs(m.predict(Xa) - A["Wear"])))
    test = L["Order"] >= test_from
    y = L.loc[test, "Wear"].to_numpy()
    rows = []
    for n in names:
        p = pred[n][test.to_numpy()]
        ok = ~np.isnan(p)
        mae = np.mean(np.abs(p[ok] - y[ok]))
        rows.append(dict(Model=n, Test_MAE=mae, Test_R2=1 - np.sum((p[ok] - y[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2),
                         Train_MAE=np.mean(train_err[n]), Overfit_gap=mae - np.mean(train_err[n]),
                         N_test=int(ok.sum())))
    return pd.DataFrame(rows).sort_values("Test_MAE").reset_index(drop=True), pred


def tree_depth_curve(L: pd.DataFrame, depths=range(1, 11), test_from: int = 202501):
    """Decision tree train vs test error as depth grows (forward-chaining)."""
    from sklearn.tree import DecisionTreeRegressor
    trees = {f"depth {k}": DecisionTreeRegressor(max_depth=k, min_samples_leaf=3, random_state=0)
             for k in depths}
    res, _ = compare_models(L, test_from, trees)
    res = res.set_index("Model")
    return pd.DataFrame([dict(depth=k, train_MAE=res.loc[f"depth {k}", "Train_MAE"],
                              test_MAE=res.loc[f"depth {k}", "Test_MAE"]) for k in depths])


def feature_importance(L: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permutation importance of a random forest on all race-level data."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    hm = HistoryModel()
    X = L[MODEL_FEATURES].assign(History=_history_feature_oof(L.reset_index(drop=True), hm))
    m = RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, random_state=seed, n_jobs=-1)
    m.fit(X, L["Wear"])
    r = permutation_importance(m, X, L["Wear"], n_repeats=10, random_state=seed,
                               scoring="neg_mean_absolute_error")
    return (pd.DataFrame(dict(Feature=X.columns, MAE_increase=r.importances_mean, std=r.importances_std))
            .sort_values("MAE_increase", ascending=False).reset_index(drop=True))


# ----------------------------------------------------------------- prediction
def _circuit_rows(races: pd.DataFrame, circuit: str) -> pd.DataFrame:
    from .data import _norm
    ok = races[races["Status"] == "ok"]
    q = _norm(circuit)
    m = (ok["Circuit"].map(_norm).str.contains(q, regex=False)
         | ok["EventName"].map(_norm).str.contains(q, regex=False))
    return ok[m]


def resolve_circuit(races: pd.DataFrame, name: str) -> str:
    """'Sao Paulo' / 'Brazil' / 'Interlagos' -> the circuit name used in the data."""
    rows = _circuit_rows(races, name)
    if rows.empty:
        raise ValueError(f"No past races found for '{name}'")
    return rows["Circuit"].mode().iloc[0]


def track_profile(races: pd.DataFrame, circuit: str) -> dict:
    """Race length, pit loss and safety-car rate from a circuit's past races."""
    rows = _circuit_rows(races, circuit)
    if rows.empty:
        raise ValueError(f"No past races at '{circuit}'")
    return dict(Circuit=rows["Circuit"].mode().iloc[0], Editions=len(rows),
                TotalLaps=int(rows["TotalLaps"].median()),
                PitLoss=float(rows["PitLoss"].median()),
                P_SC=float(rows["HadSC"].mean()))


def predict_deg_models(hm: HistoryModel, circuit: str, year: int, before: int | None = None,
                       cv_mae: float = float("nan")) -> dict[str, DegModel]:
    """Turn history-model predictions into per-compound DegModels for notebook 02.

    delta(age) = offset vs MEDIUM + wear * age. Max stint per compound = the
    longest stints usually seen (95th pct of tyre age) at this circuit, else overall.
    """
    L = hm.L_ if before is None else hm.L_[hm.L_["Order"] < before]
    out = {}
    for c in DRY_COMPOUNDS:
        wear = hm.predict_one(circuit, year, c, before, "Wear")
        off = hm.predict_one(circuit, year, c, before, "Offset")
        if np.isnan(wear):
            continue
        a = L[(L["Compound"] == c) & (L["Circuit"] == circuit)]["MaxAge"]
        a = a if len(a) else L[L["Compound"] == c]["MaxAge"]
        out[c] = DegModel(c, "history", [float(0.0 if np.isnan(off) else off), float(wear)],
                          float(cv_mae), int((L["Compound"] == c).sum()), int(round(a.median())), None)
    return out


def holdout_check(L: pd.DataFrame, races: pd.DataFrame, circuit: str, hm: HistoryModel | None = None):
    """Predict the circuit's most recent race using only races before it, and
    compare with what the race-fit measured."""
    hm = (hm or HistoryModel()).fit(L)
    c = resolve_circuit(races, circuit)
    last = L[L["Circuit"] == c]
    if last.empty:
        raise ValueError(f"No usable dry races at {c}")
    o = last["Order"].max()
    actual = last[last["Order"] == o]
    rows = []
    for r in actual.itertuples():
        rows.append(dict(Race=r.RaceId, Compound=r.Compound, Actual_wear=r.Wear,
                         Predicted_wear=hm.predict_one(c, r.Year, r.Compound, before=o),
                         Compound_avg=L[(L["Order"] < o) & (L["Compound"] == r.Compound)]["Wear"].median()))
    t = pd.DataFrame(rows)
    t["Error"] = t["Predicted_wear"] - t["Actual_wear"]
    return t
