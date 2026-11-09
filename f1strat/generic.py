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

from .data import (DRY_COMPOUNDS, clean_laps, fuel_correct, list_races, load_race,
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
            except Exception as e:  # missing / incomplete sessions are recorded, not fatal
                race_rows.append(dict(Year=year, Round=r.Round, EventName=r.EventName,
                                      Circuit=r.Circuit, Status=f"FAILED: {type(e).__name__}"))
                if verbose:
                    print(f"{tag:45s} FAILED ({e})")
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
                        Rain=wet_laps > 0,      # wet = intermediates/wets actually used
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
    return laps, pd.DataFrame(race_rows)


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
    wet = races.loc[races["Rain"].fillna(False).astype(bool), "RaceId"]
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


# --------------------------------------------------------------------- models
class StructuredDeg(BaseEstimator, RegressorMixin):
    """Physics-structured model (few parameters -> hard to overfit, can extrapolate).

        Delta = a_c + S(track) * (b1_c * age + b2_c * age^2) + e * RaceProgress
        S(track) = exp(w . z),   z = standardised track + weather features

    a_c, b1_c, b2_c: one wear curve per compound shared by every circuit
    S: track severity multiplier (1 = average track), e: track evolution.
    w gets an L2 penalty `alpha`. b1, b2 >= 0 so wear never makes a tyre faster.
    """

    def __init__(self, alpha: float = 1.0, feats=tuple(TRACK_FEATURES + ["TrackTemp"])):  # noqa
        self.alpha, self.feats = alpha, feats

    def _unpack(self, th):
        k = len(list(self.feats))
        return th[0:3], th[3:6], th[6:9], th[9:9 + k], th[9 + k]

    def _f(self, th, X):
        a, b1, b2, w, e = self._unpack(th)
        c = X["CompoundCode"].to_numpy(int)
        age = X["TyreLife"].to_numpy(float)
        z = (X[list(self.feats)].to_numpy(float) - self.mu_) / self.sd_
        sev = np.exp(np.clip(z @ w, -3, 3))
        return a[c] + sev * (b1[c] * age + b2[c] * age ** 2) + e * X["RaceProgress"].to_numpy(float)

    def fit(self, X, y):
        from scipy.optimize import least_squares
        Z = X[list(self.feats)].to_numpy(float)
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-9
        k, y = len(list(self.feats)), np.asarray(y, float)
        th0 = np.r_[[0, .4, .8], [.08, .05, .03], [.001] * 3, np.zeros(k), 0.0]
        lo = np.r_[[-np.inf] * 3, [0] * 6, [-np.inf] * (k + 1)]
        res = lambda th: np.r_[(self._f(th, X) - y) / np.sqrt(len(y)),
                               np.sqrt(self.alpha / len(y)) * th[9:9 + k]]
        self.theta_ = least_squares(res, th0, bounds=(lo, np.inf), loss="soft_l1",
                                    f_scale=0.5).x
        return self

    def predict(self, X):
        return self._f(self.theta_, X)

    def severity(self, X) -> np.ndarray:
        """Track severity multiplier for rows of X (1.0 = average circuit)."""
        _, _, _, w, _ = self._unpack(self.theta_)
        z = (X[list(self.feats)].to_numpy(float) - self.mu_) / self.sd_
        return np.exp(np.clip(z @ w, -3, 3))

    def coef_table(self) -> pd.DataFrame:
        a, b1, b2, w, e = self._unpack(self.theta_)
        rows = [dict(term=f"{c} {n}", value=v) for i, c in enumerate(COMPOUND_CODE)
                for n, v in (("offset a", a[i]), ("wear b1", b1[i]), ("wear b2", b2[i]))]
        rows += [dict(term=f"severity w[{f}]", value=v) for f, v in zip(list(self.feats), w)]
        rows.append(dict(term="track evolution e (s, start->finish)", value=e))
        return pd.DataFrame(rows)


def candidate_models(random_state: int = 0) -> dict:
    """Models compared, from simplest to most flexible. Hyper-parameters are
    deliberately conservative (shallow trees, big leaves) to limit overfitting."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    base = ColumnTransformer([("c", OneHotEncoder(drop="first"), ["CompoundCode"]),
                              ("a", PolynomialFeatures(2, include_bias=False), ["TyreLife"])])
    mono = [1 if f == "TyreLife" else 0 for f in FEATURES]   # delta can only grow with age
    return {
        "Baseline (compound + age only)": (make_pipeline(base, Ridge(1.0)),
                                           ["CompoundCode", "TyreLife"]),
        "Ridge (poly-2, all features)": (make_pipeline(StandardScaler(),
                                         PolynomialFeatures(2, include_bias=False),
                                         StandardScaler(), Ridge(10.0)), FEATURES),
        "Decision tree": (DecisionTreeRegressor(max_depth=8, min_samples_leaf=200,
                                                random_state=random_state), FEATURES),
        "Random forest": (RandomForestRegressor(n_estimators=150, max_depth=12,
                          min_samples_leaf=100, max_features=0.6, max_samples=0.5,
                          n_jobs=-1, random_state=random_state), FEATURES),
        "Gradient boosting (monotonic)": (HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, max_iter=400, min_samples_leaf=200,
            l2_regularization=1.0, monotonic_cst=mono, early_stopping=True,
            validation_fraction=0.15, random_state=random_state), FEATURES),
        "Structured (compound curve x track severity)": (StructuredDeg(alpha=1.0), FEATURES),
    }


def _scores(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=r2_score(y, p))


def compare_models(d: pd.DataFrame, n_splits: int = 5, models: dict | None = None):
    """Leave-circuits-out cross-validation.

    Folds are split by Circuit (all years of a track go to the same fold), so
    every test score is for tracks the model has never seen - the honest test
    for a generic model. A random-split score is also reported: when it is much
    better than the grouped one, the model is memorising circuits (overfitting).
    The train-vs-test gap shows overfitting within the training data.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, KFold

    models = models or candidate_models()
    y, groups = d[TARGET].to_numpy(), d["Circuit"].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    rows, oof = [], {}
    for name, (est, feats) in models.items():
        X = d[feats]
        pred = np.zeros(len(d))
        tr_sc, te_sc = [], []
        for tr, te in GroupKFold(n_splits).split(X, y, groups):
            m = clone(est).fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
            tr_sc.append(_scores(y[tr], m.predict(X.iloc[tr]))["MAE"])
            te_sc.append(_scores(y[te], pred[te])["MAE"])
        rnd = np.zeros(len(d))
        for tr, te in KFold(n_splits, shuffle=True, random_state=0).split(X):
            rnd[te] = clone(est).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        s = _scores(y, pred)
        rows.append(dict(Model=name, Test_MAE=s["MAE"], Test_RMSE=s["RMSE"], Test_R2=s["R2"],
                         Train_MAE=np.mean(tr_sc), Overfit_gap=np.mean(te_sc) - np.mean(tr_sc),
                         Fold_MAE_std=np.std(te_sc), RandomSplit_MAE=_scores(y, rnd)["MAE"]))
        oof[name] = pred
    res = pd.DataFrame(rows).sort_values("Test_MAE").reset_index(drop=True)
    return res, oof


def tree_depth_curve(d: pd.DataFrame, depths=range(2, 21, 2), n_splits: int = 5):
    """Train vs unseen-circuit error of a decision tree as depth grows - shows
    where extra complexity stops helping and starts overfitting."""
    from sklearn.model_selection import GroupKFold
    from sklearn.tree import DecisionTreeRegressor
    X, y, g = d[FEATURES], d[TARGET].to_numpy(), d["Circuit"].to_numpy()
    n_splits = min(n_splits, len(np.unique(g)))
    rows = []
    for depth in depths:
        tr_e, te_e = [], []
        for tr, te in GroupKFold(n_splits).split(X, y, g):
            m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=20, random_state=0)
            m.fit(X.iloc[tr], y[tr])
            tr_e.append(np.mean(np.abs(m.predict(X.iloc[tr]) - y[tr])))
            te_e.append(np.mean(np.abs(m.predict(X.iloc[te]) - y[te])))
        rows.append(dict(depth=depth, train_MAE=np.mean(tr_e), test_MAE=np.mean(te_e)))
    return pd.DataFrame(rows)


def fit_final(d: pd.DataFrame, name: str, models: dict | None = None):
    from sklearn.base import clone
    est, feats = (models or candidate_models())[name]
    return clone(est).fit(d[feats], d[TARGET]), feats


def permutation_importance_table(model, feats, d: pd.DataFrame, n: int = 20000, seed: int = 0):
    from sklearn.inspection import permutation_importance
    s = d.sample(min(n, len(d)), random_state=seed)
    r = permutation_importance(model, s[feats], s[TARGET], n_repeats=5, random_state=seed,
                               scoring="neg_mean_absolute_error")
    return (pd.DataFrame(dict(Feature=feats, MAE_increase=r.importances_mean, std=r.importances_std))
            .sort_values("MAE_increase", ascending=False).reset_index(drop=True))


# ----------------------------------------------------------------- prediction
def _circuit_rows(races: pd.DataFrame, circuit: str) -> pd.DataFrame:
    from .data import _norm
    ok = races[races["Status"] == "ok"]
    m = ok["Circuit"].map(_norm).str.contains(_norm(circuit), regex=False) | \
        ok["EventName"].map(_norm).str.contains(_norm(circuit), regex=False)
    return ok[m]


def track_profile(races: pd.DataFrame, circuit: str, **overrides) -> dict:
    """Feature values for a circuit = average of its past (dry) editions.

    Pass overrides for anything known in advance, e.g. a forecast
    TrackTemp=45. For a brand-new circuit, pass every TRACK_FEATURE yourself.
    """
    rows = _circuit_rows(races, circuit)
    dry = rows[~rows["Rain"].fillna(False).astype(bool)]
    rows = dry if len(dry) else rows
    if rows.empty and not all(k in overrides for k in TRACK_FEATURES):
        raise ValueError(f"No past races at '{circuit}'. Pass {TRACK_FEATURES} as keyword args.")
    prof = {k: float(rows[k].mean()) if len(rows) else np.nan
            for k in TRACK_FEATURES + WEATHER_FEATURES}
    prof.update(TotalLaps=int(rows["TotalLaps"].median()) if len(rows) else None,
                PitLoss=float(rows["PitLoss"].median()) if len(rows) else np.nan,
                P_SC=float(rows["HadSC"].mean()) if len(rows) else np.nan,
                Editions=len(rows),
                Circuit=rows["Circuit"].iloc[0] if len(rows) else circuit)
    prof.update(overrides)
    return prof


def predict_curves(model, feats, profile: dict, max_age: int = 50,
                   race_progress: float = 0.5) -> pd.DataFrame:
    """Predicted delta vs tyre age for each dry compound at this track.
    race_progress is fixed (default mid-race) so track evolution does not
    leak into the tyre curves."""
    ages = np.arange(1, max_age + 1)
    rows = []
    for comp, code in COMPOUND_CODE.items():
        X = pd.DataFrame({"TyreLife": ages, "CompoundCode": code, "RaceProgress": race_progress,
                          **{k: profile[k] for k in TRACK_FEATURES + WEATHER_FEATURES}})
        rows.append(pd.DataFrame(dict(Compound=comp, TyreLife=ages,
                                      Delta=model.predict(X[feats]))))
    return pd.concat(rows, ignore_index=True)


def predict_deg_models(model, feats, profile: dict, train: pd.DataFrame,
                       cv_rmse: float = float("nan")) -> dict[str, DegModel]:
    """Turn the generic model's curves into per-compound DegModel objects so
    the existing strategy optimiser (notebook 02) can use them unchanged.

    Max usable stint per compound = 95th percentile of tyre age seen for that
    compound in training (the model is not trusted beyond what it has seen).
    """
    max_seen = train.groupby("Compound")["TyreLife"].quantile(0.95).round().astype(int)
    cur = predict_curves(model, feats, profile, max_age=int(train["TyreLife"].max()))
    out = {}
    for comp, g in cur.groupby("Compound"):
        g = g[g["TyreLife"] <= max_seen[comp]]
        p = np.polyfit(g["TyreLife"], g["Delta"], 2)
        out[comp] = DegModel(comp, "generic", [float(p[2]), float(p[1]), float(p[0])],
                             float(cv_rmse), int(len(train[train["Compound"] == comp])),
                             int(max_seen[comp]), None)
    return out


def holdout_check(d: pd.DataFrame, races: pd.DataFrame, circuit: str, name: str):
    """Train on every other circuit, predict this one, and compare with the
    curve fitted directly to its own laps (what notebook 01 would give)."""
    from .degradation import fit_all
    from .data import _norm
    names = set(_circuit_rows(races, circuit)["Circuit"])   # 'Sao Paulo' -> 'Interlagos'
    mask = d["Circuit"].isin(names) | d["Circuit"].map(_norm).str.contains(_norm(circuit), regex=False)
    if not mask.any():
        raise ValueError(f"'{circuit}' not in the cleaned dataset")
    model, feats = fit_final(d[~mask], name)
    prof = track_profile(races, circuit)
    gen = predict_deg_models(model, feats, prof, d[~mask])
    own = fit_all(d[mask])
    rows = []
    for c in own:
        if c in gen:
            age = np.arange(1, min(own[c].max_age_seen, gen[c].max_age_seen) + 1)
            err = gen[c].predict(age) - own[c].predict(age)
            rows.append(dict(Compound=c, Own_deg=own[c].deg_per_lap, Generic_deg=gen[c].deg_per_lap,
                             Own_offset=own[c].predict(1).item(),
                             Generic_offset=gen[c].predict(1).item(),
                             Curve_MAE=float(np.mean(np.abs(err)))))
    return pd.DataFrame(rows), own, gen
