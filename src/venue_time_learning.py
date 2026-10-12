"""Independent, leakage-safe venue/class/field/time winner-tactic research.

Read actual completed races only. Exports a shadow model and chronological
comparison; never modifies production predictions, bets, or model files.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VERSION = "venue_time_tactics_v1"
JST = ZoneInfo("Asia/Tokyo")
METHODS = ("逃", "捲", "差")
# KEIRIN.JP venue list. A listed but unsupported venue must be shown as missing.
EXPECTED_VENUES = (
    "函館", "青森", "いわき平", "弥彦", "前橋", "取手", "宇都宮", "大宮", "西武園",
    "京王閣", "立川", "松戸", "千葉", "川崎", "平塚", "小田原", "伊東", "静岡",
    "名古屋", "岐阜", "大垣", "豊橋", "富山", "松阪", "四日市", "福井", "奈良",
    "向日町", "和歌山", "岸和田", "玉野", "広島", "防府", "高松", "小松島",
    "高知", "松山", "小倉", "久留米", "武雄", "佐世保", "別府", "熊本",
)
BANDS = ("morning", "day", "evening", "late")
LEVELS = ("national", "venue", "venue_band", "venue_hour")
REQUIRED = {"race_id", "date", "venue", "car_no", "finish_pos"}
USED = REQUIRED | {
    "player_id", "race_class", "race_type", "player_class", "current_term_class",
    "is_grade_race", "start_at", "entries_number", "result_factor", "line_id",
    "line_verification_status", "official_finish_pos",
}


def venue_name(value):
    name = unicodedata.normalize("NFKC", str(value or "")).replace("　", "")
    name = re.sub(r"\s+", "", name)
    return {"伊東温泉": "伊東", "伊東競輪": "伊東", "千葉競輪": "千葉"}.get(name, name)


def classify(row):
    klass = unicodedata.normalize("NFKC", str(row.get("race_class", "") or ""))
    if not klass or klass.lower() == "nan":
        raw = pd.to_numeric(row.get("current_term_class", row.get("player_class")), errors="coerce")
        klass = {1: "S", 2: "A", 3: "A", 4: "L"}.get(raw, "")
    else:
        klass = klass[0:1].upper()
    if klass not in ("A", "S", "L"):
        return "unknown", "unknown"
    kind = unicodedata.normalize("NFKC", str(row.get("race_type", "") or ""))
    grade = str(row.get("is_grade_race", "")).lower() in ("1", "1.0", "true")
    if klass == "A":
        return klass, "A_challenge" if "チャレンジ" in kind else "A_regular"
    if klass == "S":
        return klass, "S_graded" if grade else "S_regular"
    return "L", "L"


def jst_hour(value):
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or not 1000000000 < float(number) < 10000000000000:
        return None
    timestamp = float(number) / (1000 if number > 1000000000000 else 1)
    try:
        return datetime.fromtimestamp(timestamp, JST).hour
    except (ValueError, OverflowError, OSError):
        return None


def time_band(hour):
    if hour is None:
        return "unknown"
    if hour < 11:
        return "morning"
    if hour < 16:
        return "day"
    if hour < 21:
        return "evening"
    return "late"


def winner_method(raw):
    value = unicodedata.normalize("NFKC", str(raw or "")).strip()
    for prefix, label in (("逃", "逃"), ("捲", "捲"), ("まく", "捲"), ("差", "差")):
        if value.startswith(prefix):
            return label
    return None  # Unknown provider enum is not guessed from racer style.


def load_history(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        columns = set(fh.readline().strip().split(","))
    missing = REQUIRED - columns
    if missing:
        raise ValueError(f"historical results missing required columns: {sorted(missing)}")
    return pd.read_csv(path, usecols=lambda name: name in USED, dtype={"race_id": str, "player_id": str}, low_memory=False)


def to_races(entries, asof):
    """One observed race per row; rejects incomplete and contradictory starts."""
    if not REQUIRED.issubset(entries.columns):
        raise ValueError(f"history schema missing: {sorted(REQUIRED-set(entries.columns))}")
    frame = entries.copy()
    frame["_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[frame._date.lt(pd.Timestamp(asof).normalize()) & frame._date.notna()]
    rows = []
    for rid, group in frame.groupby("race_id", sort=False):
        if group.empty or group["car_no"].isna().any():
            continue
        cars = pd.to_numeric(group.car_no, errors="coerce")
        if cars.isna().any() or len(group) not in (6, 7, 8, 9) or not cars.between(1, 9).all() or cars.nunique() != len(group):
            continue
        if "entries_number" in group:
            declared = pd.to_numeric(group.entries_number, errors="coerce").dropna()
            if len(declared) and (declared.nunique() != 1 or declared.iloc[0] != len(group)):
                continue
        if "player_id" in group and group.player_id.notna().any() and group.player_id.nunique() != len(group):
            continue
        finish = pd.to_numeric(group.finish_pos, errors="coerce")
        if any(finish.eq(k).sum() != 1 for k in (1, 2, 3)):
            continue
        if "official_finish_pos" in group:
            official = pd.to_numeric(group.official_finish_pos, errors="coerce")
            if any(official.eq(k).sum() != 1 for k in (1, 2, 3)):
                continue
        one = group.loc[finish.eq(1)].iloc[0]
        two = group.loc[finish.eq(2)].iloc[0]
        venue = venue_name(one["venue"])
        if venue not in EXPECTED_VENUES or group.venue.map(venue_name).nunique() != 1:
            continue
        tier, division = classify(one)
        if tier == "unknown":
            continue
        if "race_class" in group:
            tiers = {classify(r)[0] for r in group.to_dict("records")}
            if tiers != {tier}:
                continue
        hour = jst_hour(one.get("start_at"))
        method_raw = one.get("result_factor")
        method = winner_method(method_raw) if pd.notna(method_raw) else None
        line_top2 = None
        if "line_id" in group and "line_verification_status" in group:
            if group.line_verification_status.astype(str).eq("verified").all():
                first_line, second_line = one.get("line_id"), two.get("line_id")
                if pd.notna(first_line) and pd.notna(second_line):
                    line_top2 = bool(str(first_line) == str(second_line))
        rows.append({
            "race_id": str(rid), "date": one["_date"], "venue": venue,
            "tier": tier, "division": division, "field": len(group), "hour": hour,
            "band": time_band(hour), "method": method,
            "raw_factor": str(method_raw) if pd.notna(method_raw) else "",
            "forecast_line_top2": line_top2,
        })
    return pd.DataFrame(rows, columns=["race_id", "date", "venue", "tier", "division",
                                       "field", "hour", "band", "method", "raw_factor", "forecast_line_top2"])


class TacticModel:
    """Hierarchical Dirichlet smoothing: time effects have strong venue priors."""
    def __init__(self, races):
        self.counts = defaultdict(Counter)
        observed = races[races.method.isin(METHODS)]
        for r in observed.itertuples(index=False):
            keys = self.keys(r)
            for level, key in keys.items():
                self.counts[(level, key)][r.method] += 1
        self.training_races = len(races)
        self.known_methods = len(observed)

    @staticmethod
    def keys(r):
        d = (r.division, int(r.field))
        result = {"all": (), "national": d,
                  "venue": (r.venue, *d)}
        if r.band != "unknown":
            result["venue_band"] = (r.venue, *d, r.band)
            if r.hour is not None and not pd.isna(r.hour):
                result["venue_hour"] = (r.venue, *d, int(r.hour))
        return result

    def predict(self, r):
        keys = self.keys(r)
        all_counts = self.counts[("all", ())]
        total = sum(all_counts.values())
        p = {m: (all_counts[m] + 1) / (total + len(METHODS)) for m in METHODS}
        predictions = {}
        # Compare national -> all-venue -> coarse time -> exact hour at equal input.
        for level, strength in (("national", 100), ("venue", 100),
                                ("venue_band", 120), ("venue_hour", 160)):
            counts = self.counts[(level, keys[level])] if level in keys else Counter()
            n = sum(counts.values())
            p = {m: (counts[m] + strength * p[m]) / (n + strength) for m in METHODS}
            predictions[level] = p.copy()
        return predictions

    def export(self):
        return {"version": VERSION, "trained_races": self.training_races,
                "trained_with_observed_winner_method": self.known_methods,
                "method_values": list(METHODS),
                "counts": [{"level": level, "key": list(key), "n": sum(c.values()),
                            "methods": {m: c[m] for m in METHODS}}
                           for (level, key), c in sorted(self.counts.items(), key=lambda x: (x[0][0], str(x[0][1])))]}


def assess(races, asof, min_history=500):
    eligible = races[races.method.isin(METHODS)].copy()
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=365)
    targets = eligible[eligible.date.ge(cutoff)].sort_values(["date", "race_id"])
    scored = []
    for month, chunk in targets.groupby(targets.date.dt.to_period("M"), sort=True):
        first_date = chunk.date.min()
        history = eligible[eligible.date.lt(first_date)]
        if len(history) < min_history:
            continue
        model = TacticModel(history)
        for r in chunk.itertuples(index=False):
            p = model.predict(r)
            scored.append({"race_id": r.race_id, "date": r.date.strftime("%Y-%m-%d"),
                           "venue": r.venue, "division": r.division, "field": r.field,
                           "band": r.band, "method": r.method,
                           **{f"loss_{name}": -math.log(max(1e-12, prob[r.method])) for name, prob in p.items()},
                           **{f"hit_{name}": int(max(METHODS, key=lambda m: prob[m]) == r.method)
                              for name, prob in p.items()}})
    detail = pd.DataFrame(scored)
    summary = {"eval_start_inclusive": cutoff.strftime("%Y-%m-%d"),
               "eval_end_exclusive": str(pd.Timestamp(asof).date()),
               "evaluated_races": int(len(detail)), "time_leakage_policy": "monthly chronological; fit strictly before each evaluation date",
               "metrics": {}, "paired_daily_loss_advantage_ci95": None}
    if len(detail):
        for name in LEVELS:
            summary["metrics"][name] = {
                "mean_logloss": round(float(detail[f"loss_{name}"].mean()), 6),
                "winner_tactic_accuracy": round(float(detail[f"hit_{name}"].mean()), 6)}
        daily = (detail.assign(advantage=detail.loss_national-detail.loss_venue_hour)
                 .groupby("date", sort=True).advantage.mean().to_numpy())
        if len(daily) >= 15:
            rng = np.random.default_rng(20261013)
            boot = [float(rng.choice(daily, len(daily), replace=True).mean()) for _ in range(600)]
            summary["paired_daily_loss_advantage_ci95"] = [round(float(x), 6) for x in np.quantile(boot, [.025, .975])]
    return detail, summary


def table(races, model):
    rows = []
    for venue in EXPECTED_VENUES:
        subset = races[races.venue.eq(venue)]
        known = subset[subset.method.isin(METHODS)]
        row = {"venue": venue, "history_races": len(subset), "method_coverage_races": len(known),
               "status": "learned" if len(known) >= 30 else "low_sample" if len(known) else "no_method_data"}
        for m in METHODS:
            row[f"raw_{m}_rate"] = float(known.method.eq(m).mean()) if len(known) else None
        rows.append(row)
    venues = pd.DataFrame(rows)
    strat = races.groupby(["venue", "tier", "division", "field", "band"], dropna=False)
    cells = []
    for key, group in strat:
        known = group[group.method.isin(METHODS)]
        base = {"venue": key[0], "tier": key[1], "division": key[2], "field": key[3],
                "band": key[4], "races": len(group), "known_methods": len(known),
                "same_forecast_line_top2_n": int(group.forecast_line_top2.notna().sum()),
                "same_forecast_line_top2_rate": float(group.forecast_line_top2.dropna().astype(float).mean())
                if group.forecast_line_top2.notna().any() else None}
        for m in METHODS:
            base[f"raw_{m}_rate"] = float(known.method.eq(m).mean()) if len(known) else None
        # Smoothed venue/time expectation is more reliable than small-cell raw rates.
        ref = group.iloc[-1]
        model_probs = model.predict(ref)["venue_band"]
        for m in METHODS:
            base[f"smoothed_{m}_rate"] = model_probs[m]
        cells.append(base)
    return venues, pd.DataFrame(cells)


def today_shadow(today, model):
    if today is None or today.empty or not {"race_id", "venue", "date", "car_no"}.issubset(today):
        return pd.DataFrame()
    out = []
    for rid, group in today.groupby("race_id", sort=False):
        if group.car_no.nunique() != len(group):
            continue
        first = group.iloc[0]
        tier, division = classify(first)
        if tier == "unknown":
            continue
        venue = venue_name(first["venue"])
        if venue not in EXPECTED_VENUES:
            continue
        hour = jst_hour(first.get("start_at"))
        r = type("PreRace", (), {"division": division, "field": len(group), "venue": venue,
                                 "band": time_band(hour), "hour": hour})()
        probs = model.predict(r)["venue_hour"]
        out.append({"race_id": str(rid), "date": str(first["date"]), "venue": venue,
                    "tier": tier, "division": division, "field": len(group),
                    "band": r.band, "hour_jst": hour, "purchase_authorized": False,
                    **{f"p_win_by_{m}": round(probs[m], 6) for m in METHODS}})
    return pd.DataFrame(out)


def run(history, out_dir, asof, today_path=None, min_races=5000):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    races = to_races(load_history(history), asof)
    model = TacticModel(races)
    report = {"version": VERSION, "asof_exclusive": str(pd.Timestamp(asof).date()),
              "history_races_clean": int(len(races)), "observed_winner_tactics": int(model.known_methods),
              "method_coverage_ratio": float(model.known_methods / len(races)) if len(races) else 0,
              "venues_expected": len(EXPECTED_VENUES), "venues_in_history": int(races.venue.nunique()),
              "venues_with_winner_methods": int(races.loc[races.method.isin(METHODS), "venue"].nunique()),
              "venues_without_races": sorted(set(EXPECTED_VENUES)-set(races.venue)),
              "unrecognized_winner_factors": races.loc[races.method.isna() & races.raw_factor.ne(""), "raw_factor"].value_counts().head(20).to_dict(),
              "source_rule": "official-looking result_factor on actual first finisher only; no inferred tactic from style",
              "time_rule": "JST launch hour: <11 morning, 11-15 day, 16-20 evening, 21+ late; not official meeting category",
              "forecast_line_rule": "provider-predicted line positions, NOT verified actual in-race formation",
              "production_modified": False, "autopromotion": False, "betting_return_validated": False}
    score, backtest = assess(races, asof)
    report["chronological_validation"] = backtest
    report["status"] = ("ready_for_shadow_review" if len(races) >= min_races
                        and model.known_methods >= 1000 and len(score) >= 200
                        else "insufficient_data_do_not_apply")
    venues, cells = table(races, model) if len(races) else (pd.DataFrame(), pd.DataFrame())
    venues.to_csv(out/"all_venues.csv", index=False)
    cells.to_csv(out/"venue_grade_field_time.csv", index=False)
    score.to_csv(out/"chronological_holdout.csv", index=False)
    (out/"shadow_tactics_model.json").write_text(json.dumps(model.export(), ensure_ascii=False, indent=2, allow_nan=False))
    if today_path and Path(today_path).is_file() and model.known_methods:
        shadow = today_shadow(pd.read_csv(today_path, dtype={"race_id": str}, low_memory=False), model)
    else:
        shadow = pd.DataFrame()
    shadow.to_csv(out/"today_shadow.csv", index=False)
    (out/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/raw/history.csv")
    parser.add_argument("--today", default="data/raw/today_entries.csv")
    parser.add_argument("--asof", default=datetime.now(JST).strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default="research/venue_time_latest")
    parser.add_argument("--min-races", type=int, default=5000)
    args = parser.parse_args()
    result = run(args.history, args.out_dir, args.asof, args.today, args.min_races)
    print(json.dumps({k: v for k, v in result.items() if k not in ("chronological_validation", "unrecognized_winner_factors")}, ensure_ascii=False))
    print(json.dumps(result["chronological_validation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
