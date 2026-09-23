import json
import os
import subprocess
import sys
import hashlib
from datetime import datetime
from itertools import permutations
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from common import (
    ensure_dirs,
    HISTORY_CSV,
    RAW_DIR,
    TODAY_CSV,
    TODAY_ODDS_CSV,
    MODEL_PATH,
    OUTPUT_DIR,
    add_player_prior_features,
    prepare_features,
    normalize_race_prob,
)
from multi_bet_backtest import BET_LABELS, make_candidates as make_multi_bet_candidates

TRIFECTA_ODDS_CSV = RAW_DIR / "today_trifecta_odds.csv"
PROFIT_GATE_PATH = OUTPUT_DIR / "external_holdout_overall.json"
DEFAULT_BET_CONFIGS = {
    "exacta": {"min_prob": 0.05, "min_ev": 800, "max_odds": 200, "max_per_race": 2},
    "quinella": {"min_prob": 0.05, "min_ev": 1200, "max_odds": 300, "max_per_race": 2},
    "quinella_place": {"min_prob": 0.10, "min_ev": 300, "max_odds": 100, "max_per_race": 2},
    "trio": {"min_prob": 0.07, "min_ev": 800, "max_odds": 300, "max_per_race": 2},
    "trifecta": {"min_prob": 0.07, "min_ev": 800, "max_odds": 300, "max_per_race": 2},
}


def get_stake_yen():
    raw = os.getenv("BET_STAKE_YEN", "100").strip()
    try:
        stake = int(raw)
    except ValueError:
        stake = 100
    return max(stake, 0)


def get_float_env(name, default):
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return float(default)


def get_int_env(name, default):
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profit_gate():
    if not PROFIT_GATE_PATH.exists():
        return {"target_passed": False, "reason": "external holdout has not been completed"}
    try:
        result = json.loads(PROFIT_GATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"target_passed": False, "reason": "external holdout result is unreadable"}
    if not bool(result.get("target_passed", False)):
        return {
            "target_passed": False,
            "reason": "external holdout ROI target was not met",
            "roi": result.get("roi"),
            "bets": result.get("bets"),
        }
    return {"target_passed": True, "reason": "external holdout gate passed", **result}


def stake_from_edge(expected_profit_100yen, base_stake=100, max_stake=500):
    if pd.isna(expected_profit_100yen):
        return base_stake
    if expected_profit_100yen < 100:
        return base_stake
    extra_units = int(min((expected_profit_100yen // 300), (max_stake - base_stake) // 100))
    return int(base_stake + extra_units * 100)


def allocate_daily_budget(bets, budget_yen=10000, max_per_bet_yen=2000):
    """Allocate in 100-yen units by probability and positive expected edge."""
    if bets.empty:
        return bets.assign(stake_yen=pd.Series(dtype=int))
    result = bets.copy()
    budget = max(int(budget_yen), 0) // 100 * 100
    cap = max(int(max_per_bet_yen), 100) // 100 * 100
    edge = pd.to_numeric(result["expected_profit_100yen"], errors="coerce").clip(lower=0)
    probability = pd.to_numeric(result["prob"], errors="coerce").clip(lower=0, upper=1)
    result["_allocation_weight"] = edge * probability
    result = result.sort_values("_allocation_weight", ascending=False, kind="mergesort")
    result["stake_yen"] = 0
    # First reserve the minimum for the strongest candidates; remaining funds
    # are assigned in 100-yen increments using the same confidence/edge weight.
    eligible = result[result["_allocation_weight"].gt(0)].index.tolist()
    eligible = eligible[: min(len(eligible), budget // 100)]
    result.loc[eligible, "stake_yen"] = 100
    remaining_units = max(budget // 100 - len(eligible), 0)
    while remaining_units and eligible:
        active = [i for i in eligible if result.at[i, "stake_yen"] < cap]
        if not active:
            break
        weights = result.loc[active, "_allocation_weight"]
        if float(weights.sum()) <= 0:
            break
        # Weighted water filling: each next unit goes to the ticket furthest
        # below its target share, bounded by the per-ticket cap.
        chosen = min(active, key=lambda i: (result.at[i, "stake_yen"] / result.at[i, "_allocation_weight"], str(i)))
        result.at[chosen, "stake_yen"] += 100
        remaining_units -= 1
    return result.drop(columns=["_allocation_weight"]).sort_index()


def prior_daily_stake(today_tag, replacing_race_ids):
    """Reserve stakes from earlier same-day snapshots for other races."""
    latest_by_race = {}
    paths = sorted(OUTPUT_DIR.glob(f"shadow_bets_{today_tag}*.csv"), key=lambda p: p.stat().st_mtime)
    replacing = {str(race_id) for race_id in replacing_race_ids}
    for path in paths:
        try:
            frame = pd.read_csv(path, dtype={"race_id": str, "buy": str, "bet_type": str})
        except (OSError, pd.errors.ParserError):
            continue
        if frame.empty or not {"race_id", "stake_yen"}.issubset(frame.columns):
            continue
        for race_id, group in frame.groupby("race_id", sort=False):
            if str(race_id) not in replacing:
                latest_by_race[str(race_id)] = group.copy()
    used = sum(
        pd.to_numeric(group.get("stake_yen", 0), errors="coerce").fillna(0).sum()
        for group in latest_by_race.values()
    )
    return int(used)


def ensure_ready():
    if not TODAY_CSV.exists():
        print("today_entries.csv not found; generating sample data")
        subprocess.check_call([sys.executable, "src/make_sample_data.py", "--if-missing"])

    if not MODEL_PATH.exists():
        print("model not found; training model")
        subprocess.check_call([sys.executable, "src/train.py"])


def make_trifecta_candidates(race_df: pd.DataFrame, top_k_riders=5):
    riders = race_df.sort_values("p_win", ascending=False).head(top_k_riders)
    rows = riders[["car_no", "p_win"]].to_dict("records")
    results = []

    for a, b, c in permutations(rows, 3):
        p1 = float(a["p_win"])
        denom2 = max(1.0 - p1, 1e-9)
        p2 = float(b["p_win"]) / denom2
        denom3 = max(1.0 - p1 - float(b["p_win"]), 1e-9)
        p3 = float(c["p_win"]) / denom3
        prob = max(0.0, min(1.0, p1 * p2 * p3))

        results.append({
            "buy": f'{int(a["car_no"])}-{int(b["car_no"])}-{int(c["car_no"])}',
            "trifecta_prob_approx": prob,
        })

    results = sorted(results, key=lambda x: x["trifecta_prob_approx"], reverse=True)
    return results


def load_trifecta_odds():
    if not TRIFECTA_ODDS_CSV.exists():
        return pd.DataFrame(columns=["race_id", "buy", "trifecta_odds"])
    odds = pd.read_csv(TRIFECTA_ODDS_CSV, dtype={"race_id": str, "buy": str})
    if "race_id" not in odds.columns or "buy" not in odds.columns:
        return pd.DataFrame(columns=["race_id", "buy", "trifecta_odds"])
    odds["race_id"] = odds["race_id"].astype(str)
    odds["buy"] = odds["buy"].astype(str)
    odds["trifecta_odds"] = pd.to_numeric(odds.get("trifecta_odds", np.nan), errors="coerce")
    return odds[["race_id", "buy", "trifecta_odds"]]


def load_today_odds():
    if TODAY_ODDS_CSV.exists():
        odds = pd.read_csv(TODAY_ODDS_CSV, dtype={"race_id": str, "buy": str, "bet_type": str})
        required = ["race_id", "bet_type", "buy", "odds_used"]
        if all(c in odds.columns for c in required):
            odds["race_id"] = odds["race_id"].astype(str)
            odds["buy"] = odds["buy"].astype(str)
            odds["bet_type"] = odds["bet_type"].astype(str)
            odds["odds_used"] = pd.to_numeric(odds["odds_used"], errors="coerce")
            return odds[required]

    trifecta = load_trifecta_odds()
    if len(trifecta):
        trifecta = trifecta.rename(columns={"trifecta_odds": "odds_used"})
        trifecta["bet_type"] = "trifecta"
        return trifecta[["race_id", "bet_type", "buy", "odds_used"]]
    return pd.DataFrame(columns=["race_id", "bet_type", "buy", "odds_used"])


def add_today_prior_features(df):
    if not HISTORY_CSV.exists():
        return df
    hist = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    df = df.copy()
    hist["_today_row_id"] = np.nan
    df["_today_row_id"] = np.arange(len(df))
    if "finish_pos" not in df.columns:
        df["finish_pos"] = np.nan
    combined = pd.concat([hist, df], ignore_index=True, sort=False)
    combined = add_player_prior_features(combined)
    today = combined[combined["_today_row_id"].notna()].copy()
    today = today.sort_values("_today_row_id", kind="mergesort")
    return today.drop(columns=["_today_row_id"], errors="ignore")


def apply_nexus_race_reading(pred):
    """Blend model probability with pre-race, leakage-safe race-development signals.

    Only observed fields are used. Unobserved behaviour rates (tsuppari/kamashi/
    tobitsuki/chigire/seri, etc.) are intentionally left neutral until enough
    measured events exist; they must never be fabricated.
    """
    pred = pred.copy()
    def num(col):
        return pd.to_numeric(pred[col], errors="coerce") if col in pred.columns else pd.Series(np.nan, index=pred.index)

    score = num("score")
    recent = num("recent_avg_finish")
    current = num("current_cup_avg_finish")
    line_pos = num("line_position")
    line_size = num("line_size")
    line_role = num("line_role_win_rate")
    track = num("track_win_rate")
    weather = num("weather_win_rate")
    hour = num("hour_win_rate")
    wind = num("wind_speed")
    back = num("back_count")
    front = num("front_runner_count")
    stalker = num("stalker_count")
    closer = num("deep_closer_count")
    marker = num("marker_count")

    def z_by_race(x):
        return x.groupby(pred["race_id"]).transform(
            lambda g: (g - g.mean()) / (g.std(ddof=0) if pd.notna(g.std(ddof=0)) and g.std(ddof=0) > 1e-9 else 1.0)
        ).fillna(0.0)

    # 27-point philosophy mapped only to currently observed, pre-race fields:
    # current-meeting/recent form > old reputation; line role without blind line
    # trust; solo/leader/second-wheel context; style and B/front-run evidence;
    # track/weather/time/wind compatibility. Odds remain outside this adjustment.
    form = (-z_by_race(recent) * 0.22) + (-z_by_race(current) * 0.34) + (z_by_race(score) * 0.10)
    line = z_by_race(line_role) * 0.12
    line += ((line_pos == 2) & (line_size >= 2)).astype(float) * 0.05
    line += ((line_pos == 1) & (line_size >= 2)).astype(float) * 0.025
    line -= (line_size == 1).astype(float) * 0.015

    style_pressure = z_by_race(back.fillna(0) + front.fillna(0)) * 0.045
    style_finish = z_by_race(stalker.fillna(0) + closer.fillna(0) + marker.fillna(0)) * 0.035
    condition = z_by_race(track) * 0.05 + z_by_race(weather) * 0.025 + z_by_race(hour) * 0.015
    # Strong wind increases uncertainty rather than pretending to know direction.
    uncertainty = wind.fillna(0).clip(lower=0) * 0.004

    pred["nexus_form_adj"] = form
    pred["nexus_line_adj"] = line
    pred["nexus_style_adj"] = style_pressure + style_finish
    pred["nexus_condition_adj"] = condition
    pred["nexus_uncertainty"] = uncertainty

    logp = np.log(pred["p_raw"].clip(1e-6, 1.0))
    pred["p_nexus_raw"] = np.exp(
        logp + pred["nexus_form_adj"] + pred["nexus_line_adj"]
        + pred["nexus_style_adj"] + pred["nexus_condition_adj"]
        - pred["nexus_uncertainty"]
    )
    pred = normalize_race_prob(pred, "p_nexus_raw", "p_win")
    return pred


def main():
    ensure_dirs()
    ensure_ready()
    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    prediction_epoch = now_jst.timestamp()
    base_stake_yen = get_int_env("BET_BASE_STAKE_YEN", get_stake_yen())
    max_stake_yen = get_int_env("BET_MAX_STAKE_YEN", 500)
    max_seconds_to_close = get_int_env("BET_MAX_SECONDS_TO_CLOSE", 3600)
    snapshot_mode = os.getenv("PREDICTION_SNAPSHOT_MODE", "false").strip().lower() in {"1", "true", "yes"}
    strategy_version = os.getenv("PREDICTION_STRATEGY_VERSION", "near_close_v2_20260805").strip()
    odds_snapshot_label = os.getenv("ODDS_SNAPSHOT_LABEL", "near-close snapshot").strip()
    profit_gate = load_profit_gate()

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    fill_values = bundle["fill_values"]

    df = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
    if "race_id" not in df.columns:
        raise ValueError("today_entries.csv must have race_id column")
    df = add_today_prior_features(df)

    X, _ = prepare_features(df, fill_values)

    raw = model.predict_proba(X)[:, 1]
    calibrated = calibrator.predict(raw) if calibrator is not None else raw

    pred = df.copy()
    pred["p_raw"] = np.clip(calibrated, 1e-6, 1.0)
    pred = apply_nexus_race_reading(pred)
    if "odds_win" not in pred.columns:
        pred["odds_win"] = np.nan
    pred["odds_win"] = pd.to_numeric(pred["odds_win"], errors="coerce")
    valid_win_odds = pred["odds_win"].gt(0) & np.isfinite(pred["odds_win"])
    pred["expected_value_win"] = (pred["p_win"] * pred["odds_win"] - 1).where(valid_win_odds)
    pred["stake_yen"] = np.where(valid_win_odds, base_stake_yen, 0)
    pred["win_return_yen"] = (base_stake_yen * pred["odds_win"]).round(0).where(valid_win_odds)
    pred["win_profit_yen"] = (pred["win_return_yen"] - base_stake_yen).where(valid_win_odds)
    pred["loss_amount_yen"] = np.where(valid_win_odds, base_stake_yen, 0)
    pred["expected_profit_yen"] = (base_stake_yen * pred["expected_value_win"]).round(0)
    pred["rank_in_race"] = pred.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)

    pred["_start_sort"] = pd.to_numeric(pred.get("start_at", np.nan), errors="coerce").fillna(float("inf"))
    sort_cols = ["date", "_start_sort", "venue", "race_no", "rank_in_race"]
    pred = pred.sort_values(sort_cols, kind="mergesort")

    today_jst = now_jst.strftime("%Y%m%d")
    output_tag = now_jst.strftime("%Y%m%d_%H%M%S") if snapshot_mode else today_jst
    pred_path = OUTPUT_DIR / f"predictions_{output_tag}.csv"
    latest_path = OUTPUT_DIR / "latest_predictions.csv"

    cols = [
        "date", "venue", "race_no", "race_id", "start_at", "close_at", "rank_in_race",
        "car_no", "player_id", "style", "score", "odds_win",
        "p_win", "nexus_form_adj", "nexus_line_adj", "nexus_style_adj", "nexus_condition_adj", "nexus_uncertainty", "expected_value_win", "stake_yen", "win_return_yen",
        "win_profit_yen", "loss_amount_yen", "expected_profit_yen",
    ]
    for c in cols:
        if c not in pred.columns:
            pred[c] = ""

    pred[cols].to_csv(pred_path, index=False)
    pred[cols].to_csv(latest_path, index=False)

    bet_rows = []
    today_odds = load_today_odds()
    for race_id, g in pred.groupby("race_id", sort=False):
        base = g.iloc[0]
        close_at = pd.to_numeric(base.get("close_at", np.nan), errors="coerce")
        seconds_to_close = close_at - prediction_epoch if pd.notna(close_at) else np.nan
        if pd.isna(seconds_to_close) or seconds_to_close <= 300 or seconds_to_close > max_seconds_to_close:
            continue
        candidates = make_multi_bet_candidates(g, top_k=5)
        race_odds = today_odds[today_odds["race_id"].eq(str(race_id))]
        if len(candidates) == 0 or len(race_odds) == 0:
            continue
        merged = candidates.merge(race_odds, on=["bet_type", "buy"], how="inner")
        if len(merged) == 0:
            continue
        merged = merged.sort_values(["bet_type", "prob"], ascending=[True, False])
        merged["candidate_rank"] = merged.groupby("bet_type")["prob"].rank(ascending=False, method="first").astype(int)
        for _, cand in merged.iterrows():
            expected_value = cand["prob"] * cand["odds_used"] - 1 if pd.notna(cand["odds_used"]) else np.nan
            bet_rows.append({
                "date": base.get("date", ""),
                "venue": base.get("venue", ""),
                "race_no": base.get("race_no", ""),
                "race_id": race_id,
                "start_at": base.get("start_at", np.nan),
                "close_at": close_at,
                "seconds_to_close_at_prediction": round(float(seconds_to_close)),
                "strategy_version": strategy_version,
                "bet_type": cand["bet_type"],
                "bet_label": BET_LABELS.get(cand["bet_type"], cand["bet_type"]),
                "candidate_rank": int(cand["candidate_rank"]),
                "buy": cand["buy"],
                "prob": cand["prob"],
                "odds_used": cand["odds_used"],
                "expected_profit_100yen": round(100 * expected_value) if pd.notna(expected_value) else np.nan,
            })

    candidates = pd.DataFrame(bet_rows)
    if len(candidates):
        candidates["is_selected"] = False
        for bet_type, config in DEFAULT_BET_CONFIGS.items():
            mask = candidates["bet_type"].eq(bet_type)
            candidates.loc[mask, "is_selected"] = (
                (pd.to_numeric(candidates.loc[mask, "prob"], errors="coerce") >= config["min_prob"])
                & (pd.to_numeric(candidates.loc[mask, "expected_profit_100yen"], errors="coerce") >= config["min_ev"])
                & (pd.to_numeric(candidates.loc[mask, "odds_used"], errors="coerce") <= config["max_odds"])
            )
        selected_parts = []
        for (_, bet_type), g in candidates[candidates["is_selected"]].groupby(["race_id", "bet_type"], sort=False):
            max_per_race = DEFAULT_BET_CONFIGS.get(bet_type, {}).get("max_per_race", 1)
            selected_parts.append(g.sort_values("expected_profit_100yen", ascending=False).head(max_per_race))
        bets = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates.head(0).copy()
        if len(bets):
            daily_budget_yen = get_int_env("BET_DAILY_BUDGET_YEN", 10000)
            reserved_yen = prior_daily_stake(today_jst, pred["race_id"].astype(str).unique())
            remaining_budget_yen = max(daily_budget_yen - reserved_yen, 0)
            bets = allocate_daily_budget(bets, remaining_budget_yen, max_stake_yen)
            bets = bets[pd.to_numeric(bets["stake_yen"], errors="coerce").ge(100)].copy()
        else:
            reserved_yen = prior_daily_stake(today_jst, pred["race_id"].astype(str).unique())
            remaining_budget_yen = max(get_int_env("BET_DAILY_BUDGET_YEN", 10000) - reserved_yen, 0)
        if len(bets):
            bets["return_if_hit_yen"] = (bets["stake_yen"] * pd.to_numeric(bets["odds_used"], errors="coerce")).round(0)
            bets["profit_if_hit_yen"] = bets["return_if_hit_yen"] - bets["stake_yen"]
            bets["loss_amount_yen"] = bets["stake_yen"]
            bets["expected_profit_yen"] = (bets["stake_yen"] * pd.to_numeric(bets["expected_profit_100yen"], errors="coerce") / 100).round(0)
    else:
        candidates = pd.DataFrame(columns=["date", "venue", "race_no", "race_id", "bet_type", "bet_label", "candidate_rank", "buy"])
        bets = candidates.copy()

    shadow_bets = bets.copy()
    if len(shadow_bets):
        shadow_bets["purchase_authorized"] = bool(profit_gate["target_passed"])
        shadow_bets["authorization_reason"] = profit_gate["reason"]
        shadow_bets["model_sha256"] = file_sha256(MODEL_PATH)
        shadow_bets["prediction_created_at_jst"] = now_jst.isoformat(timespec="seconds")
        shadow_bets["odds_snapshot"] = odds_snapshot_label
    shadow_path = OUTPUT_DIR / f"shadow_bets_{output_tag}.csv"
    latest_shadow_path = OUTPUT_DIR / "latest_shadow_bets.csv"
    shadow_bets.to_csv(shadow_path, index=False)
    shadow_bets.to_csv(latest_shadow_path, index=False)
    if not profit_gate["target_passed"]:
        bets = bets.head(0).copy()

    bets_path = OUTPUT_DIR / f"bets_{output_tag}.csv"
    latest_bets_path = OUTPUT_DIR / "latest_bets.csv"
    latest_candidates_path = OUTPUT_DIR / "latest_bet_candidates.csv"
    candidates.to_csv(latest_candidates_path, index=False)
    bets.to_csv(bets_path, index=False)
    bets.to_csv(latest_bets_path, index=False)

    html_path = OUTPUT_DIR / "index.html"
    top_table = pred[cols].head(100).copy()
    top_table["odds_win"] = pd.to_numeric(top_table["odds_win"], errors="coerce").map(
        lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "未取得"
    )
    top_table = top_table.fillna("未取得")
    for col in ["p_win", "expected_value_win"]:
        top_table[col] = pd.to_numeric(top_table[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    race_cards = []
    for (venue, race_no, race_id), group in pred.groupby(["venue", "race_no", "race_id"], sort=False):
        base_row = group.iloc[0]
        start_epoch = pd.to_numeric(base_row.get("start_at", np.nan), errors="coerce")
        close_epoch = pd.to_numeric(base_row.get("close_at", np.nan), errors="coerce")
        start_label = datetime.fromtimestamp(float(start_epoch), ZoneInfo("Asia/Tokyo")).strftime("%H:%M") if pd.notna(start_epoch) else ""
        seconds_to_close_ui = float(close_epoch) - prediction_epoch if pd.notna(close_epoch) else np.nan
        timing_label = "締め切り間近" if pd.notna(seconds_to_close_ui) and 0 < seconds_to_close_ui <= 1800 else (f"{start_label} 発走" if start_label else "時刻確認中")
        leaders = group.sort_values("rank_in_race").head(3)
        picks = " / ".join(
            f'<span class="car car-{int(row.car_no)}">{int(row.car_no)}</span> {float(row.p_win):.1%}'
            for row in leaders.itertuples()
        )
        race_bets = shadow_bets[shadow_bets["race_id"].astype(str).eq(str(race_id))] if len(shadow_bets) else shadow_bets
        if len(race_bets):
            bet_html = "".join(
                f'<div class="bet"><b>{row.get("bet_label", row.get("bet_type", ""))}</b>'
                f'<strong>{row["buy"]}</strong><span>{int(row["stake_yen"]):,}円</span>'
                f'<small>オッズ {float(row["odds_used"]):.1f}</small></div>'
                for _, row in race_bets.iterrows()
            )
        else:
            bet_html = '<div class="waiting">買い目候補は締切前オッズ取得後に表示</div>'
        riders_html = "".join(
            f'<button class="rider" type="button" data-car="{int(row.car_no)}" data-name="{row.player_id}" data-score="{float(row.score) if pd.notna(row.score) else 0:.1f}" data-win="{float(row.p_win)*100:.1f}" onclick="compareRider(this)"><i class="car car-{int(row.car_no)}">{int(row.car_no)}</i><span>{row.player_id}</span></button>'
            for row in group.sort_values("car_no").itertuples()
        )
        race_cards.append(
            f'<article class="race" id="race-{race_id}" data-venue="{venue}" data-start="{float(start_epoch) if pd.notna(start_epoch) else 99999999999}"><button class="race-select" type="button" onclick="selectRace(this)">'
            f'<span><b>{venue} {int(race_no)}R</b><small>{timing_label}</small></span><em>›</em></button>'
            f'<div class="race-detail"><div class="race-head"><span>AI上位　{picks}</span></div>'
            f'<div class="race-actions"><button type="button" onclick="showPanel(this, \'picks\')">買い目</button>'
            f'<button type="button" onclick="showPanel(this, \'flow\')">展開予想</button><button type="button" onclick="showPanel(this, \'compare\')">選手比較</button></div>'
            f'<div class="race-panel picks-panel"><h3>買い目</h3>{bet_html}</div>'
            f'<div class="race-panel flow-panel"><h3>展開予想</h3><p>AI上位　{picks}</p></div>'
            f'<div class="race-panel compare-panel"><h3>選手能力比較</h3><div class="riders">{riders_html}</div><div class="compare-view"><div class="donut" style="--v:0"><span>0%</span></div><div><b class="compare-name">選手を選択</b><p>AI勝率 <strong class="compare-win">--</strong></p><p>競走得点 <strong class="compare-score">--</strong></p></div></div></div></div></article>'
        )
    generated = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y/%m/%d %H:%M")
    html = f"""
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS | 今日の競輪予想</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f8fd;color:#10233f;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{background:#fff;color:#17395f;padding:20px 18px 16px;border-bottom:5px solid #1d7de8}}.brand{{display:inline-block;background:linear-gradient(135deg,#07121f,#0b263e);color:#fff;border-radius:9px;padding:15px 26px;font-size:25px;font-weight:900;letter-spacing:.1em;min-width:250px;box-shadow:0 7px 22px #0b5bd326}}.brand:before{{content:"N";color:#35e2e7;font-size:34px;margin-right:12px}}.tag{{display:inline-block;color:#1680ea;font-size:10px;font-weight:800;letter-spacing:.16em;margin-left:12px}}
main{{max-width:920px;margin:auto;padding:22px 15px 90px}}nav{{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin:0 0 18px;background:#fff;border-bottom:4px solid #1d7de8}}nav a{{text-align:center;text-decoration:none;color:#1680ea;background:#fff;padding:15px 5px;font-weight:800}}
.hero{{background:linear-gradient(135deg,#071f43,#0d4f9f);color:white;border-radius:20px;padding:20px;margin-bottom:16px;display:flex;justify-content:space-between;gap:16px;align-items:center;box-shadow:0 10px 28px #0b4f9b22}}.eyebrow{{font-size:10px;letter-spacing:.18em;color:#9fcfff;font-weight:900}}.hero h1{{font-size:23px;margin:5px 0}}.hero p{{font-size:12px;opacity:.78;margin:0}}.trust{{background:#ffffff12;border:1px solid #ffffff2e;border-radius:14px;padding:11px 13px;min-width:190px}}.trust b{{display:block;font-size:12px}}.trust small{{display:block;font-size:10px;opacity:.75;margin-top:3px}}
.venue-jump{{background:#fff;border:1px solid #dce8f5;border-radius:18px;padding:14px;margin-bottom:12px}}.race{{display:none}}.race.venue-visible{{display:block}}.venue-jump>b{{font-size:13px;color:#10233f}}#venueJump{{display:flex;gap:8px;overflow-x:auto;padding-top:10px}}#venueJump button{{white-space:nowrap;border:1px solid #cfe1f4;background:#f7fbff;color:#1679e8;border-radius:999px;padding:9px 14px;font-weight:900}}.race{{background:white;border:1px solid #dce8f5;border-radius:18px;padding:15px;margin:11px 0;box-shadow:0 7px 22px #173d7010;transition:.18s ease}}.race.open{{border-color:#9bc7f5;box-shadow:0 10px 28px #176fc51c}}.race-head{{display:flex;justify-content:space-between;gap:10px;align-items:center;border-bottom:1px solid #edf2f8;padding-bottom:11px}}.race-head>b{{font-size:18px}}.race-head>span{{font-size:12px;color:#60728b}}
.car{{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;font-weight:900;border:1px solid #ccd6e4;background:#fff;color:#111}}.car-2{{background:#222;color:#fff}}.car-3{{background:#e53935;color:#fff}}.car-4{{background:#1769d2;color:#fff}}.car-5{{background:#f0c52d}}.car-6{{background:#45a95a;color:#fff}}.car-7{{background:#f18b28;color:#fff}}
.bet{{display:grid;grid-template-columns:60px 1fr auto;gap:8px;align-items:center;padding:12px 0 0}}.bet b{{font-size:12px;color:#0b5bd3}}.bet strong{{font-size:20px;letter-spacing:.04em}}.bet span{{font-weight:800}}.bet small{{grid-column:2/4;color:#718096}}.waiting{{padding-top:12px;color:#7a899d;font-size:13px}}.race-select{{width:100%;border:0;background:#fff;color:#10233f;display:flex;justify-content:space-between;align-items:center;padding:2px;cursor:pointer;text-align:left}}.race-select b{{font-size:19px}}.race-select small{{display:block;color:#8191a4;margin-top:4px}}.race-select em{{font-style:normal;color:#1679e8;font-size:32px}}.race-detail{{display:none;padding-top:12px}}.race.open .race-detail{{display:block}}.race-actions{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}}.race-actions button{{border:1px solid #cfe1f4;background:#f7fbff;color:#1679e8;border-radius:12px;padding:11px;font-weight:800}}.race-panel{{display:none;margin-top:12px;border-top:1px solid #e6eef7;padding-top:10px}}.race-panel.active{{display:block}}.race-panel h3{{margin:0 0 8px;font-size:15px;color:#1679e8}}.riders{{display:flex;gap:7px;overflow-x:auto;padding:5px 0 12px}}.rider{{border:1px solid #d8e6f5;background:#fff;border-radius:12px;padding:7px;min-width:70px}}.rider span{{display:block;font-size:9px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.compare-view{{display:flex;align-items:center;gap:18px;background:linear-gradient(135deg,#f8fbff,#eef6ff);border:1px solid #dbeafb;border-radius:18px;padding:16px;box-shadow:inset 0 1px #fff}}.donut{{--v:0;width:110px;height:110px;border-radius:50%;background:conic-gradient(#1679e8 calc(var(--v)*1%),#e7eff8 0);display:grid;place-items:center;position:relative}}.donut:after{{content:"";position:absolute;width:76px;height:76px;border-radius:50%;background:#fff}}.donut span{{z-index:1;font-size:18px;font-weight:900;color:#1679e8}}
footer{{text-align:center;padding:24px;color:#7b899b;font-size:11px}}@media(max-width:520px){{main{{padding:12px 12px 90px}}.race-head{{align-items:flex-start;flex-direction:column}}.brand{{font-size:22px;min-width:220px}}.hero{{align-items:flex-start;flex-direction:column}}.trust{{width:100%}}}}
</style></head><body><header><div class="brand">NEXUS</div><div class="tag">KEIRIN PREDICTION SYSTEM</div></header><main>
<nav><a href="index.html">今日の予想</a><a href="history.html">予想履歴</a></nav>
<section class="hero"><div><span class="eyebrow">TODAY’S KEIRIN</span><h1>今日のレース</h1><p>更新 {generated} JST ｜ 1日予算 10,000円を基準にAIが配分</p></div><div class="trust"><b>予想は事前保存</b><small>的中・不的中を結果確定後に記録</small></div></section>
<section class="venue-jump"><b>開催場を選択</b><div id="venueJump"></div></section>
{"".join(race_cards) if race_cards else '<div class="race">本日の予想データを取得中です。</div>'}
</main><script>
function selectRace(btn){var r=btn.closest(".race");document.querySelectorAll(".race").forEach(function(x){if(x!==r)x.classList.remove("open")});r.classList.toggle("open");if(r.classList.contains("open"))setTimeout(function(){r.scrollIntoView({behavior:"smooth",block:"center"})},80)}
function buildVenueJump(){var box=document.getElementById("venueJump");if(!box)return;var cards=[...document.querySelectorAll(".race")],parent=cards[0]&&cards[0].parentNode;cards.sort(function(a,b){return (+a.dataset.start||9e12)-(+b.dataset.start||9e12)}).forEach(function(r){parent.appendChild(r)});var venues=[...new Set(cards.map(function(r){return r.dataset.venue}).filter(Boolean))];venues.forEach(function(v){var bt=document.createElement("button");bt.textContent=v;bt.onclick=function(){box.querySelectorAll("button").forEach(function(x){x.classList.remove("active")});bt.classList.add("active");cards.forEach(function(r){r.classList.toggle("venue-visible",r.dataset.venue===v);r.classList.remove("open")})};box.appendChild(bt)})}
document.addEventListener("DOMContentLoaded",buildVenueJump);
function showPanel(btn,type){var r=btn.closest(".race");r.querySelectorAll(".race-panel").forEach(function(x){x.classList.remove("active")});r.querySelector("."+type+"-panel").classList.add("active")}\nfunction compareRider(btn){var p=btn.closest(".compare-panel"),v=parseFloat(btn.dataset.win)||0;p.querySelector(".donut").style.setProperty("--v",v);p.querySelector(".donut span").textContent=v.toFixed(1)+"%";p.querySelector(".compare-name").textContent=btn.dataset.car+"番 "+btn.dataset.name;p.querySelector(".compare-win").textContent=v.toFixed(1)+"%";p.querySelector(".compare-score").textContent=btn.dataset.score}
</script><footer>NEXUS ｜ オッズ取得状況により買い目は締切前に更新されます</footer></body></html>
"""
    html_path.write_text(html, encoding="utf-8")

    summary = {
        "created": str(latest_path),
        "bets": str(latest_bets_path),
        "candidates": str(latest_candidates_path),
        "shadow_bets": str(latest_shadow_path),
        "html": str(html_path),
        "n_rows": int(len(pred)),
        "n_races": int(pred["race_id"].nunique()),
        "base_stake_yen": base_stake_yen,
        "max_stake_yen": max_stake_yen,
        "daily_budget_yen": get_int_env("BET_DAILY_BUDGET_YEN", 10000),
        "daily_reserved_yen": reserved_yen if "reserved_yen" in locals() else 0,
        "daily_remaining_budget_yen": remaining_budget_yen if "remaining_budget_yen" in locals() else get_int_env("BET_DAILY_BUDGET_YEN", 10000),
        "max_seconds_to_close": max_seconds_to_close,
        "strategy_version": strategy_version,
        "snapshot_mode": snapshot_mode,
        "selected_bets": int(len(bets)),
        "shadow_selected_bets": int(len(shadow_bets)),
        "profit_gate": profit_gate,
        "bet_filter": {
            "configs": DEFAULT_BET_CONFIGS,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pred[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
