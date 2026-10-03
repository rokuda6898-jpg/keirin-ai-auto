import json
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from common import (
    ensure_dirs,
    HISTORY_CSV,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_COLS,
    OUTPUT_DIR,
    add_player_prior_features,
    add_player_elo_features,
    add_pair_history_features,
    prepare_features,
    normalize_race_prob,
)


def ensure_history():
    if not HISTORY_CSV.exists():
        print("history.csv not found; generating sample data")
        subprocess.check_call([sys.executable, "src/make_sample_data.py", "--if-missing"])


def safe_auc(y, p):
    try:
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def get_stake_yen():
    raw = os.getenv("BET_STAKE_YEN", "100").strip()
    try:
        stake = int(raw)
    except ValueError:
        stake = 100
    return max(stake, 0)


def summarize_strategy(name, bets):
    if len(bets) == 0:
        return {
            "strategy": name,
            "bets": 0,
            "hits": 0,
            "hit_rate": None,
            "stake_yen": 0,
            "win_return_yen": 0,
            "loss_amount_yen": 0,
            "profit_yen": 0,
            "roi": None,
        }

    total_stake = float(bets["stake_yen"].sum())
    total_return = float(bets["actual_return_yen"].sum())
    loss_amount = float(bets.loc[~bets["is_hit"], "stake_yen"].sum())
    profit = float(bets["actual_profit_yen"].sum())

    return {
        "strategy": name,
        "bets": int(len(bets)),
        "hits": int(bets["is_hit"].sum()),
        "hit_rate": float(bets["is_hit"].mean()),
        "stake_yen": int(total_stake),
        "win_return_yen": int(total_return),
        "loss_amount_yen": int(loss_amount),
        "profit_yen": int(profit),
        "roi": float(profit / total_stake) if total_stake else None,
    }


def write_backtest_outputs(test_df, test_prob, stake_yen):
    backtest = test_df.copy()
    backtest["p_raw"] = test_prob
    backtest = normalize_race_prob(backtest)
    backtest["odds_win"] = pd.to_numeric(backtest.get("odds_win", np.nan), errors="coerce")
    backtest["expected_value_win"] = backtest["p_win"] * backtest["odds_win"] - 1
    backtest["rank_in_race"] = backtest.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
    backtest["stake_yen"] = stake_yen
    backtest["is_hit"] = pd.to_numeric(backtest["finish_pos"], errors="coerce").eq(1)
    backtest["actual_return_yen"] = np.where(backtest["is_hit"], stake_yen * backtest["odds_win"], 0).round(0)
    backtest["actual_profit_yen"] = backtest["actual_return_yen"] - stake_yen
    backtest["loss_amount_yen"] = np.where(backtest["is_hit"], 0, stake_yen)
    backtest["expected_profit_yen"] = (stake_yen * backtest["expected_value_win"]).round(0)

    valid_odds = pd.to_numeric(backtest["odds_win"], errors="coerce").gt(0)
    top1_bets = backtest[(backtest["rank_in_race"] == 1) & valid_odds].copy()
    value_bets = backtest[(backtest["expected_value_win"] > 0) & valid_odds].copy()
    summaries = [
        summarize_strategy("top_p_win_each_race", top1_bets),
        summarize_strategy("positive_expected_value_all", value_bets),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backtest_cols = [
        "date", "venue", "race_no", "race_id", "rank_in_race", "car_no",
        "player_id", "finish_pos", "odds_win", "p_win", "expected_value_win",
        "stake_yen", "is_hit", "actual_return_yen", "actual_profit_yen",
        "loss_amount_yen", "expected_profit_yen",
    ]
    for c in backtest_cols:
        if c not in backtest.columns:
            backtest[c] = ""

    backtest.sort_values(["date", "venue", "race_no", "rank_in_race"])[backtest_cols].to_csv(
        OUTPUT_DIR / "latest_backtest.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)
    with open(OUTPUT_DIR / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    return summaries


def evaluate_feature_ablation(df, train_df, calib_df, test_df, feature_drop, name):
    """Time-ordered shadow experiment; never changes the production model."""
    from common import FEATURE_COLS
    keep = [x for x in FEATURE_COLS if x not in set(feature_drop)]
    def prep(frame, fills=None):
        from common import add_categorical_codes, add_strength_features
        x = add_strength_features(add_categorical_codes(frame))[keep].copy()
        for col in keep:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        if fills is None:
            fills = {col: float(x[col].median()) if pd.notna(x[col].median()) else 0.0 for col in keep}
        return x.fillna(fills), fills

    Xtr, fills = prep(train_df)
    Xcal, _ = prep(calib_df, fills)
    Xte, _ = prep(test_df, fills)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.02, random_state=42)
    m.fit(Xtr, train_df["target_win"].astype(int))
    raw_cal = m.predict_proba(Xcal)[:,1] if len(Xcal) else np.array([])
    raw_test = m.predict_proba(Xte)[:,1]
    cal = None
    if len(raw_cal) > 20 and len(set(calib_df["target_win"].astype(int))) == 2:
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_cal, calib_df["target_win"].astype(int))
    prob = cal.predict(raw_test) if cal is not None else raw_test
    tmp = test_df[["race_id","finish_pos"]].copy()
    tmp["p_raw"] = prob
    tmp = normalize_race_prob(tmp)
    tmp["rank"] = tmp.groupby("race_id")["p_win"].rank(ascending=False, method="first")
    top = tmp[tmp["rank"].eq(1)]
    winners = tmp[pd.to_numeric(tmp["finish_pos"], errors="coerce").eq(1)]["p_win"].clip(1e-9,1)
    return {
        "variant": name,
        "dropped_features": feature_drop,
        "top1_hit_rate": float(pd.to_numeric(top["finish_pos"], errors="coerce").eq(1).mean()) if len(top) else None,
        "race_logloss": float(-np.log(winners).mean()) if len(winners) else None,
        "brier": float(brier_score_loss(test_df["target_win"].astype(int), prob)) if len(test_df) else None,
        "features": len(keep),
    }


def evaluate_race_ranker(train_df, test_df):
    """Shadow race-ranking model.

    Learns a graded within-race relevance target (winner highest) and is scored
    only by which rider it ranks first. It never replaces the production model.
    """
    Xtr, fills = prepare_features(train_df)
    Xte, _ = prepare_features(test_df, fills)
    finish_train = pd.to_numeric(train_df["finish_pos"], errors="coerce")
    field_train = pd.to_numeric(train_df.get("entries_number"), errors="coerce")
    if field_train.isna().all():
        field_train = train_df.groupby("race_id")["race_id"].transform("count")
    relevance = ((field_train + 1 - finish_train).clip(lower=0) / field_train.clip(lower=1)).fillna(0.0)
    ranker = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=43,
    )
    ranker.fit(Xtr, relevance)
    score = ranker.predict(Xte)
    out = test_df[["race_id","finish_pos"]].copy()
    out["rank_score"] = score
    out["rank"] = out.groupby("race_id")["rank_score"].rank(ascending=False, method="first")
    top = out[out["rank"].eq(1)]
    hit = float(pd.to_numeric(top["finish_pos"], errors="coerce").eq(1).mean()) if len(top) else None
    return ranker, fills, {
        "variant": "shadow_race_ranker",
        "races": int(out["race_id"].nunique()),
        "top1_hit_rate": hit,
    }


def evaluate_walk_forward(df, folds=4):
    """Expanding-window Top1 validation across several future periods."""
    dates = sorted(pd.Series(df["date"].dropna().unique()).tolist())
    if len(dates) < 20:
        return []
    start = max(int(len(dates) * 0.50), 5)
    remaining = len(dates) - start
    step = max(remaining // folds, 1)
    rows = []
    for fold in range(folds):
        test_start_i = start + fold * step
        test_end_i = len(dates) if fold == folds - 1 else min(start + (fold + 1) * step, len(dates))
        if test_start_i >= len(dates) or test_end_i <= test_start_i:
            continue
        train_end = dates[test_start_i]
        test_end = dates[test_end_i - 1]
        tr = df[df["date"] < train_end].copy()
        te = df[(df["date"] >= train_end) & (df["date"] <= test_end)].copy()
        if len(tr) < 100 or te["race_id"].nunique() < 5:
            continue
        Xtr, fills = prepare_features(tr)
        Xte, _ = prepare_features(te, fills)
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.045, max_leaf_nodes=31,
            l2_regularization=0.02, random_state=100 + fold,
        )
        m.fit(Xtr, tr["target_win"].astype(int))
        prob = m.predict_proba(Xte)[:, 1]
        scored = te[["race_id","finish_pos"]].copy()
        scored["p_raw"] = prob
        scored = normalize_race_prob(scored)
        scored["rank"] = scored.groupby("race_id")["p_win"].rank(ascending=False, method="first")
        top = scored[scored["rank"].eq(1)]
        winners = scored[pd.to_numeric(scored["finish_pos"], errors="coerce").eq(1)]["p_win"].clip(1e-9,1)
        rows.append({
            "fold": fold + 1,
            "train_through": str(pd.Timestamp(train_end).date()),
            "test_through": str(pd.Timestamp(test_end).date()),
            "train_races": int(tr["race_id"].nunique()),
            "test_races": int(te["race_id"].nunique()),
            "top1_hit_rate": float(pd.to_numeric(top["finish_pos"], errors="coerce").eq(1).mean()) if len(top) else None,
            "race_logloss": float(-np.log(winners).mean()) if len(winners) else None,
        })
    return rows


def _topk_position_coverage(frame, score_col, finish_position, k):
    work = frame[["race_id", "finish_pos", score_col]].copy()
    work["_rank"] = work.groupby("race_id")[score_col].rank(ascending=False, method="first")
    actual = work[pd.to_numeric(work["finish_pos"], errors="coerce").eq(finish_position)]
    if len(actual) == 0:
        return None
    return float(actual["_rank"].le(k).mean())


def _adaptive_trifecta_top10_coverage(frame, second_col, third_col):
    """Race-level coverage of the actual trifecta inside the model's top 10 tickets."""
    hits = 0
    races = 0
    for _, race in frame.groupby("race_id", sort=False):
        actual = race[pd.to_numeric(race["finish_pos"], errors="coerce").isin([1, 2, 3])].copy()
        if len(actual) < 3:
            continue
        actual = actual.sort_values("finish_pos")
        actual_buy = "-".join(str(int(x)) for x in actual["car_no"].tolist()[:3])

        win_ranked = race.sort_values("p_win", ascending=False, kind="mergesort")
        probs = pd.to_numeric(win_ranked["p_win"], errors="coerce").fillna(0).tolist()
        if len(probs) < 3:
            continue
        margin = float(probs[0] - probs[1])
        head_count = 1 if margin >= 0.40 else (2 if margin >= 0.03 else 3)

        heads = set(pd.to_numeric(win_ranked.head(head_count)["car_no"], errors="coerce").dropna().astype(int))
        second_ranked = race.sort_values(second_col, ascending=False, kind="mergesort")
        third_ranked = race.sort_values(third_col, ascending=False, kind="mergesort")
        seconds = set(pd.to_numeric(second_ranked.head(min(4, len(race)))["car_no"], errors="coerce").dropna().astype(int))
        thirds = set(pd.to_numeric(third_ranked.head(min(6, len(race)))["car_no"], errors="coerce").dropna().astype(int))

        candidates = []
        rows = race[["car_no", "p_win", second_col, third_col]].to_dict("records")
        for a in rows:
            for b in rows:
                for d in rows:
                    ca, cb, cd = int(a["car_no"]), int(b["car_no"]), int(d["car_no"])
                    if len({ca, cb, cd}) < 3:
                        continue
                    if ca not in heads or cb not in seconds or cd not in thirds:
                        continue
                    p1 = float(a["p_win"])
                    a2 = float(a[second_col]); b2 = float(b[second_col])
                    a3 = float(a[third_col]); b3 = float(b[third_col]); d3 = float(d[third_col])
                    p2 = b2 / max(1.0 - a2, 1e-9)
                    p3 = d3 / max(1.0 - a3 - b3, 1e-9)
                    prob = max(0.0, min(1.0, p1 * p2 * p3))
                    candidates.append((prob, f"{ca}-{cb}-{cd}"))

        top10 = sorted(candidates, key=lambda x: x[0], reverse=True)[:10]
        races += 1
        hits += int(any(buy == actual_buy for _, buy in top10))
    return (float(hits / races) if races else None, int(hits), int(races))


def evaluate_position_models(train_df, test_df, X_train, X_test, baseline_pred):
    """Train held-out second/third-place specialists and compare to p_win ranking."""
    models = {}
    scored = test_df[["race_id", "finish_pos"]].copy()
    for position, name in [(2, "second"), (3, "third")]:
        y = pd.to_numeric(train_df["finish_pos"], errors="coerce").eq(position).astype(int)
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            random_state=50 + position,
        )
        model.fit(X_train, y)
        raw = model.predict_proba(X_test)[:, 1]
        scored[f"p_{name}_raw"] = raw
        scored = normalize_race_prob(scored, f"p_{name}_raw", f"p_{name}")
        models[name] = model

    base = test_df[["race_id", "car_no", "finish_pos"]].copy()
    base["p_win"] = pd.to_numeric(baseline_pred["p_win"], errors="coerce").values
    base["p_second"] = base["p_win"]
    base["p_third"] = base["p_win"]

    candidate = test_df[["race_id", "car_no", "finish_pos"]].copy()
    candidate["p_win"] = pd.to_numeric(baseline_pred["p_win"], errors="coerce").values
    candidate["p_second"] = pd.to_numeric(scored["p_second"], errors="coerce").values
    candidate["p_third"] = pd.to_numeric(scored["p_third"], errors="coerce").values

    baseline_tri_rate, baseline_tri_hits, tri_races = _adaptive_trifecta_top10_coverage(
        base, "p_second", "p_third"
    )
    candidate_tri_rate, candidate_tri_hits, _ = _adaptive_trifecta_top10_coverage(
        candidate, "p_second", "p_third"
    )

    metrics = {
        "second_top4_baseline": _topk_position_coverage(base, "p_win", 2, 4),
        "second_top4_candidate": _topk_position_coverage(candidate, "p_second", 2, 4),
        "third_top6_baseline": _topk_position_coverage(base, "p_win", 3, 6),
        "third_top6_candidate": _topk_position_coverage(candidate, "p_third", 3, 6),
        "trifecta_top10_baseline": baseline_tri_rate,
        "trifecta_top10_candidate": candidate_tri_rate,
        "trifecta_top10_baseline_hits": baseline_tri_hits,
        "trifecta_top10_candidate_hits": candidate_tri_hits,
        "trifecta_top10_races": tri_races,
    }
    gains = []
    for cand_key, base_key in [
        ("second_top4_candidate", "second_top4_baseline"),
        ("third_top6_candidate", "third_top6_baseline"),
    ]:
        cand = metrics.get(cand_key)
        base_value = metrics.get(base_key)
        if cand is not None and base_value is not None:
            gains.append(cand - base_value)

    metrics["average_coverage_gain"] = float(np.mean(gains)) if gains else None
    tri_gain = None
    if candidate_tri_rate is not None and baseline_tri_rate is not None:
        tri_gain = float(candidate_tri_rate - baseline_tri_rate)
    metrics["trifecta_top10_gain"] = tri_gain
    metrics["target_passed"] = bool(
        gains
        and all(gain >= -1e-9 for gain in gains)
        and float(np.mean(gains)) >= 0.005
        and tri_gain is not None
        and tri_gain >= 0.005
    )
    metrics["gate_reason"] = (
        "position specialists improved held-out placement and trifecta top10 coverage"
        if metrics["target_passed"]
        else "position specialists failed the held-out placement/trifecta coverage gate"
    )
    return models, metrics


def main():
    ensure_dirs()
    ensure_history()
    stake_yen = get_stake_yen()

    df = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    required = {"race_id", "date", "finish_pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history.csv missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["target_win"] = (pd.to_numeric(df["finish_pos"], errors="coerce") == 1).astype(int)
    df = df.sort_values(["date", "race_id", "car_no"])
    # Compute all stateful pre-race history features once on the full
    # chronological stream, before any time split. Each helper is leakage-safe:
    # the current race result is applied only after its pre-race features exist.
    df = add_player_prior_features(df)
    df = add_player_elo_features(df)
    df = add_pair_history_features(df)

    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 10:
        raise ValueError("not enough dates in history.csv")

    d1 = dates[int(len(dates) * 0.70)]
    d2 = dates[int(len(dates) * 0.85)]

    train_df = df[df["date"] < d1].copy()
    calib_df = df[(df["date"] >= d1) & (df["date"] < d2)].copy()
    test_df = df[df["date"] >= d2].copy()

    if len(train_df) < 100:
        raise ValueError("not enough training rows")

    X_train, fill_values = prepare_features(train_df)
    y_train = train_df["target_win"].astype(int)

    X_calib, _ = prepare_features(calib_df, fill_values)
    y_calib = calib_df["target_win"].astype(int)

    X_test, _ = prepare_features(test_df, fill_values)
    y_test = test_df["target_win"].astype(int)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )

    model.fit(X_train, y_train)

    calibrator = None
    calib_raw = model.predict_proba(X_calib)[:, 1] if len(calib_df) else np.array([])
    if len(calib_raw) > 20 and len(set(y_calib)) == 2:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calib_raw, y_calib)

    test_raw = model.predict_proba(X_test)[:, 1]
    test_prob = calibrator.predict(test_raw) if calibrator is not None else test_raw

    pred_df = test_df[["race_id", "finish_pos"]].copy()
    pred_df["p_raw"] = test_prob
    pred_df = normalize_race_prob(pred_df)
    pred_df["rank_in_race"] = pred_df.groupby("race_id")["p_win"].rank(ascending=False, method="first")

    position_eval_models, position_metrics = evaluate_position_models(
        train_df, test_df, X_train, X_test, pred_df
    )

    winner_probs = pred_df[pred_df["finish_pos"] == 1]["p_win"].clip(1e-9, 1.0)
    race_logloss = float(-np.log(winner_probs).mean()) if len(winner_probs) else None
    top1 = pred_df[pred_df["rank_in_race"].eq(1)]
    top1_hit_rate = float(pd.to_numeric(top1["finish_pos"], errors="coerce").eq(1).mean()) if len(top1) else None
    backtest_summaries = write_backtest_outputs(test_df, test_prob, stake_yen)
    ranker_model, ranker_fill_values, ranker_metrics = evaluate_race_ranker(train_df, test_df)
    joblib.dump(
        {"model": ranker_model, "fill_values": ranker_fill_values, "features": FEATURE_COLS, "metrics": ranker_metrics},
        MODEL_PATH.parent / "shadow_rank_model.joblib",
    )
    (OUTPUT_DIR / "shadow_rank_metrics.json").write_text(
        json.dumps(ranker_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ablations = [
        evaluate_feature_ablation(df, train_df, calib_df, test_df, [], "all_features_calibrated"),
        evaluate_feature_ablation(df, train_df, calib_df, test_df, ["player_id_code"], "no_player_id"),
        evaluate_feature_ablation(df, train_df, calib_df, test_df, ["car_no","bracket_no"], "no_car_bracket"),
        evaluate_feature_ablation(df, train_df, calib_df, test_df, ["player_id_code","car_no","bracket_no"], "no_identity_or_gate"),
    ]
    (OUTPUT_DIR / "model_ablation.json").write_text(json.dumps(ablations, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(ablations).drop(columns=["dropped_features"], errors="ignore").to_csv(OUTPUT_DIR / "model_ablation.csv", index=False)

    walk_forward = evaluate_walk_forward(df, folds=4)
    pd.DataFrame(walk_forward).to_csv(OUTPUT_DIR / "walk_forward_top1.csv", index=False)
    (OUTPUT_DIR / "walk_forward_top1.json").write_text(
        json.dumps(walk_forward, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = {
        "trained_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "model": "HistGradientBoostingClassifier + race probability normalization",
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_calib": int(len(calib_df)),
        "n_test": int(len(test_df)),
        "n_races": int(df["race_id"].nunique()),
        "production_n_train": int(len(df)),
        "features": FEATURE_COLS,
        "brier_score_binary": float(brier_score_loss(y_test, test_prob)) if len(test_df) else None,
        "auc_binary": safe_auc(y_test, test_prob),
        "race_logloss": race_logloss,
        "top1_hit_rate": top1_hit_rate,
        "stake_yen": stake_yen,
        "backtest": backtest_summaries,
        "feature_ablation": ablations,
        "shadow_race_ranker": ranker_metrics,
        "walk_forward": walk_forward,
        "position_models": position_metrics,
    }

    X_full, production_fill_values = prepare_features(df)
    production_model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    production_model.fit(X_full, df["target_win"].astype(int))

    bundle = {
        "model": production_model,
        "calibrator": None,
        "fill_values": production_fill_values,
        "features": FEATURE_COLS,
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_PATH)

    position_model_path = MODEL_PATH.parent / "position_models.joblib"
    position_metrics_path = MODEL_PATH.parent / "position_metrics.json"
    with open(position_metrics_path, "w", encoding="utf-8") as f:
        json.dump(position_metrics, f, ensure_ascii=False, indent=2)

    if position_metrics.get("target_passed"):
        second_model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.045, max_leaf_nodes=31,
            l2_regularization=0.03, random_state=52,
        )
        third_model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.045, max_leaf_nodes=31,
            l2_regularization=0.03, random_state=53,
        )
        second_model.fit(X_full, pd.to_numeric(df["finish_pos"], errors="coerce").eq(2).astype(int))
        third_model.fit(X_full, pd.to_numeric(df["finish_pos"], errors="coerce").eq(3).astype(int))
        joblib.dump(
            {
                "second_model": second_model,
                "third_model": third_model,
                "fill_values": production_fill_values,
                "features": FEATURE_COLS,
                "metrics": position_metrics,
            },
            position_model_path,
        )
        print(f"promoted position models: {position_model_path}")
    else:
        print("position model gate failed; keeping existing production position models")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
