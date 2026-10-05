import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (
    HISTORY_CSV,
    MODEL_DIR,
    OUTPUT_DIR,
    add_pair_history_features,
    add_player_elo_features,
    add_player_prior_features,
    normalize_race_prob,
    prepare_features,
)
from multi_bet_backtest import make_candidates as make_multi_bet_candidates
from predict import apply_nexus_race_reading, apply_validated_top1_consensus, filter_trifecta_candidates_by_confidence
from trifecta_reranker import build_candidate_features

SUMMARY_JSON = OUTPUT_DIR / "production_replay_summary.json"
FOLDS_CSV = OUTPUT_DIR / "production_replay_folds.csv"
RACES_CSV = OUTPUT_DIR / "production_replay_races.csv"
CANDIDATES_CSV = OUTPUT_DIR / "production_replay_trifecta_candidates.csv"


def _model(random_state, l2=0.02):
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=l2,
        random_state=random_state,
    )


def actual_trifecta(group):
    finish = pd.to_numeric(group.get("finish_pos"), errors="coerce")
    car = pd.to_numeric(group.get("car_no"), errors="coerce")
    work = pd.DataFrame({"finish": finish, "car": car}).dropna()
    work = work[work["finish"].isin([1, 2, 3])].sort_values("finish")
    if len(work) != 3:
        return ""
    if work["finish"].astype(int).tolist() != [1, 2, 3]:
        return ""
    return "-".join(str(int(x)) for x in work["car"].tolist())


def enrich_history(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["date", "race_id", "car_no"], kind="mergesort")
    out = add_player_prior_features(out)
    out = add_player_elo_features(out)
    out = add_pair_history_features(out)
    out["target_win"] = pd.to_numeric(out["finish_pos"], errors="coerce").eq(1).astype(int)
    return out


def fit_fold_models(train_df, position_variant):
    X_train, fills = prepare_features(train_df)
    win = _model(42, 0.02)
    win.fit(X_train, train_df["target_win"].astype(int))

    finish = pd.to_numeric(train_df["finish_pos"], errors="coerce")
    if position_variant == "exact_position":
        y2 = finish.eq(2).astype(int)
        y3 = finish.eq(3).astype(int)
    else:
        y2 = finish.le(2).astype(int)
        y3 = finish.le(3).astype(int)

    second = _model(72, 0.03)
    third = _model(73, 0.03)
    second.fit(X_train, y2)
    third.fit(X_train, y3)
    return win, second, third, fills


def score_fold(test_df, win, second, third, fills):
    X, _ = prepare_features(test_df, fills)

    pred = test_df.copy()
    pred["p_raw"] = np.clip(win.predict_proba(X)[:, 1], 1e-9, 1.0)
    pred = normalize_race_prob(pred, "p_raw", "p_core")

    core_idx = pred.groupby("race_id")["p_core"].idxmax()
    core_pick = pred.loc[core_idx, ["race_id", "car_no"]].copy()
    core_pick["race_id"] = core_pick["race_id"].astype(str)
    core_map = dict(zip(core_pick["race_id"], pd.to_numeric(core_pick["car_no"], errors="coerce")))

    pred = apply_nexus_race_reading(pred)
    pred = apply_validated_top1_consensus(pred)

    p2raw = np.clip(second.predict_proba(X)[:, 1], 1e-9, 1.0)
    p3raw = np.clip(third.predict_proba(X)[:, 1], 1e-9, 1.0)
    pred["p_second_raw"] = p2raw
    pred["p_third_raw"] = p3raw
    pred = normalize_race_prob(pred, "p_second_raw", "p_second")
    pred = normalize_race_prob(pred, "p_third_raw", "p_third")
    pred["position_model_source"] = "walk_forward_position_specialists"
    pred["rank_in_race"] = pred.groupby("race_id")["p_win"].rank(
        ascending=False, method="first"
    ).astype(int)

    race_rows = []
    candidate_rows = []
    for race_id, g in pred.groupby("race_id", sort=False):
        rid = str(race_id)
        actual = actual_trifecta(g)
        if not actual:
            continue
        actual_winner = int(actual.split("-")[0])
        production_top = g.sort_values("p_win", ascending=False, kind="mergesort").iloc[0]
        production_pick = int(production_top["car_no"])
        core_pick_car = int(core_map.get(rid)) if pd.notna(core_map.get(rid)) else None

        candidates = make_multi_bet_candidates(g, top_k=min(len(g), 9))
        candidates = filter_trifecta_candidates_by_confidence(candidates, g)
        tri = candidates[candidates["bet_type"].eq("trifecta")].copy()
        tri["prob"] = pd.to_numeric(tri["prob"], errors="coerce")
        tri = tri.sort_values("prob", ascending=False, kind="mergesort")
        top10 = tri.head(10)
        hit10 = bool(top10["buy"].astype(str).eq(actual).any())

        probs = pd.to_numeric(
            g.sort_values("p_win", ascending=False, kind="mergesort")["p_win"],
            errors="coerce",
        ).dropna()
        margin = float(probs.iloc[0] - probs.iloc[1]) if len(probs) >= 2 else 1.0

        race_rows.append({
            "date": str(pd.Timestamp(g["date"].iloc[0]).date()),
            "race_id": rid,
            "venue": g.iloc[0].get("venue", ""),
            "race_no": g.iloc[0].get("race_no", ""),
            "actual_trifecta": actual,
            "actual_winner_car_no": actual_winner,
            "core_pick_car_no": core_pick_car,
            "production_pick_car_no": production_pick,
            "core_hit": bool(core_pick_car == actual_winner),
            "production_hit": bool(production_pick == actual_winner),
            "top1_top2_margin": margin,
            "trifecta_top10_hit": hit10,
            "trifecta_ticket_count": int(len(top10)),
        })

        if len(tri):
            train_pool = tri.head(80).copy()
            feat = build_candidate_features(g, train_pool)
            if not feat.empty:
                feat["date"] = str(pd.Timestamp(g["date"].iloc[0]).date())
                feat["race_id"] = rid
                feat["actual_trifecta"] = actual
                feat["target"] = feat["buy"].astype(str).eq(actual).astype(int)
                candidate_rows.append(feat)

    return pred, pd.DataFrame(race_rows), (
        pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    )


def _position_variant():
    path = MODEL_DIR / "v4_validation_summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = str(payload.get("selected_position_variant") or "")
        if value in {"cumulative_place", "exact_position"}:
            return value
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return "cumulative_place"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-races", type=int, default=12000)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-train-races", type=int, default=15000)
    args = parser.parse_args()

    df = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    required = {"race_id", "date", "finish_pos", "car_no"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history.csv missing columns: {sorted(missing)}")

    df = enrich_history(df)
    race_dates = (
        df.groupby("race_id", as_index=False)["date"].min()
        .sort_values(["date", "race_id"], kind="mergesort")
    )
    eval_races = min(max(int(args.eval_races), 1000), len(race_dates))
    eval_ids = set(race_dates.tail(eval_races)["race_id"].astype(str))
    eval_dates = sorted(df[df["race_id"].astype(str).isin(eval_ids)]["date"].dropna().unique())
    date_chunks = [list(x) for x in np.array_split(eval_dates, max(int(args.folds), 1)) if len(x)]

    all_races = []
    all_candidates = []
    fold_rows = []
    position_variant = _position_variant()

    for fold_no, chunk in enumerate(date_chunks, start=1):
        start = pd.Timestamp(chunk[0])
        end = pd.Timestamp(chunk[-1])
        train_df = df[df["date"] < start].copy()
        test_df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        train_races = int(train_df["race_id"].nunique())
        test_races = int(test_df["race_id"].nunique())
        if train_races < int(args.min_train_races) or test_races == 0:
            print(f"skip fold {fold_no}: train={train_races} test={test_races}")
            continue

        print(
            f"production replay fold={fold_no} train={train_races} test={test_races} "
            f"start={start.date()} end={end.date()}"
        )
        win, second, third, fills = fit_fold_models(train_df, position_variant)
        _, races, candidates = score_fold(test_df, win, second, third, fills)
        if races.empty:
            continue
        races["fold"] = fold_no
        if len(candidates):
            candidates["fold"] = fold_no
            all_candidates.append(candidates)
        all_races.append(races)

        fold_rows.append({
            "fold": fold_no,
            "train_races": train_races,
            "test_races": int(len(races)),
            "test_start": str(start.date()),
            "test_end": str(end.date()),
            "core_top1_hit_rate": float(races["core_hit"].mean()),
            "production_top1_hit_rate": float(races["production_hit"].mean()),
            "trifecta_top10_hit_rate": float(races["trifecta_top10_hit"].mean()),
            "candidate_rows": int(len(candidates)),
        })

    if not all_races:
        raise ValueError("no valid production replay folds")

    races = pd.concat(all_races, ignore_index=True)
    candidates = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates else pd.DataFrame()
    )
    folds = pd.DataFrame(fold_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    races.to_csv(RACES_CSV, index=False)
    folds.to_csv(FOLDS_CSV, index=False)
    candidates.to_csv(CANDIDATES_CSV, index=False)

    summary = {
        "updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "scheme": "expanding chronological replay with fold-local win/second/third training and current production post-processing",
        "position_variant": position_variant,
        "evaluated_races": int(len(races)),
        "core_top1_hits": int(races["core_hit"].sum()),
        "core_top1_hit_rate": float(races["core_hit"].mean()),
        "production_top1_hits": int(races["production_hit"].sum()),
        "production_top1_hit_rate": float(races["production_hit"].mean()),
        "trifecta_top10_hits": int(races["trifecta_top10_hit"].sum()),
        "trifecta_top10_hit_rate": float(races["trifecta_top10_hit"].mean()),
        "trifecta_training_candidate_rows": int(len(candidates)),
        "folds": fold_rows,
        "production_auto_change": False,
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
