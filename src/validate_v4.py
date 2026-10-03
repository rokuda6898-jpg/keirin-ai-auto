import json
from pathlib import Path

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
    ensure_dirs,
    normalize_race_prob,
    prepare_features,
)
from predict import apply_nexus_race_reading, apply_validated_top1_consensus


MIN_TRAIN_RACES = 8000
SELECTION_RACES = 7000
MIN_FINAL_VALIDATION_RACES = 30000
FINAL_FOLDS = 5


def split_date_at_race_count(df, target_count):
    counts = (
        df[["date", "race_id"]]
        .drop_duplicates()
        .groupby("date")["race_id"]
        .nunique()
        .sort_index()
    )
    cumulative = counts.cumsum()
    eligible = cumulative[cumulative >= int(target_count)]
    if eligible.empty:
        return None
    return eligible.index[0]


def topk_position_coverage(frame, score_col, finish_position, k):
    work = frame[["race_id", "finish_pos", score_col]].copy()
    work["_rank"] = work.groupby("race_id")[score_col].rank(ascending=False, method="first")
    actual = work[pd.to_numeric(work["finish_pos"], errors="coerce").eq(finish_position)]
    if actual.empty:
        return None
    return float(actual["_rank"].le(int(k)).mean())


def ordered_ticket_probability(a, b, c, second_col, third_col):
    p1 = float(a["p_win"])
    a2 = float(a[second_col])
    b2 = float(b[second_col])
    a3 = float(a[third_col])
    b3 = float(b[third_col])
    c3 = float(c[third_col])
    p2 = b2 / max(1.0 - a2, 1e-9)
    p3 = c3 / max(1.0 - a3 - b3, 1e-9)
    return max(0.0, min(1.0, p1 * p2 * p3))


def top10_tickets(race, second_col, third_col, adaptive):
    win_ranked = race.sort_values("p_win", ascending=False, kind="mergesort")
    if len(win_ranked) < 3:
        return []

    if adaptive:
        probs = pd.to_numeric(win_ranked["p_win"], errors="coerce").fillna(0.0).tolist()
        margin = float(probs[0] - probs[1]) if len(probs) > 1 else 1.0
        head_count = 1 if margin >= 0.40 else (2 if margin >= 0.03 else 3)
        head_cars = set(pd.to_numeric(win_ranked.head(head_count)["car_no"], errors="coerce").dropna().astype(int))
        second_ranked = race.sort_values(second_col, ascending=False, kind="mergesort")
        third_ranked = race.sort_values(third_col, ascending=False, kind="mergesort")
        second_cars = set(pd.to_numeric(second_ranked.head(min(4, len(race)))["car_no"], errors="coerce").dropna().astype(int))
        third_cars = set(pd.to_numeric(third_ranked.head(min(6, len(race)))["car_no"], errors="coerce").dropna().astype(int))
    else:
        cars = set(pd.to_numeric(race["car_no"], errors="coerce").dropna().astype(int))
        head_cars = second_cars = third_cars = cars

    rows = race[["car_no", "p_win", second_col, third_col]].to_dict("records")
    candidates = []
    for a in rows:
        ca = int(a["car_no"])
        if ca not in head_cars:
            continue
        for b in rows:
            cb = int(b["car_no"])
            if cb == ca or cb not in second_cars:
                continue
            for c in rows:
                cc = int(c["car_no"])
                if cc in {ca, cb} or cc not in third_cars:
                    continue
                prob = ordered_ticket_probability(a, b, c, second_col, third_col)
                candidates.append((prob, f"{ca}-{cb}-{cc}"))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [buy for _, buy in candidates[:10]]


def actual_trifecta(race):
    ordered = race[pd.to_numeric(race["finish_pos"], errors="coerce").isin([1, 2, 3])].copy()
    ordered["finish_pos"] = pd.to_numeric(ordered["finish_pos"], errors="coerce")
    ordered = ordered.sort_values("finish_pos")
    if len(ordered) < 3:
        return None
    return "-".join(str(int(x)) for x in ordered.head(3)["car_no"])


def ticket_coverage(frame, second_col, third_col, adaptive):
    races = hits = 0
    for _, race in frame.groupby("race_id", sort=False):
        actual = actual_trifecta(race)
        if actual is None:
            continue
        races += 1
        hits += int(actual in top10_tickets(race, second_col, third_col, adaptive))
    return {
        "races": int(races),
        "hits": int(hits),
        "rate": float(hits / races) if races else None,
    }


def fit_win_model(train_df):
    X_train, fills = prepare_features(train_df)
    y = pd.to_numeric(train_df["finish_pos"], errors="coerce").eq(1).astype(int)
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42,
    )
    model.fit(X_train, y)
    return model, fills


def fit_position_models(train_df, variant):
    X_train, fills = prepare_features(train_df)
    finish = pd.to_numeric(train_df["finish_pos"], errors="coerce")
    if variant == "cumulative_place":
        y2 = finish.le(2).astype(int)
        y3 = finish.le(3).astype(int)
    else:
        y2 = finish.eq(2).astype(int)
        y3 = finish.eq(3).astype(int)

    second = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.045, max_leaf_nodes=31,
        l2_regularization=0.03, random_state=72,
    )
    third = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.045, max_leaf_nodes=31,
        l2_regularization=0.03, random_state=73,
    )
    second.fit(X_train, y2)
    third.fit(X_train, y3)
    return second, third, fills


def score_win_model(model, fills, test_df):
    X, _ = prepare_features(test_df, fills)
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        for col in cols:
            if col not in X.columns:
                X[col] = float(fills.get(col, 0.0))
        X = X.reindex(columns=cols)
    raw = model.predict_proba(X)[:, 1]

    core = test_df.copy()
    core["p_raw"] = np.clip(raw, 1e-9, 1.0)
    core = normalize_race_prob(core, "p_raw", "p_core")

    nexus = apply_nexus_race_reading(core.copy())
    nexus["p_nexus_only"] = nexus["p_win"]
    final = apply_validated_top1_consensus(nexus.copy())
    return final


def attach_position_scores(frame, train_df, variant):
    second, third, fills = fit_position_models(train_df, variant)
    X, _ = prepare_features(frame, fills)
    if hasattr(second, "feature_names_in_"):
        cols = list(second.feature_names_in_)
        for col in cols:
            if col not in X.columns:
                X[col] = float(fills.get(col, 0.0))
        X = X.reindex(columns=cols)

    out = frame.copy()
    out["p_second_raw"] = np.clip(second.predict_proba(X)[:, 1], 1e-9, 1.0)
    out["p_third_raw"] = np.clip(third.predict_proba(X)[:, 1], 1e-9, 1.0)
    out = normalize_race_prob(out, "p_second_raw", "p_second")
    out = normalize_race_prob(out, "p_third_raw", "p_third")
    return out


def top1_rate(frame, score_col):
    ranked = frame.copy()
    idx = ranked.groupby("race_id")[score_col].idxmax()
    top = ranked.loc[idx]
    actual = pd.to_numeric(top["finish_pos"], errors="coerce").eq(1)
    return {
        "races": int(len(top)),
        "hits": int(actual.sum()),
        "rate": float(actual.mean()) if len(top) else None,
    }


def prepare_history():
    df = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    required = {"race_id", "date", "car_no", "finish_pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "race_id", "car_no"], kind="mergesort")
    df = add_player_prior_features(df)
    df = add_player_elo_features(df)
    df = add_pair_history_features(df)
    return df


def select_position_variant(df, selection_start, validation_start):
    train = df[df["date"] < selection_start].copy()
    select = df[(df["date"] >= selection_start) & (df["date"] < validation_start)].copy()
    win_model, win_fills = fit_win_model(train)
    scored = score_win_model(win_model, win_fills, select)
    scored["p_second"] = scored["p_win"]
    scored["p_third"] = scored["p_win"]

    rows = []
    for variant in ["exact_position", "cumulative_place"]:
        candidate = attach_position_scores(scored, train, variant)
        coverage = ticket_coverage(candidate, "p_second", "p_third", adaptive=True)
        rows.append({
            "variant": variant,
            **coverage,
            "second_top4": topk_position_coverage(candidate, "p_second", 2, 4),
            "third_top6": topk_position_coverage(candidate, "p_third", 3, 6),
        })
    rows.sort(key=lambda row: (row["rate"] if row["rate"] is not None else -1.0), reverse=True)
    return rows[0]["variant"], rows


def date_folds(df, start_date, folds):
    dates = sorted(pd.Series(df.loc[df["date"] >= start_date, "date"].dropna().unique()).tolist())
    if not dates:
        return []
    boundaries = np.linspace(0, len(dates), int(folds) + 1, dtype=int)
    result = []
    for i in range(int(folds)):
        chunk = dates[boundaries[i]:boundaries[i + 1]]
        if not chunk:
            continue
        result.append((pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1])))
    return result


def main():
    ensure_dirs()
    df = prepare_history()
    total_races = int(df["race_id"].nunique())
    required_total = MIN_TRAIN_RACES + SELECTION_RACES + MIN_FINAL_VALIDATION_RACES
    if total_races < required_total:
        raise RuntimeError(
            f"need at least {required_total} races for strict v4 validation; found {total_races}"
        )

    selection_start = split_date_at_race_count(df, MIN_TRAIN_RACES)
    validation_start = split_date_at_race_count(df, MIN_TRAIN_RACES + SELECTION_RACES)
    if selection_start is None or validation_start is None:
        raise RuntimeError("could not create chronological validation splits")

    selected_variant, selection_results = select_position_variant(
        df, selection_start, validation_start
    )

    fold_rows = []
    all_scored = []
    for fold, (fold_start, fold_end) in enumerate(date_folds(df, validation_start, FINAL_FOLDS), start=1):
        train = df[df["date"] < fold_start].copy()
        test = df[(df["date"] >= fold_start) & (df["date"] <= fold_end)].copy()
        if test.empty:
            continue

        win_model, win_fills = fit_win_model(train)
        scored = score_win_model(win_model, win_fills, test)
        scored["p_second"] = scored["p_win"]
        scored["p_third"] = scored["p_win"]

        legacy = ticket_coverage(scored, "p_second", "p_third", adaptive=False)
        adaptive = ticket_coverage(scored, "p_second", "p_third", adaptive=True)

        position = attach_position_scores(scored, train, selected_variant)
        v4 = ticket_coverage(position, "p_second", "p_third", adaptive=True)

        fold_rows.append({
            "fold": fold,
            "train_races": int(train["race_id"].nunique()),
            "test_races": int(test["race_id"].nunique()),
            "test_start": str(fold_start.date()),
            "test_end": str(fold_end.date()),
            "top1_core_rate": top1_rate(position, "p_core")["rate"],
            "top1_nexus_rate": top1_rate(position, "p_nexus_only")["rate"],
            "top1_final_rate": top1_rate(position, "p_win")["rate"],
            "legacy_top10_rate": legacy["rate"],
            "adaptive_top10_rate": adaptive["rate"],
            "v4_top10_rate": v4["rate"],
            "v4_minus_adaptive": (
                float(v4["rate"] - adaptive["rate"])
                if v4["rate"] is not None and adaptive["rate"] is not None else None
            ),
        })
        keep = position[[
            "date", "race_id", "car_no", "finish_pos",
            "p_core", "p_nexus_only", "p_win", "p_second", "p_third",
        ]].copy()
        keep["fold"] = fold
        all_scored.append(keep)

    scored_all = pd.concat(all_scored, ignore_index=True)
    evaluated_races = int(scored_all["race_id"].nunique())
    if evaluated_races < MIN_FINAL_VALIDATION_RACES:
        raise RuntimeError(
            f"strict final validation requires {MIN_FINAL_VALIDATION_RACES} races; got {evaluated_races}"
        )

    baseline = scored_all.copy()
    baseline["p_second"] = baseline["p_win"]
    baseline["p_third"] = baseline["p_win"]

    core_top1 = top1_rate(scored_all, "p_core")
    nexus_top1 = top1_rate(scored_all, "p_nexus_only")
    final_top1 = top1_rate(scored_all, "p_win")
    legacy = ticket_coverage(baseline, "p_second", "p_third", adaptive=False)
    adaptive = ticket_coverage(baseline, "p_second", "p_third", adaptive=True)
    v4 = ticket_coverage(scored_all, "p_second", "p_third", adaptive=True)

    gains = [
        row["v4_minus_adaptive"]
        for row in fold_rows
        if row.get("v4_minus_adaptive") is not None
    ]
    overall_gain = (
        float(v4["rate"] - adaptive["rate"])
        if v4["rate"] is not None and adaptive["rate"] is not None else None
    )
    target_passed = bool(
        evaluated_races >= MIN_FINAL_VALIDATION_RACES
        and overall_gain is not None
        and overall_gain >= 0.005
        and gains
        and min(gains) >= -0.01
    )

    summary = {
        "validation_scheme": "8000 train + 7000 model-selection + >=30000 untouched chronological walk-forward",
        "total_history_races": total_races,
        "evaluated_races": evaluated_races,
        "selection_start": str(pd.Timestamp(selection_start).date()),
        "final_validation_start": str(pd.Timestamp(validation_start).date()),
        "selected_position_variant": selected_variant,
        "selection_results": selection_results,
        "top1_core": core_top1,
        "top1_nexus": nexus_top1,
        "top1_final": final_top1,
        "legacy_trifecta_top10": legacy,
        "adaptive_trifecta_top10": adaptive,
        "v4_trifecta_top10": v4,
        "v4_minus_adaptive": overall_gain,
        "folds": fold_rows,
        "target_passed": target_passed,
        "gate_rule": ">=30000 final races, v4 top10 +0.5pp overall, no fold worse than -1.0pp",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "v4_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "v4_validation_folds.csv", index=False)
    scored_all.to_csv(OUTPUT_DIR / "v4_validation_predictions.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
