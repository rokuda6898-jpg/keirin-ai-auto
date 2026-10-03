import json

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (
    FEATURE_COLS,
    HISTORY_CSV,
    MODEL_DIR,
    add_pair_history_features,
    add_player_elo_features,
    add_player_prior_features,
    prepare_features,
)


def main():
    summary_path = MODEL_DIR / "v4_validation_summary.json"
    if not summary_path.exists():
        raise RuntimeError("v4 validation summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("target_passed", False)) or int(summary.get("evaluated_races") or 0) < 30000:
        raise RuntimeError("strict 30000-race v4 validation did not pass")

    variant = summary.get("selected_position_variant")
    if variant not in {"exact_position", "cumulative_place"}:
        raise RuntimeError(f"invalid selected position variant: {variant}")

    df = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "race_id", "car_no"], kind="mergesort")
    df = add_player_prior_features(df)
    df = add_player_elo_features(df)
    df = add_pair_history_features(df)

    X, fills = prepare_features(df)
    finish = pd.to_numeric(df["finish_pos"], errors="coerce")
    if variant == "cumulative_place":
        y2 = finish.le(2).astype(int)
        y3 = finish.le(3).astype(int)
    else:
        y2 = finish.eq(2).astype(int)
        y3 = finish.eq(3).astype(int)

    second = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.045, max_leaf_nodes=31,
        l2_regularization=0.03, random_state=172,
    )
    third = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.045, max_leaf_nodes=31,
        l2_regularization=0.03, random_state=173,
    )
    second.fit(X, y2)
    third.fit(X, y3)

    bundle = {
        "second_model": second,
        "third_model": third,
        "fill_values": fills,
        "features": FEATURE_COLS,
        "variant": variant,
        "strict_validation": summary,
        "metrics": {
            "target_passed": True,
            "evaluated_races": int(summary["evaluated_races"]),
            "trifecta_top10_rate": summary.get("v4_trifecta_top10", {}).get("rate"),
            "baseline_adaptive_rate": summary.get("adaptive_trifecta_top10", {}).get("rate"),
            "gain": summary.get("v4_minus_adaptive"),
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "position_models.joblib")
    print(
        json.dumps(
            {
                "trained_position_models": True,
                "variant": variant,
                "history_races": int(df["race_id"].nunique()),
                "strict_validation_races": int(summary["evaluated_races"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
