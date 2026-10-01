from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

HISTORY_CSV = RAW_DIR / "history.csv"
TODAY_CSV = RAW_DIR / "today_entries.csv"
TODAY_ODDS_CSV = RAW_DIR / "today_odds.csv"
HISTORY_ODDS_CSV = RAW_DIR / "history_odds.csv"
HISTORY_TRIFECTA_ODDS_CSV = RAW_DIR / "history_trifecta_odds.csv"
MODEL_PATH = MODEL_DIR / "win_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_COLS = [
    "race_no",
    "car_no",
    "bracket_no",
    "age",
    "term",
    "region_id",
    "player_class",
    "player_group",
    "score",
    "rider_strength",
    "rider_strength_rank",
    "rider_strength_gap_to_best",
    "rider_strength_vs_field",
    "score_rank",
    "score_gap_to_best",
    "score_vs_field",
    "player_prior_races",
    "player_prior_win_rate",
    "player_prior_place2_rate",
    "player_prior_place3_rate",
    "player_prior_avg_finish",
    "player_prior_days_since_last_race",
    "player_prior_strength",
    "player_prior_strength_rank",
    "player_prior_strength_gap_to_best",
    "player_prior_strength_vs_field",
    "player_elo",
    "player_elo_rank",
    "player_elo_vs_field",
    "player_recent_weighted_finish",
    "player_recent5_avg_finish",
    "player_recent10_avg_finish",
    "player_recent5_win_rate",
    "player_recent10_win_rate",
    "player_form_trend_5_vs_10",
    "meeting_prior_races",
    "meeting_prior_avg_finish",
    "meeting_form_delta",
    "meeting_finish_trend",
    "h2h_prior_meetings",
    "h2h_prior_win_share",
    "line_pair_prior_races",
    "line_pair_second_win_rate",
    "win_rate",
    "win_rate_rank",
    "win_rate_gap_to_best",
    "place2_rate",
    "place2_rate_rank",
    "place3_rate",
    "place3_rate_rank",
    "back_count",
    "standing_count",
    "front_runner_count",
    "stalker_count",
    "deep_closer_count",
    "marker_count",
    "gear_ratio",
    "prediction_mark",
    "recent_avg_finish",
    "recent_avg_finish_rank",
    "recent_races_count",
    "current_cup_avg_finish",
    "days_since_last_race",
    "venue_win_rate",
    "track_win_rate",
    "track_place2_rate",
    "track_place3_rate",
    "track_races",
    "weather_win_rate",
    "weather_place2_rate",
    "weather_place3_rate",
    "weather_races",
    "race_type_win_rate",
    "race_type_place2_rate",
    "race_type_place3_rate",
    "race_type_races",
    "hour_win_rate",
    "hour_place2_rate",
    "hour_place3_rate",
    "hour_races",
    "line_role_win_rate",
    "line_role_place2_rate",
    "line_role_place3_rate",
    "line_role_races",
    "line_id",
    "line_position",
    "line_size",
    "is_line_leader",
    "number_of_lines",
    "line_leader_strength",
    "line_second_strength",
    "strength_vs_line_leader",
    "strength_vs_line_second",
    "line_strength_mean",
    "line_strength_max",
    "other_line_attack_pressure",
    "race_attack_pressure",
    "wind_speed",
    "meeting_day",
    "entries_number",
    "field_size_7",
    "field_size_9",
    "lines_per_field",
    "is_grade_race",
    "start_hour",
    "day_of_week",
    "month",
    "current_term_class",
    "current_term_group",
    "previous_term_class",
    "previous_term_group",
    "odds_win",
    "distance",
    "track_style_code",
    "class_change",
    "is_promoted",
    "is_demoted",
    "style_code",
    "venue_code",
    "race_class_code",
    "race_type_code",
    "weather_code",
    "line_type_code",
    "prefecture_code",
    "gender_code",
    "player_id_code",
]

STYLE_MAP = {
    "逃": 0,
    "捲": 1,
    "ま": 1,
    "差": 2,
    "追": 3,
    "両": 4,
    "自在": 4,
}


def ensure_dirs():
    for p in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, MODEL_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def add_style_code(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "style" not in df.columns:
        df["style"] = "追"
    df["style_code"] = df["style"].map(STYLE_MAP).fillna(-1).astype(float)
    return df


def stable_bucket(value, modulo=1000):
    if pd.isna(value):
        return -1.0
    text = str(value).strip()
    if not text:
        return -1.0
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) % modulo)


def add_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = add_style_code(df)
    df = df.copy()
    df["venue_code"] = df.get("venue", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 200))
    df["race_class_code"] = df.get("race_class", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 50))
    df["race_type_code"] = df.get("race_type", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 100))
    df["weather_code"] = df.get("weather", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 20))
    df["line_type_code"] = df.get("line_type", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 20))
    df["prefecture_code"] = df.get("prefecture", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 100))
    df["gender_code"] = df.get("gender", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 10))
    df["player_id_code"] = df.get("player_id", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 2000))

    # Interaction features known before the race.
    distance = pd.to_numeric(df.get("distance", pd.Series(index=df.index, dtype=float)), errors="coerce")
    track_bucket = pd.Series(np.where(distance < 375, 333, np.where(distance >= 450, 500, 400)), index=df.index)
    style_text = df.get("style", pd.Series("", index=df.index)).fillna("").astype(str)
    df["track_style_code"] = [
        stable_bucket(f"{int(tb)}:{style}", 100) if pd.notna(tb) else -1.0
        for tb, style in zip(track_bucket, style_text)
    ]
    current_class = pd.to_numeric(df.get("current_term_class", pd.Series(index=df.index, dtype=float)), errors="coerce")
    previous_class = pd.to_numeric(df.get("previous_term_class", pd.Series(index=df.index, dtype=float)), errors="coerce")
    df["class_change"] = current_class - previous_class
    df["is_promoted"] = df["class_change"].gt(0).astype(float)
    df["is_demoted"] = df["class_change"].lt(0).astype(float)

    dates = pd.to_datetime(df.get("date", pd.Series(index=df.index, dtype=object)), errors="coerce")
    df["day_of_week"] = dates.dt.dayofweek
    df["month"] = dates.dt.month
    start_at = pd.to_numeric(df.get("start_at", pd.Series(index=df.index, dtype=float)), errors="coerce")
    df["start_hour"] = ((start_at // 3600 + 9) % 24).where(start_at.notna())
    if "entries_number" not in df.columns and "race_id" in df.columns:
        df["entries_number"] = df.groupby("race_id")["race_id"].transform("count")
    entries = pd.to_numeric(df.get("entries_number"), errors="coerce")
    lines = pd.to_numeric(df.get("number_of_lines"), errors="coerce")
    df["field_size_7"] = entries.eq(7).astype(float)
    df["field_size_9"] = entries.eq(9).astype(float)
    df["lines_per_field"] = lines / entries.replace(0, np.nan)
    return df


def add_player_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"player_id", "date", "finish_pos"}
    if not required.issubset(df.columns):
        return df

    original_index = df.index
    work = df.copy()
    work["_original_order"] = np.arange(len(work))
    work["_date_dt"] = pd.to_datetime(work["date"], errors="coerce")
    sort_cols = [c for c in ["_date_dt", "race_id", "race_no", "car_no", "_original_order"] if c in work.columns]
    work = work.sort_values(sort_cols, kind="mergesort")

    player = work["player_id"].astype(str)
    finish = pd.to_numeric(work["finish_pos"], errors="coerce")
    observed = finish.notna().astype(float)
    prior_races = observed.groupby(player, dropna=False).cumsum() - observed

    win = finish.eq(1).astype(float) * observed
    place2 = finish.le(2).astype(float) * observed
    place3 = finish.le(3).astype(float) * observed
    finish_sum = finish.fillna(0) * observed

    prior_wins = win.groupby(player, dropna=False).cumsum() - win
    prior_place2 = place2.groupby(player, dropna=False).cumsum() - place2
    prior_place3 = place3.groupby(player, dropna=False).cumsum() - place3
    prior_finish_sum = finish_sum.groupby(player, dropna=False).cumsum() - finish_sum

    def safe_rate(values):
        return np.where(prior_races > 0, values / prior_races, np.nan)

    work["player_prior_races"] = prior_races
    work["player_prior_win_rate"] = safe_rate(prior_wins)
    work["player_prior_place2_rate"] = safe_rate(prior_place2)
    work["player_prior_place3_rate"] = safe_rate(prior_place3)
    work["player_prior_avg_finish"] = safe_rate(prior_finish_sum)

    # Leakage-safe rolling form: shift first so the current race result is never
    # used to predict itself. These react faster than lifetime priors.
    prior_finish = finish.groupby(player, dropna=False).shift(1)
    prior_win_obs = finish.eq(1).astype(float).where(observed.astype(bool)).groupby(player, dropna=False).shift(1)
    work["player_recent5_avg_finish"] = prior_finish.groupby(player, dropna=False).transform(lambda s: s.rolling(5, min_periods=2).mean())
    work["player_recent10_avg_finish"] = prior_finish.groupby(player, dropna=False).transform(lambda s: s.rolling(10, min_periods=3).mean())
    work["player_recent5_win_rate"] = prior_win_obs.groupby(player, dropna=False).transform(lambda s: s.rolling(5, min_periods=2).mean())
    work["player_recent10_win_rate"] = prior_win_obs.groupby(player, dropna=False).transform(lambda s: s.rolling(10, min_periods=3).mean())
    work["player_form_trend_5_vs_10"] = work["player_recent10_avg_finish"] - work["player_recent5_avg_finish"]

    # Current-meeting form, using only earlier races in the same meeting.
    # Prefer an explicit meeting/cup identifier when available; otherwise the
    # venue + meeting day/date sequence is approximated from contiguous dates.
    if "meeting_id" in work.columns and work["meeting_id"].notna().any():
        meeting_key = work["meeting_id"].astype(str)
    else:
        venue = work.get("venue", pd.Series("", index=work.index)).fillna("").astype(str)
        # Keirin meetings normally span consecutive days. A gap >1 day starts
        # a new inferred meeting for that rider/venue.
        gap = work.groupby([player, venue], dropna=False)["_date_dt"].diff().dt.days
        block = gap.gt(1).groupby([player, venue], dropna=False).cumsum()
        meeting_key = venue + ":" + block.astype(str)
    meeting_group = [player, meeting_key]
    prior_meeting_finish = finish.groupby(meeting_group, dropna=False).shift(1)
    work["meeting_prior_races"] = prior_meeting_finish.groupby(meeting_group, dropna=False).transform("count")
    work["meeting_prior_avg_finish"] = prior_meeting_finish.groupby(meeting_group, dropna=False).transform("mean")
    work["meeting_form_delta"] = work["player_prior_avg_finish"] - work["meeting_prior_avg_finish"]
    prev_meeting_finish = prior_meeting_finish.groupby(meeting_group, dropna=False).shift(1)
    work["meeting_finish_trend"] = prev_meeting_finish - prior_meeting_finish

    event_dates = work["_date_dt"].where(observed.astype(bool))
    prev_dates = event_dates.groupby(player, dropna=False).transform(lambda s: s.ffill().shift(1))
    work["player_prior_days_since_last_race"] = (work["_date_dt"] - prev_dates).dt.days

    work["player_prior_strength"] = (
        work["player_prior_win_rate"].fillna(0) * 10.0
        + work["player_prior_place2_rate"].fillna(0) * 4.0
        + work["player_prior_place3_rate"].fillna(0) * 2.0
        - work["player_prior_avg_finish"].fillna(work["player_prior_avg_finish"].median()) * 0.45
        + np.log1p(work["player_prior_races"].fillna(0)) * 0.1
    )

    if "race_id" in work.columns:
        race_ids = work["race_id"]
        values = pd.to_numeric(work["player_prior_strength"], errors="coerce")
        group = values.groupby(race_ids, dropna=False)
        work["player_prior_strength_rank"] = group.rank(ascending=False, method="average")
        work["player_prior_strength_gap_to_best"] = group.transform("max") - values
        work["player_prior_strength_vs_field"] = values - group.transform("mean")

    work = work.sort_values("_original_order", kind="mergesort").set_index(original_index)
    return work.drop(columns=["_original_order", "_date_dt"], errors="ignore")


def add_player_elo_features(df: pd.DataFrame, k_factor=20.0, base_rating=1500.0) -> pd.DataFrame:
    """Leakage-safe multiplayer Elo plus recency-weighted finish.

    Each row receives the player's rating immediately before that race. Ratings
    are updated only after all starters in the race have been assigned their
    pre-race features.
    """
    df = df.copy()
    if not {"race_id", "player_id", "date"}.issubset(df.columns):
        return df
    work = df.copy()
    work["_orig"] = np.arange(len(work))
    work["_date"] = pd.to_datetime(work["date"], errors="coerce")
    sort_cols = [x for x in ["_date", "race_id", "race_no", "car_no", "_orig"] if x in work.columns]
    work = work.sort_values(sort_cols, kind="mergesort")
    ratings = {}
    histories = {}
    elo_out = pd.Series(np.nan, index=work.index, dtype=float)
    weighted_finish = pd.Series(np.nan, index=work.index, dtype=float)

    for _, idx in work.groupby("race_id", sort=False).groups.items():
        idx = list(idx)
        players = work.loc[idx, "player_id"].astype(str)
        pre = np.array([ratings.get(pid, base_rating) for pid in players], dtype=float)
        elo_out.loc[idx] = pre
        race_date = work.loc[idx, "_date"].iloc[0]
        for row_idx, pid in zip(idx, players):
            hist = histories.get(pid, [])
            vals, weights = [], []
            for d, pos in hist[-20:]:
                age = max((race_date - d).days, 0) if pd.notna(race_date) and pd.notna(d) else 365
                vals.append(pos)
                weights.append(0.5 ** (age / 90.0))
            if weights and sum(weights) > 0:
                weighted_finish.loc[row_idx] = float(np.average(vals, weights=weights))

        finish = pd.to_numeric(work.loc[idx, "finish_pos"], errors="coerce").to_numpy(dtype=float)
        observed = np.isfinite(finish)
        if observed.sum() >= 2:
            n = len(idx)
            delta = np.zeros(n, dtype=float)
            for i in range(n):
                if not observed[i]:
                    continue
                for j in range(n):
                    if i == j or not observed[j]:
                        continue
                    actual = 1.0 if finish[i] < finish[j] else (0.5 if finish[i] == finish[j] else 0.0)
                    expected = 1.0 / (1.0 + 10.0 ** ((pre[j] - pre[i]) / 400.0))
                    delta[i] += actual - expected
                delta[i] /= max(observed.sum() - 1, 1)
            for i, pid in enumerate(players):
                if observed[i]:
                    ratings[pid] = pre[i] + k_factor * delta[i]
                    histories.setdefault(pid, []).append((race_date, float(finish[i])))

    work["player_elo"] = elo_out
    work["player_recent_weighted_finish"] = weighted_finish
    grp = work["player_elo"].groupby(work["race_id"], dropna=False)
    work["player_elo_rank"] = grp.rank(ascending=False, method="average")
    work["player_elo_vs_field"] = work["player_elo"] - grp.transform("mean")
    work = work.sort_values("_orig", kind="mergesort")
    return work.drop(columns=["_orig", "_date"], errors="ignore")


def add_pair_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe direct matchup and leader/second-wheel partnership history."""
    out = df.copy()
    defaults = {
        "h2h_prior_meetings": 0.0, "h2h_prior_win_share": np.nan,
        "line_pair_prior_races": 0.0, "line_pair_second_win_rate": np.nan,
    }
    for col, val in defaults.items():
        out[col] = val
    required = {"race_id","player_id","date"}
    if not required.issubset(out.columns):
        return out
    work = out.copy()
    work["_orig"] = np.arange(len(work))
    work["_date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.sort_values([x for x in ["_date","race_id","race_no","car_no","_orig"] if x in work.columns], kind="mergesort")
    pair_stats = {}
    line_stats = {}
    for _, idx in work.groupby("race_id", sort=False).groups.items():
        idx = list(idx)
        race = work.loc[idx]
        players = race["player_id"].astype(str).tolist()
        finish = pd.to_numeric(race.get("finish_pos"), errors="coerce")
        pos_by_player = dict(zip(players, finish.tolist()))
        for row_idx, pid in zip(idx, players):
            meetings = wins = 0.0
            for opp in players:
                if opp == pid:
                    continue
                key = tuple(sorted((pid, opp)))
                stat = pair_stats.get(key, {"meetings":0.0, "wins":{}})
                meetings += stat["meetings"]
                wins += stat["wins"].get(pid, 0.0)
            work.loc[row_idx, "h2h_prior_meetings"] = meetings
            work.loc[row_idx, "h2h_prior_win_share"] = wins / meetings if meetings else np.nan

        line_id = pd.to_numeric(race.get("line_id"), errors="coerce")
        line_pos = pd.to_numeric(race.get("line_position"), errors="coerce")
        for lid in line_id.dropna().unique():
            leader_rows = race.index[(line_id.eq(lid)) & (line_pos.eq(1))]
            second_rows = race.index[(line_id.eq(lid)) & (line_pos.eq(2))]
            if len(leader_rows) != 1 or len(second_rows) != 1:
                continue
            li, si = leader_rows[0], second_rows[0]
            leader = str(work.at[li, "player_id"]); second = str(work.at[si, "player_id"])
            stat = line_stats.get((leader, second), {"races":0.0, "second_wins":0.0})
            work.loc[si, "line_pair_prior_races"] = stat["races"]
            work.loc[si, "line_pair_second_win_rate"] = stat["second_wins"] / stat["races"] if stat["races"] else np.nan

        # Update only after all pre-race features have been assigned.
        observed = [(pid, pos_by_player.get(pid)) for pid in players if pd.notna(pos_by_player.get(pid))]
        for i, (pid, pos) in enumerate(observed):
            for opp, opp_pos in observed[i+1:]:
                key = tuple(sorted((pid, opp)))
                stat = pair_stats.setdefault(key, {"meetings":0.0, "wins":{}})
                stat["meetings"] += 1.0
                if pos < opp_pos:
                    stat["wins"][pid] = stat["wins"].get(pid, 0.0) + 1.0
                elif opp_pos < pos:
                    stat["wins"][opp] = stat["wins"].get(opp, 0.0) + 1.0
        for lid in line_id.dropna().unique():
            leader_rows = race.index[(line_id.eq(lid)) & (line_pos.eq(1))]
            second_rows = race.index[(line_id.eq(lid)) & (line_pos.eq(2))]
            if len(leader_rows) != 1 or len(second_rows) != 1:
                continue
            leader = str(work.at[leader_rows[0], "player_id"]); second = str(work.at[second_rows[0], "player_id"])
            stat = line_stats.setdefault((leader, second), {"races":0.0, "second_wins":0.0})
            stat["races"] += 1.0
            if pd.to_numeric(work.at[second_rows[0], "finish_pos"], errors="coerce") == 1:
                stat["second_wins"] += 1.0
    work = work.sort_values("_orig", kind="mergesort")
    for col in defaults:
        out[col] = work[col].to_numpy()
    return out


def add_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    race_key = "race_id" if "race_id" in df.columns else None

    def num(col, default=np.nan):
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    score = num("score")
    win_rate = num("win_rate")
    place2_rate = num("place2_rate")
    place3_rate = num("place3_rate")
    recent_avg_finish = num("recent_avg_finish")

    # Compact, human-readable rider strength prior. The model still learns the
    # final weights, but this gives it a direct "who is strong" signal.
    df["rider_strength"] = (
        score.fillna(score.median()) * 0.08
        + win_rate.fillna(0) * 10.0
        + place2_rate.fillna(0) * 4.0
        + place3_rate.fillna(0) * 2.0
        - recent_avg_finish.fillna(recent_avg_finish.median()) * 0.45
    )

    if race_key:
        race_ids = df[race_key]
        for col, ascending in [
            ("rider_strength", False),
            ("score", False),
            ("win_rate", False),
            ("place2_rate", False),
            ("place3_rate", False),
            ("recent_avg_finish", True),
        ]:
            values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            ranks = values.groupby(race_ids, dropna=False).rank(ascending=ascending, method="average")
            df[f"{col}_rank"] = ranks

        strength_group = df["rider_strength"].groupby(race_ids, dropna=False)
        score_group = score.groupby(race_ids, dropna=False)
        win_rate_group = win_rate.groupby(race_ids, dropna=False)
        df["rider_strength_gap_to_best"] = strength_group.transform("max") - df["rider_strength"]
        df["rider_strength_vs_field"] = df["rider_strength"] - strength_group.transform("mean")
        df["score_gap_to_best"] = score_group.transform("max") - score
        df["score_vs_field"] = score - score_group.transform("mean")
        df["win_rate_gap_to_best"] = win_rate_group.transform("max") - win_rate

        # Pre-race interaction features: strength of own line and pressure from
        # rival attacking lines. These use only fields known before the race.
        line_id = pd.to_numeric(df.get("line_id"), errors="coerce")
        line_pos = pd.to_numeric(df.get("line_position"), errors="coerce")
        attack = num("back_count", 0).fillna(0) + num("front_runner_count", 0).fillna(0)
        df["_attack_tmp"] = attack
        df["_line_key_tmp"] = race_ids.astype(str) + ":" + line_id.fillna(-1).astype(str)
        line_key = df["_line_key_tmp"]

        df["line_strength_mean"] = df["rider_strength"].groupby(line_key, dropna=False).transform("mean")
        df["line_strength_max"] = df["rider_strength"].groupby(line_key, dropna=False).transform("max")

        leader_strength = df["rider_strength"].where(line_pos.eq(1))
        second_strength = df["rider_strength"].where(line_pos.eq(2))
        df["line_leader_strength"] = leader_strength.groupby(line_key, dropna=False).transform("max")
        df["line_second_strength"] = second_strength.groupby(line_key, dropna=False).transform("max")
        df["strength_vs_line_leader"] = df["rider_strength"] - df["line_leader_strength"]
        df["strength_vs_line_second"] = df["rider_strength"] - df["line_second_strength"]

        line_attack = attack.groupby(line_key, dropna=False).transform("sum")
        race_attack = attack.groupby(race_ids, dropna=False).transform("sum")
        df["race_attack_pressure"] = race_attack
        df["other_line_attack_pressure"] = (race_attack - line_attack).clip(lower=0)
        df = df.drop(columns=["_attack_tmp", "_line_key_tmp"], errors="ignore")
    else:
        df["rider_strength_rank"] = np.nan
        df["rider_strength_gap_to_best"] = np.nan
        df["rider_strength_vs_field"] = np.nan
        df["score_rank"] = np.nan
        df["score_gap_to_best"] = np.nan
        df["score_vs_field"] = np.nan
        df["win_rate_rank"] = np.nan
        df["win_rate_gap_to_best"] = np.nan
        df["place2_rate_rank"] = np.nan
        df["place3_rate_rank"] = np.nan
        df["recent_avg_finish_rank"] = np.nan
        for col in [
            "line_leader_strength","line_second_strength","strength_vs_line_leader",
            "strength_vs_line_second","line_strength_mean","line_strength_max",
            "other_line_attack_pressure","race_attack_pressure",
        ]:
            df[col] = np.nan

    return df


def prepare_features(df: pd.DataFrame, fill_values=None):
    df = add_categorical_codes(df)
    # Live prediction may already contain Elo calculated on history+today.
    # Do not overwrite that with an Elo calculation using today's races only.
    if "player_elo" not in df.columns or pd.to_numeric(df["player_elo"], errors="coerce").notna().sum() == 0:
        df = add_player_elo_features(df)
    if "h2h_prior_meetings" not in df.columns or pd.to_numeric(df["h2h_prior_meetings"], errors="coerce").sum() == 0:
        df = add_pair_history_features(df)
    df = add_strength_features(df)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    X = df[FEATURE_COLS].copy()

    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if fill_values is None:
        fill_values = {}
        for col in FEATURE_COLS:
            med = X[col].median()
            if pd.isna(med):
                med = 0.0
            fill_values[col] = float(med)

    X = X.fillna(fill_values)
    return X, fill_values


def normalize_race_prob(df: pd.DataFrame, raw_col="p_raw", out_col="p_win") -> pd.DataFrame:
    df = df.copy()
    df[raw_col] = np.clip(pd.to_numeric(df[raw_col], errors="coerce").fillna(1e-6), 1e-6, 1.0)
    sums = df.groupby("race_id")[raw_col].transform("sum")
    df[out_col] = df[raw_col] / sums.replace(0, np.nan)
    df[out_col] = df[out_col].fillna(1.0 / df.groupby("race_id")["race_id"].transform("count"))
    return df
