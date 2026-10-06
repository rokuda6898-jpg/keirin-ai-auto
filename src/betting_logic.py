"""Provisional, position-specific trifecta strategy. EV = probability * odds.

Scores and sequential probabilities are heuristics, not calibrated ticket models.
Never normalize over only the formation or the available odds.
"""
from itertools import permutations

import numpy as np
import pandas as pd

STRATEGY_VERSION = "position_ev_v5_20261006"
MAIN_EV = 1.10
HOLE_EV = 1.25


def numbers(frame, column, default=np.nan):
    return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce")


def relative(values, fallback):
    values = values.where(np.isfinite(values) & values.ge(0))
    maximum = values.max()
    return (values / maximum).fillna(fallback) if pd.notna(maximum) and maximum > 0 else fallback


def divergence_points(gap):
    return 10 if gap >= 5 else {4: 7, 3: 5, 2: 3}.get(int(max(gap, 0)), 0)


def score_riders(race, odds=None):
    riders = race.copy().sort_values("car_no", kind="mergesort").reset_index(drop=True)
    cars = numbers(riders, "car_no")
    if len(riders) < 3 or not cars.between(1, 9).all() or not cars.mod(1).eq(0).all():
        raise ValueError("race needs at least three valid car numbers")
    riders["car_no"] = cars.astype(int)
    if riders.car_no.duplicated().any():
        raise ValueError("duplicate car_no in race")
    win = numbers(riders, "p_win")
    if not np.isfinite(win).all() or win.lt(0).any() or win.sum() <= 0:
        raise ValueError("p_win must be finite, nonnegative and have positive mass")
    riders["p_win"] = win / win.sum()
    riders["score_first"] = riders.p_win * 100
    strength = relative(riders.p_win, pd.Series(1.0, index=riders.index))
    # WINTICKET secondRate/thirdRate are exact-place rates. Prior features
    # are cumulative top-2/top-3, so subtract to obtain exact-place rates.
    prior2 = (numbers(riders, "player_prior_place2_rate") - numbers(riders, "player_prior_win_rate")).clip(lower=0)
    prior3 = (numbers(riders, "player_prior_place3_rate") - numbers(riders, "player_prior_place2_rate")).clip(lower=0)
    for position, prior in [(2, prior2), (3, prior3)]:
        exact = relative(numbers(riders, f"place{position}_rate"), strength)
        historical = relative(prior, exact)
        context = pd.concat([
            relative(numbers(riders, f"{prefix}_place{position}_rate"), exact)
            for prefix in ["track", "weather", "race_type", "line_role"]
        ], axis=1).mean(axis=1)
        follower = numbers(riders, "line_position").eq(position).astype(float)
        riders["score_second" if position == 2 else "score_third"] = (
            100 * (0.45 * exact + 0.25 * historical + 0.25 * context + 0.05 * strength)
            + 3 * follower
        ).clip(0.01, 100)
    # Retain already gated position specialists. Heuristic fallback is used
    # only when the existing pipeline could not provide validated specialists.
    specialist = riders.get("position_model_source", pd.Series("", index=riders.index)).eq("position_specialists")
    valid = specialist.copy()
    for column in ["p_second", "p_third"]:
        values = numbers(riders, column)
        valid &= np.isfinite(values) & values.ge(0) & values.le(1)
    if valid.all() and specialist.all():
        for position, column in [("second", "p_second"), ("third", "p_third")]:
            values = numbers(riders, column)
            if values.sum() > 0:
                riders[f"score_{position}"] = (100 * values / values.sum()).clip(lower=0.01)
        riders["position_score_source"] = "validated_position_specialists"
    else:
        riders["position_score_source"] = "provisional_rate_scores"
    for position in ["first", "second", "third"]:
        riders[f"rank_{position}"] = riders[f"score_{position}"].rank(ascending=False, method="first").astype(int)
    # An explicit rider popularity rank is preferred. Otherwise use the
    # first-place marginal of inverse trifecta odds, only for a FULL market.
    popularity = numbers(riders, "popularity_rank")
    valid_rank = popularity.between(1, len(riders)) & popularity.mod(1).eq(0)
    popularity = popularity.where(valid_rank)
    source = "rider_rank" if popularity.notna().all() and popularity.nunique() == len(riders) else "unavailable"
    if source == "unavailable":
        popularity[:] = np.nan
        market = clean_odds(odds)
        keys = {"-".join(map(str, triple)) for triple in permutations(riders.car_no.astype(int), 3)}
        if keys and set(market.buy) == keys:
            market["head"] = market.buy.str.split("-").str[0].astype(int)
            marginal = (1 / market.odds_used).groupby(market["head"]).sum()
            ranked = marginal.sort_index().rank(ascending=False, method="first")
            popularity = riders.car_no.map(ranked)
            source = "full_trifecta_market_proxy"
    riders["popularity_rank"] = popularity
    riders["popularity_source"] = source
    riders["rank_divergence"] = popularity - riders.rank_first
    riders["undervalued_points"] = riders.rank_divergence.fillna(0).map(divergence_points)
    riders["overpopular_points"] = (-riders.rank_divergence).fillna(0).map(divergence_points)
    return riders


def clean_odds(odds):
    if odds is None or odds.empty:
        return pd.DataFrame(columns=["buy", "odds_used"])
    market = odds.copy()
    if "bet_type" in market:
        market = market[market.bet_type.eq("trifecta")].copy()
    market["odds_used"] = numbers(market, "odds_used")
    market = market[np.isfinite(market.odds_used) & market.odds_used.ge(1)].copy()
    # Conflicting duplicate snapshots use the lower odds conservatively.
    return market.groupby("buy", as_index=False).odds_used.min()


def gap(riders, position, left=0, right=1):
    values = sorted(riders[f"score_{position}"], reverse=True)
    return float(values[left] - values[right]) if len(values) > right else float("inf")


def main_limit(first_gap, third_boundary):
    limit = 6 if first_gap >= 10 else 8 if first_gap >= 7 else 10 if first_gap >= 4 else 12
    return min(12, limit + (2 if third_boundary <= 2 else 0))


def race_plan(riders):
    first = gap(riders, "first")
    second = gap(riders, "second")
    third = gap(riders, "third", 3, 4)
    line_ids = numbers(riders, "line_id").dropna()
    line_count = max(int(line_ids.nunique()), int(numbers(riders, "number_of_lines", 0).fillna(0).max()))
    # Five bounded components total 100. Missing line/market data contributes
    # no invented evidence and is explicitly reported.
    components = {
        "first_closeness": round(30 * np.clip(1 - first / 10, 0, 1), 2),
        "line_competition": 25 if line_count >= 3 else 12.5 if line_count == 2 else 0,
        "second_closeness": round(20 * np.clip(1 - second / 10, 0, 1), 2),
        "third_closeness": round(15 * np.clip(1 - third / 5, 0, 1), 2),
        "market_divergence": int(max(riders.undervalued_points.max(), riders.overpopular_points.max())),
    }
    chaos = round(sum(components.values()), 2)
    fixed = first >= 10
    return {
        "first_gap": first, "third_boundary_gap": third if np.isfinite(third) else None,
        "main_limit": main_limit(first, third), "first_fixed": fixed,
        "fixed_car": int(riders.loc[riders.rank_first.eq(1), "car_no"].iloc[0]) if fixed else None,
        "chaos_index": chaos,
        "chaos_label": "固め" if chaos < 30 else "やや荒れ" if chaos < 50 else "荒れ" if chaos < 70 else "大荒れ警戒",
        "chaos_components": components, "line_count": line_count,
        "popularity_source": riders.popularity_source.iloc[0],
        "probability_method": "position_sequential",
        "position_score_source": riders.position_score_source.iloc[0],
    }


def expanded_pool(riders, position, count):
    ordered = riders.sort_values([f"score_{position}", "car_no"], ascending=[False, True])
    cutoff = ordered.iloc[min(count, len(ordered)) - 1][f"score_{position}"]
    return set(ordered.loc[ordered[f"score_{position}"].ge(cutoff - 2), "car_no"].astype(int))


def generate_formations(riders, plan):
    cars = riders.car_no.astype(int).tolist()
    by_car = riders.set_index("car_no")
    heads = {plan["fixed_car"]} if plan["first_fixed"] else expanded_pool(riders, "first", 2)
    seconds = expanded_pool(riders, "second", 4) | expanded_pool(riders, "first", 3)
    thirds = expanded_pool(riders, "third", 5) | expanded_pool(riders, "second", 4)
    # Strong axis: spread all lower positions before EV compression.
    if plan["first_fixed"]:
        seconds = thirds = set(cars)
    plan["formation"] = {"first": sorted(heads), "second": sorted(seconds), "third": sorted(thirds)}
    rows = []
    for a, b, c in permutations(cars, 3):
        if plan["first_fixed"] and a != plan["fixed_car"]:
            continue
        second_mass = by_car.loc[[x for x in cars if x != a], "score_second"].sum()
        third_mass = by_car.loc[[x for x in cars if x not in (a, b)], "score_third"].sum()
        probability = float(by_car.at[a, "p_win"] * by_car.at[b, "score_second"] / second_mass * by_car.at[c, "score_third"] / third_mass)
        core = a in heads and b in seconds and c in thirds
        value_rider = any(by_car.at[x, "undervalued_points"] >= 3 for x in (a, b, c))
        rows.append({"buy": f"{a}-{b}-{c}", "bet_type": "trifecta", "prob": probability,
                     "head": a, "main_formation": core,
                     "hole_formation": not core or value_rider,
                     "value_rider": value_rider})
    return pd.DataFrame(rows, columns=["buy", "bet_type", "prob", "head", "main_formation", "hole_formation", "value_rider"])


def hole_limit(eligible, plan, riders):
    if eligible.empty:
        return 0
    # 12 requires 12 EV-qualified distinct tickets, chaos >= 60 and
    # multiple heads (or multiple undervalued lower riders under a fixed axis).
    multiple = eligible["head"].nunique() >= 2
    if plan["first_fixed"]:
        multiple = int(riders.undervalued_points.ge(3).sum()) >= 2
    if plan["chaos_index"] >= 60 and len(eligible) >= 12 and multiple:
        return 12
    if multiple:
        return 10
    if gap(riders, "second") <= 2 or gap(riders, "third", 3, 4) <= 2:
        return 8
    return 6


def select_race(riders, odds, main_ev=MAIN_EV, hole_ev=HOLE_EV):
    if not np.isfinite(main_ev) or not np.isfinite(hole_ev) or main_ev < 1 or hole_ev < main_ev:
        raise ValueError("EV thresholds must be finite and 1 <= main <= hole")
    plan = race_plan(riders)
    candidates = generate_formations(riders, plan).merge(clean_odds(odds), on="buy", how="left", validate="one_to_one")
    candidates["ev"] = candidates.prob * candidates.odds_used
    candidates["expected_profit_100yen"] = 100 * (candidates.ev - 1)
    candidates["ticket_group"] = ""
    candidates["is_selected"] = False
    ranked = candidates.sort_values(["ev", "prob", "buy"], ascending=[False, False, True], na_position="last")
    main = ranked[ranked.main_formation & ranked.ev.ge(main_ev)].head(plan["main_limit"])
    candidates.loc[main.index, "ticket_group"] = "本線"
    hole_pool = ranked[ranked.hole_formation & ranked.ev.ge(hole_ev) & ~ranked.index.isin(main.index)]
    plan["hole_limit"] = hole_limit(hole_pool, plan, riders)
    holes = hole_pool.head(plan["hole_limit"])
    candidates.loc[holes.index, "ticket_group"] = "穴"
    candidates.loc[list(main.index) + list(holes.index), "is_selected"] = True
    candidates["candidate_rank"] = candidates.prob.rank(ascending=False, method="first").astype(int)
    candidates["chaos_index"] = plan["chaos_index"]
    candidates["first_fixed"] = plan["first_fixed"]
    candidates["probability_method"] = plan["probability_method"]
    plan.update(main_count=len(main), hole_count=len(holes),
                skip_reason="期待値条件を満たす買い目なし（オッズ未取得含む）" if main.empty and holes.empty else "",
                main_ev=main_ev, hole_ev=hole_ev)
    return candidates, plan
