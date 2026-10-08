"""Rolling-year observed rider knowledge and independent department shadow forecasts."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import HISTORY_CSV, OUTPUT_DIR, valid_finish_mask
from betting_logic import score_riders, select_race


def frame_read(path):
    try:
        return pd.read_csv(path, dtype={"race_id": str, "player_id": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def collect_observations(race_data, url, output_dir=OUTPUT_DIR):
    """Retain real official result factors/events; never infer missing events."""
    race = race_data.get("race", {})
    race_id = str(race.get("id", ""))
    if int(race.get("status", 0) or 0) < 3 or not race_data.get("results"):
        return
    from race_features import build_entry_rows
    date = datetime.strptime(race_id[-8:], "%Y%m%d").strftime("%Y-%m-%d")
    rows = build_entry_rows(race_data, date, "", race.get("number", 0), race_id, url, True)
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rider_official_observations.csv"
    combined = pd.concat([frame_read(path), pd.DataFrame(rows)], ignore_index=True, sort=False)
    combined.drop_duplicates(["race_id", "player_id"], keep="last").to_csv(path, index=False)


def collect_prior_record_events(race_data, url, output_dir=OUTPUT_DIR):
    """Observed past-result records embedded in a current official race card."""
    from race_features import prior_results, _race_date_from_id
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    start = (pd.Timestamp(today) - pd.DateOffset(years=3)).date()
    rows = []
    names = {str(player.get("id")): str(player.get("name", "")) for player in race_data.get("players", [])}
    for record in race_data.get("records", []):
        for result in prior_results(record, today.isoformat(), limit=100):
            date = _race_date_from_id(result.get("raceId"))
            if date is None or date < start:
                continue
            row = {"race_id": str(result["raceId"]), "player_id": str(result.get("playerId", record.get("playerId"))),
                   "date": date.isoformat(), "finish_pos": result.get("order"), "result_factor": result.get("factor"),
                   "observation_source_url": url, "observation_kind": "official_prior_result_record"}
            row["player_name"] = names.get(row["player_id"], "")
            for key in ["back", "spurtSucceeded", "thrustSucceeded", "leftBehind", "splitLine", "snatchSucceeded", "competeSucceeded", "hasAccident"]:
                row[f"result_event_{key}"] = result.get(key)
            rows.append(row)
    if rows:
        folder = output_dir / "company"; folder.mkdir(parents=True, exist_ok=True)
        path = folder / "rider_official_observations.csv"
        previous = frame_read(path)
        fresh = pd.DataFrame(rows).drop_duplicates(["race_id", "player_id"], keep="last")
        if not previous.empty:
            fresh = fresh.set_index(["race_id", "player_id"]).combine_first(previous.set_index(["race_id", "player_id"])).reset_index()
        fresh.to_csv(path, index=False)
    return len(rows)


def backfill_observations(entries, limit=40, history_path=HISTORY_CSV, output_dir=OUTPUT_DIR):
    """Bounded official-result enrichment, including current-card past records."""
    history = frame_read(history_path)
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Tokyo")).date())
    urls = []
    if not history.empty and {"date", "source_url", "race_id", "player_id"}.issubset(history):
        dates = pd.to_datetime(history.date, errors="coerce")
        candidates = history[dates.ge(today - pd.DateOffset(years=3)) & dates.lt(today)].copy()
        existing = frame_read(output_dir / "company/rider_official_observations.csv")
        known = set(existing.race_id.astype(str)) if "race_id" in existing else set()
        candidates = candidates[~candidates.race_id.astype(str).isin(known)]
        active = set(entries.player_id.astype(str)) if "player_id" in entries else set()
        candidates["_active"] = candidates.player_id.astype(str).isin(active)
        races = candidates.sort_values(["_active", "date"], ascending=False).drop_duplicates("race_id")
        urls = races.source_url.dropna().astype(str).tolist()
    # Current race cards carry actual past factor/event records even when the
    # legacy archive omitted source URLs; these are never fabricated by style.
    current = entries.source_url.dropna().astype(str).unique().tolist() if "source_url" in entries else []
    urls = list(dict.fromkeys(current + urls))
    urls = [url for url in urls if url.startswith("https://www.winticket.jp/")][:limit]
    from fetch_today_entries import http_get, extract_preloaded_state, find_query_data
    def fetch(url):
        return url, find_query_data(extract_preloaded_state(http_get(url)), "FETCH_KEIRIN_RACE")
    fetched = failures = observed_rows = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, url) for url in urls]
        for future in as_completed(futures):
            try:
                url, data = future.result()
                if data:
                    collect_observations(data, url, output_dir)
                    observed_rows += collect_prior_record_events(data, url, output_dir)
                    fetched += 1
            except Exception as exc:
                failures += 1
                print(f"annual official-event backfill unavailable: {exc}")
    report = {"requested": len(futures), "fetched": fetched, "failures": failures, "prior_result_rows_seen": observed_rows,
              "limit_per_run": limit, "scope": "actual past-year events; current riders prioritized"}
    folder = output_dir / "company"; folder.mkdir(parents=True, exist_ok=True)
    (folder / "annual_event_backfill.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return report


def build_annual_profiles(asof=None, history_path=HISTORY_CSV, output_dir=OUTPUT_DIR):
    asof = pd.Timestamp(asof or datetime.now(ZoneInfo("Asia/Tokyo")).date()).normalize().tz_localize(None)
    start = asof - pd.DateOffset(years=1)
    start2 = asof - pd.DateOffset(years=2)
    start3 = asof - pd.DateOffset(years=3)
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    observations = folder / "rider_official_observations.csv"
    saved = folder / "annual_rider_knowledge.json"
    if not history_path.exists() and saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        previous["history_refresh_status"] = "preserved_until_full_history_is_mounted"
        return previous
    digest = hashlib.sha256(str(asof.date()).encode())
    for path in [history_path, observations, Path(__file__)]:
        if path.exists():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    fingerprint = digest.hexdigest()
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint:
            return previous
    history = frame_read(history_path)
    total_races = int(history.race_id.nunique()) if "race_id" in history else 0
    primary_ids = set(history.race_id.astype(str)) if "race_id" in history else set()
    primary_dates = pd.to_datetime(history.get("date", pd.Series(dtype=str)), errors="coerce")
    annual_archive_races = int(history.loc[primary_dates.ge(start) & primary_dates.lt(asof), "race_id"].nunique()) if "race_id" in history else 0
    extra = frame_read(observations)
    supplemental_races = len(set(extra.race_id.astype(str)) - primary_ids) if "race_id" in extra else 0
    if not history.empty and not extra.empty:
        # Missing fields from a supplemental observation cannot erase old values.
        history = history.set_index(["race_id", "player_id"])
        extra = extra.replace("", np.nan).set_index(["race_id", "player_id"])
        history = extra.combine_first(history).reset_index()
    elif not extra.empty:
        history = extra
    reference_races = int(history.race_id.nunique()) if "race_id" in history else 0
    profiles = {}
    annual_races = 0
    coverage = {}
    if not history.empty and {"date", "player_id", "race_id", "finish_pos"}.issubset(history):
        dates = pd.to_datetime(history.date, errors="coerce")
        annual = history[dates.ge(start3) & dates.lt(asof)].copy()
        annual = annual.drop_duplicates(["race_id", "player_id"], keep="last")
        annual["_date"] = pd.to_datetime(annual.date, errors="coerce")
        finish = pd.to_numeric(annual.finish_pos, errors="coerce")
        field = pd.to_numeric(annual.get("entries_number", pd.Series(9, index=annual.index)), errors="coerce").fillna(9)
        annual["_valid"] = valid_finish_mask(annual)
        for position in [1, 2, 3]:
            annual[f"_p{position}"] = finish.eq(position).astype(int)
        annual["_weight"] = np.where(annual._date.ge(start), 1.0, np.where(annual._date.ge(start2), .5, .25))
        latest_year = annual[annual._date.ge(start)]
        annual_races = int(latest_year.race_id.nunique())
        coverage = {name: int(latest_year[name].notna().sum()) for name in annual.columns
                    if name.startswith("result_event_") or name == "result_factor"}
        for player_id, group in annual.groupby("player_id", sort=False):
            recent_year = group[group._date.ge(start)]
            year_count = int(recent_year._valid.sum())
            years = 1 if year_count >= 30 else 2 if year_count >= 15 else 3
            reference_start = [start, start2, start3][years - 1]
            reference = group[group._date.ge(reference_start)].copy()
            if not int(reference._valid.sum()):
                continue
            def stats(rows):
                count = int(rows._valid.sum())
                mass = float(rows.loc[rows._valid, "_weight"].sum())
                return {"races": count, "effective_races": mass,
                        "rates": [float((rows.loc[rows._valid, f"_p{p}"] * rows.loc[rows._valid, "_weight"]).sum() / mass)
                                  if mass else None for p in [1, 2, 3]]}
            evaluation = {**stats(reference), "entries": len(reference),
                          "unplaced_rows": int((~reference._valid).sum()),
                          "effective_entries": float(reference._weight.sum()),
                          "effective_unplaced": float(reference.loc[~reference._valid, "_weight"].sum())}
            profile = {**stats(recent_year), "entries": len(recent_year), "unplaced_rows": int((~recent_year._valid).sum()),
                       "evaluation": evaluation, "reference_years": years, "reference_start": str(reference_start.date()),
                       "year_weights": [1.0, .5, .25][:years], "annual_races": year_count,
                       "recent90": stats(group[group._date.ge(asof - pd.Timedelta(days=90))]),
                       "line_positions": {}, "tactics": {}, "events": {},
                       "style_counts": reference.get("style", pd.Series(dtype=str)).dropna().astype(str).value_counts().to_dict()}
            group = reference
            if "line_position" in group:
                for position, rows in group.groupby("line_position"):
                    if pd.notna(position):
                        profile["line_positions"][str(int(float(position)))] = stats(rows)
            if "result_factor" in group:
                profile["tactics"] = group.result_factor.dropna().astype(str).loc[lambda s: s.ne("")].value_counts().to_dict()
                profile["winning_tactics"] = group.loc[group._p1.eq(1), "result_factor"].dropna().astype(str).loc[lambda s: s.ne("")].value_counts().to_dict()
            for column in group:
                if column.startswith("result_event_"):
                    observed = group[column].dropna().astype(str).str.lower()
                    profile["events"][column.removeprefix("result_event_")] = {
                        "observed": len(observed), "true": int(observed.isin(["true", "1", "1.0"]).sum()),
                        "effective_observed": float(group.loc[group[column].notna(), "_weight"].sum()),
                        "effective_true": float(group.loc[group[column].astype(str).str.lower().isin(["true", "1", "1.0"]), "_weight"].sum()),
                        "true_results": stats(group.loc[group[column].astype(str).str.lower().isin(["true", "1", "1.0"])])}
            names = group.get("player_name", pd.Series(dtype=str)).dropna().astype(str).loc[lambda value: value.ne("")]
            profile["name"] = str(names.iloc[-1]) if len(names) else ""
            profiles[str(player_id)] = profile
    report = {"updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
              "fingerprint": fingerprint, "asof_date": str(asof.date()), "window_start": str(start.date()),
              "window_end_exclusive": str(asof.date()), "reference_window_start": str(start3.date()),
              "reference_policy": {"annual_30_or_more": 1, "annual_15_to_29": 2, "annual_0_to_14": 3,
                                   "year_weights": [1.0, .5, .25]}, "total_archive_races": total_races,
              "annual_races": annual_races, "annual_archive_races": annual_archive_races,
              "supplemental_result_races": supplemental_races, "reference_races_including_supplemental": reference_races,
              "players": len(profiles), "profiles": profiles,
              "observation_coverage": coverage, "status": "ready" if profiles else "history_unavailable",
              "policy": "full archive retained; adaptive previous 1/2/3 calendar years weighted 1/.5/.25 for all riders; same-day/future results excluded",
              "limitations": ["脚質と決まり手を区別。決まり手・行動の未取得分は推測しない。",
                               "部署予想は暫定の影予想。実戦検証前に本番モデルを置換しない。"]}
    saved.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def forecast_departments(pred, odds, report, now, output_dir=OUTPUT_DIR):
    """Publish one explicit response per race and specialist, even without annual data.

    The decision to forecast is distinct from the decision to issue a wager.
    Closed races can only display an existing pre-close snapshot; there is no
    retrospective prediction derived from the final result.
    """
    started = time.monotonic()
    from department_position_experiment import make_bundle, append_bundle
    comparison_decisions = []
    cutoff = pd.to_datetime(report.get("window_end_exclusive"), errors="coerce")
    asof = pd.to_datetime(report.get("asof_date"), errors="coerce")
    today = now.astimezone(ZoneInfo("Asia/Tokyo")).date()
    if pd.isna(cutoff) or pd.isna(asof) or cutoff != asof or cutoff.date() > today:
        raise ValueError("Annual reference cutoff is missing, inconsistent or later than prediction date")

    departments = ("data_department", "pace_department", "line_department", "risk_department")
    feature_map = {
        "data_department": ["annual_place_rates", "model_fallback_if_missing"],
        "pace_department": ["annual_place_rates", "recent90", "observed_behavior_result_associations"],
        "line_department": ["annual_place_rates", "line_position_rates"],
        "risk_department": ["annual_place_rates", "sample_reliability", "unplaced_rate"],
    }
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    ledger = folder / "annual_department_prediction_ledger.jsonl"
    saved = {}
    if ledger.exists():
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    old = json.loads(line)
                    key = (str(old["race_id"]), old["department"])
                    order = old.get("top3_cars")
                    valid_order = (isinstance(order, list) and len(order) == 3
                                   and len({str(car) for car in order}) == 3)
                    # Legacy pre-close ledgers stored only a winner and have no
                    # forecast_available / top3_cars. Preserve their evidence
                    # without falsely claiming they predicted the full order.
                    if "forecast_available" not in old or (old["forecast_available"] and not valid_order):
                        old["forecast_available"] = bool(valid_order)
                        old["top3_cars"] = list(order) if valid_order else []
                        old["display_status"] = ("legacy_preclose_full_order" if valid_order
                                                 else "legacy_preclose_winner_only")
                    if (old.get("winner_car") is not None
                            and datetime.fromisoformat(old["snapshot_at"]).timestamp() < float(old["close_at"])
                            and (key not in saved or old["snapshot_at"] > saved[key]["snapshot_at"])):
                        saved[key] = old
                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    continue

    proposals = []
    fresh = []
    for race_id, race in pred.groupby("race_id", sort=False):
        race_id = str(race_id)
        close = pd.to_numeric(race.iloc[0].get("close_at"), errors="coerce")
        race_date = pd.to_datetime(race.iloc[0].get("date"), errors="coerce")
        valid_close = bool(pd.notna(close) and np.isfinite(close))
        valid_date = bool(pd.notna(race_date) and race_date.date() >= cutoff.date())
        valid_riders = (len(race) >= 3 and race["car_no"].nunique() == len(race)
                        and pd.to_numeric(race["car_no"], errors="coerce").between(1, 9).all())
        can_forecast = valid_close and close > now.timestamp() and valid_date and valid_riders
        if not can_forecast:
            if not valid_riders:
                status = "invalid_rider_entries"
            elif not valid_close:
                status = "closing_time_unverified"
            elif not valid_date:
                status = "historical_race_no_new_forecast"
            else:
                status = "closed_without_preclose_forecast"
            for department in departments:
                prior = saved.get((race_id, department))
                if prior:
                    proposals.append({**prior, "display_status":
                                      ("preclose_forecast_preserved" if prior["forecast_available"]
                                       else prior.get("display_status", "legacy_preclose_winner_only"))})
                else:
                    proposals.append({
                        "department": department, "race_id": race_id,
                        "venue": str(race.iloc[0].get("venue", "")),
                        "race_no": int(race.iloc[0].get("race_no", 0)),
                        "close_at": float(close) if valid_close else None,
                        "snapshot_at": now.isoformat(timespec="seconds"),
                        "winner_car": None, "top3_cars": [],
                        "forecast_available": False, "display_status": status,
                        "tickets": [], "main_count": 0, "hole_count": 0,
                        "ticket_decision": "not_evaluated",
                        "purchase_authorized": False,
                    })
            continue

        profiles = [report.get("profiles", {}).get(str(pid)) for pid in race.player_id]
        missing_reference = sum(not bool(p) for p in profiles)
        market = odds[odds.race_id.astype(str).eq(race_id)] if not odds.empty and "race_id" in odds else pd.DataFrame()
        from quote_quality import validate_quotes
        if not market.empty:
            market = validate_quotes(market, now.timestamp(), float(close))
        fallback = np.stack([
            np.maximum(pd.to_numeric(race.get(col, pd.Series(0, index=race.index)), errors="coerce").fillna(0).to_numpy(dtype=float), 0)
            for col in ("p_win", "p_second", "p_third")
        ], axis=1)
        for column in range(3):
            if fallback[:, column].sum() <= 0:
                fallback[:, column] = 1.0 / len(race)
            else:
                fallback[:, column] /= fallback[:, column].sum()

        experiment_inputs = {}
        for department in departments:
            values = []
            for row_index, ((_, rider), profile) in enumerate(zip(race.iterrows(), profiles)):
                baseline = fallback[row_index].copy()
                if not profile:
                    values.append(baseline)
                    continue
                evaluation = profile.get("evaluation", profile)
                count = float(evaluation.get("effective_races", evaluation.get("races", 0)))
                rates = (np.array(evaluation["rates"], dtype=float) * count + baseline * 20) / (count + 20)
                context = None
                if department == "pace_department":
                    context = profile.get("recent90")
                if department == "line_department":
                    position = pd.to_numeric(rider.get("line_position"), errors="coerce")
                    context = profile.get("line_positions", {}).get(str(int(position))) if pd.notna(position) else None
                if context and context.get("races", 0):
                    mass = float(context.get("effective_races", context["races"]))
                    weight = mass / (mass + 20)
                    rates = rates * (1 - weight) + np.array(context["rates"]) * weight
                if department == "pace_department":
                    associations, weights = [], []
                    for event in profile.get("events", {}).values():
                        result = event.get("true_results", {})
                        observed = float(event.get("observed", 0))
                        n = float(result.get("effective_races", result.get("races", 0)))
                        if observed >= 10 and n >= 5:
                            associations.append((np.array(result["rates"]) * n + rates * 20) / (n + 20))
                            weights.append(float(event.get("effective_true", event.get("true", 0))) /
                                           max(1.0, float(event.get("effective_observed", observed))))
                    if weights and sum(weights) > 0:
                        strength = min(.25, sum(weights) / len(weights) * count / (count + 40))
                        rates = rates * (1 - strength) + np.average(associations, axis=0, weights=weights) * strength
                if department == "risk_department":
                    entries = max(float(evaluation.get("effective_entries", evaluation.get("entries", 0))), 1)
                    unplaced = float(evaluation.get("effective_unplaced", evaluation.get("unplaced_rows", 0)))
                    reliability = count / (count + 40) * max(0, 1 - unplaced / entries)
                    rates = rates * reliability + baseline * (1 - reliability)
                values.append(np.maximum(np.nan_to_num(rates, nan=0, posinf=0, neginf=0), 0))
            matrix = np.array(values, dtype=float)
            for column in range(3):
                total = matrix[:, column].sum()
                matrix[:, column] = matrix[:, column] / total if total > 0 else fallback[:, column]
            temp = race.copy()
            temp["p_win"] = matrix[:, 0]
            for j, name in enumerate(("first", "second", "third")):
                temp[f"score_{name}"] = matrix[:, j] * 100
                temp[f"rank_{name}"] = temp[f"score_{name}"].rank(ascending=False, method="first").astype(int)
            for name in ("second", "third"):
                temp[f"department_score_{name}"] = temp[f"score_{name}"]
            experiment_inputs[department] = temp.copy()
            temp["position_score_source"] = ("annual_empirical_shadow" if missing_reference == 0
                                              else "annual_shadow_with_model_fallback")
            # Optimise a three-position scenario with distinct cars. This is
            # advisory, not a calibrated trifecta probability.
            from itertools import permutations
            best = max(
                permutations(range(len(race)), 3),
                key=lambda indices: (
                    sum(np.log(max(matrix[index, pos], 1e-12)) for pos, index in enumerate(indices)),
                    tuple(-int(race.iloc[index]["car_no"]) for index in indices),
                ),
            )
            top3 = [int(race.iloc[index]["car_no"]) for index in best]
            try:
                riders = score_riders(temp, market, preserve_position_scores=department != "risk_department")
                candidates, plan = select_race(riders, market)
                selected = candidates[candidates.is_selected]
                tickets = [
                    {"buy": str(t.buy), "group": str(t.ticket_group),
                     "prob": float(t.prob), "ev": float(t.ev)}
                    for t in selected.itertuples()
                ]
                main_count, hole_count = int(plan["main_count"]), int(plan["hole_count"])
                ticket_decision = "eligible" if tickets else "skipped_by_ev_or_odds"
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                tickets, main_count, hole_count = [], 0, 0
                ticket_decision = f"ticket_calculation_unavailable:{type(exc).__name__}"
            proposal = {
                "department": department, "race_id": race_id,
                "venue": str(race.iloc[0].get("venue", "")), "race_no": int(race.iloc[0].get("race_no", 0)),
                "close_at": float(close), "snapshot_at": now.isoformat(timespec="seconds"),
                "winner_car": top3[0], "top3_cars": top3,
                "forecast_available": True,
                "display_status": ("model_fallback_reference_missing" if missing_reference
                                   else "independent_annual_reference"),
                "missing_reference_riders": missing_reference,
                "rider_year_races": {str(int(car)): (p.get("races", 0) if p else 0)
                                     for car, p in zip(race.car_no, profiles)},
                "rider_reference": {str(int(car)): {
                    "years": p.get("reference_years", 1) if p else 0,
                    "races": p.get("evaluation", p).get("races", 0) if p else 0,
                    "effective_races": p.get("evaluation", p).get("effective_races", p.get("races", 0)) if p else 0}
                    for car, p in zip(race.car_no, profiles)},
                "tickets": tickets, "main_count": main_count, "hole_count": hole_count,
                "ticket_decision": ticket_decision,
                "probability_status": "provisional_annual_shadow",
                "ticket_strategy": "risk_legacy_unchanged" if department == "risk_department" else "preserve_only_v1",
                "reference_cutoff_exclusive": report["window_end_exclusive"],
                "features_used": feature_map[department],
                "purchase_authorized": False,
            }
            proposals.append(proposal)
            fresh.append(proposal)
        completed = now + timedelta(seconds=time.monotonic() - started)
        try:
            bundle, reason = make_bundle(race, experiment_inputs, market, completed, report["window_end_exclusive"])
            completed = now + timedelta(seconds=time.monotonic() - started)
            if bundle is not None and not append_bundle(bundle, completed, output_dir):
                reason = "expired_during_calculation"
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            reason = f"calculation_unavailable:{type(exc).__name__}"
        comparison_decisions.append({"race_id": race_id, "reason": reason})
    (folder / "annual_position_experiment_capture.json").write_text(
        json.dumps({"updated_at_jst": now.isoformat(), "decisions": comparison_decisions},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "annual_department_predictions.json").write_text(
        json.dumps({
            "updated_at_jst": now.isoformat(timespec="seconds"),
            "annual_races": report["annual_races"],
            "coverage_policy": "every known race x four departments; no post-close backfill",
            "required_departments": list(departments), "race_count": int(pred["race_id"].nunique()),
            "forecast_count": sum(bool(p.get("forecast_available", False)) for p in proposals),
            "proposals": proposals,
        }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if fresh:
        with ledger.open("a", encoding="utf-8") as handle:
            for proposal in fresh:
                handle.write(json.dumps(proposal, ensure_ascii=False, allow_nan=False) + "\n")
    from strategist_validation import build_strategist_validation
    build_strategist_validation(output_dir, now)
    return proposals


def audit_department_predictions(output_dir=OUTPUT_DIR):
    from department_position_experiment import build_report
    build_report(output_dir)
    from verified_live_audit import build_verified_live_audit
    verified_live = build_verified_live_audit(output_dir)
    folder = output_dir / "company"
    folder.mkdir(parents=True, exist_ok=True)
    ledger = folder / "annual_department_prediction_ledger.jsonl"
    latest = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if datetime.fromisoformat(row["snapshot_at"]).timestamp() >= row["close_at"]:
                continue
            key = (row["department"], row["race_id"])
            if key not in latest or row["snapshot_at"] > latest[key]["snapshot_at"]:
                latest[key] = row
    saved = folder / "annual_department_settled.json"
    previous = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else []
    settled = {(r["department"], r["race_id"]): r for r in previous}
    path = output_dir / "latest_results.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    for result in results:
        if str(result.get("official_result_available")).lower() not in {"true", "1"}:
            continue
        actual = str(result.get("actual_trifecta") or "")
        if len(actual.split("-")) != 3:
            continue
        for key, forecast in latest.items():
            if key[1] == str(result["race_id"]):
                odds = pd.to_numeric(result.get("actual_trifecta_odds"), errors="coerce")
                settled[key] = {**forecast, "actual": actual,
                    "actual_odds": float(odds) if pd.notna(odds) and 0 < odds < float("inf") else None}
    saved.write_text(json.dumps(list(settled.values()), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    comparisons = []
    for department in ["data_department", "pace_department", "line_department", "risk_department"]:
        rows = [r for r in settled.values() if r["department"] == department]
        top_hits = sum(str(r["winner_car"]) == r["actual"].split("-")[0] for r in rows)
        valid = [r for r in rows if r["actual_odds"] is not None and r["tickets"]]
        stake = sum(len(r["tickets"]) * 100 for r in valid)
        returned = sum(r["actual_odds"] * 100 for r in valid if any(t["buy"] == r["actual"] for t in r["tickets"]))
        comparisons.append({"department": department, "races": len(rows), "top1_hits": top_hits,
            "top1_hit_rate": top_hits / len(rows) if rows else None,
            "portfolio_payout_races": len(valid), "flat_stake_yen": stake,
            "flat_return_yen": returned, "flat_return_rate": returned / stake if stake else None,
            "auto_promotion": False})
    report = {"updated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
              "departments": comparisons, "scope": "frozen-before-close independent annual department proposals",
              "snapshot_policy": "legacy_latest_preclose_per_department_not_matched",
              "controlled_comparison": "annual_position_experiment_report.json"}
    (folder / "annual_department_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    knowledge_path = folder / "annual_rider_knowledge.json"
    knowledge = json.loads(knowledge_path.read_text(encoding="utf-8")) if knowledge_path.exists() else {}
    labels = {"data_department": "データ部", "pace_department": "展開部", "line_department": "ライン部", "risk_department": "リスク部"}
    def percent(value):
        return "未集計" if value is None else f"{value * 100:.1f}%"
    cells = "".join(f'<tr><td>{labels[r["department"]]}</td><td>{r["races"]}</td>'
                    f'<td>{percent(r["top1_hit_rate"])}</td><td>{r["portfolio_payout_races"]}</td><td>{percent(r["flat_return_rate"])}</td></tr>' for r in comparisons)
    page = ('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>選手別1〜3年の部署予想</title><style>body{font-family:system-ui;background:#f4f7fb;padding:20px;color:#172b45}'
            'main{max-width:900px;margin:auto;background:white;padding:24px;border-radius:16px}'
            'table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}p{line-height:1.8}</style>'
            '<main><a href="../index.html">レース一覧へ戻る</a> ／ <a href="../performance.html">成績と数字の見方</a><h1>選手別1〜3年の部署予想</h1>'
            f'<p>保持している全履歴 {knowledge.get("total_archive_races", 0):,}レース<br>'
            f'選手別の補足結果 {knowledge.get("supplemental_result_races", 0):,}レース（一部選手の記録）<br>'
            f'直近1年の全体履歴 {knowledge.get("annual_archive_races", 0):,}レース<br>'
            f'補足込みの直近1年の参考記録 {knowledge.get("annual_races", 0):,}レース ／ {knowledge.get("players", 0):,}選手<br>'
            f'集計期間 {knowledge.get("window_start", "未取得")} ～ {knowledge.get("window_end_exclusive", "未取得")}の前日</p>'
            f'<p>従来集計 {verified_live["legacy_settled_rows"]}件のうち、締切前時刻を確認できる記録は {verified_live["timestamp_verified_races"]}件です。未確認分を新しい実戦検証に混ぜません。</p>'
            '<p>全選手に共通で、直近1年30走以上は1年、15〜29走は2年、14走以下は3年を参照。重みは直近1年100%、1〜2年前50%、2〜3年前25%です。各部署が成績・最近の調子・ライン位置・出走数を使って独立した検証用予想を作ります。'
            '全履歴は保持します。決まり手と行動記録は、取得できた実測分だけを集計します。</p>'
            '<p>1着候補の的中率は、選んだ選手が勝った割合。3連単買い目の回収率は、各点100円で買ったと仮定した試算です。対象が異なります。実購入の成績ではありません。</p>'
            '<table><tr><th>部署</th><th>1着候補の検証レース</th><th>1着候補の的中率</th><th>買い目の検証レース</th><th>3連単買い目の回収率（各点100円）</th></tr>' + cells + '</table>'
            '<p>発走前に固定した予想のみ検証します。本番モデルの自動置換・購入許可は行いません。'
            '回収率120%は未検証です。</p><a href="annual_department_predictions.json">最新の部署別予想</a>'
            ' ／ <a href="annual_rider_knowledge.json">選手の年間成績と取得状況</a></main></html>')
    player_view = """<h2>選手の成績と参考期間</h2><label for="player-search">選手名・選手IDで検索</label>
<input id="player-search" placeholder="選手名またはID" style="width:90%;padding:12px;margin:12px 0">
<p>出走数が少ない選手の率は参考値です。脚質から決まり手を推測せず、実際に取得できた記録だけを表示します。</p>
<div id="player-list">成績を読み込み中</div><script>
let riderKnowledge = {};
const percent = x => x == null ? '未取得' : (100*x).toFixed(1)+'%';
function renderPlayers() {
 const query = document.getElementById('player-search').value.trim().toLowerCase();
 const target = document.getElementById('player-list'); target.replaceChildren();
 const matched = Object.entries(riderKnowledge).filter(([id,p]) => (id+' '+(p.name||'')).toLowerCase().includes(query));
 for (const [id,p] of matched.slice(0,20)) {
  const article = document.createElement('article'); article.style.borderBottom='1px solid #ddd';
  const title = document.createElement('h3'); title.textContent=(p.name||'名前未取得')+' / ID '+id;
  const rates = document.createElement('p'); rates.textContent='直近1年 '+p.races+'レース：1着 '+percent(p.rates[0])+' / 2着 '+percent(p.rates[1])+' / 3着 '+percent(p.rates[2]);
  const reference = document.createElement('p'); const evaluation=p.evaluation||p; reference.textContent='予想の参考期間：直近'+(p.reference_years||1)+'年 ／ '+evaluation.races+'走 ／ 重み付き参考走数 '+(evaluation.effective_races||evaluation.races).toFixed(1)+' ／ 1着 '+percent(evaluation.rates[0])+'・2着 '+percent(evaluation.rates[1])+'・3着 '+percent(evaluation.rates[2]);
  const line = document.createElement('p'); line.textContent='ライン位置別：'+Object.entries(p.line_positions||{}).map(([pos,v])=>pos+'番手 '+v.races+'R・1着 '+percent(v.rates[0])).join(' ／ ');
  const tactics = document.createElement('p'); const records=Object.entries(p.winning_tactics||{}); tactics.textContent='取得済みの勝利時の決まり手：'+(records.length?records.map(([name,count])=>name+' '+count+'回').join(' ／ '):'未取得');
  const eventLabels = {back:'バック獲得',spurtSucceeded:'先行成功',thrustSucceeded:'突っ張り成功',leftBehind:'離れ',splitLine:'ライン分断',snatchSucceeded:'捲り成功',competeSucceeded:'競り成功',hasAccident:'事故あり'};
  const events = document.createElement('p'); events.textContent='取得済みの動き：'+Object.entries(p.events||{}).filter(([key,v])=>v.observed>0).map(([key,v])=>(eventLabels[key]||key)+' '+v.true+'/'+v.observed+'記録').join(' ／ ');
  article.append(title,rates,reference,line,tactics,events); target.append(article);
 }
 const count = document.createElement('p'); count.textContent=matched.length+'人中、最大20人を表示'; target.append(count);
}
document.getElementById('player-search').addEventListener('input',renderPlayers);
fetch('annual_rider_knowledge.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('load');return r.json()}).then(d=>{riderKnowledge=d.profiles||{};renderPlayers()}).catch(()=>{document.getElementById('player-list').textContent='成績を読み込めませんでした。ページを更新してください。'});
</script>"""
    forecast_view = """<p><a href="annual_position_experiment_report.html">同じレース・時点・点数で2・3着評価の改善案を比較</a></p><p>この従来集計は各部署の締切前最新予想です。初回予想を使う7部署成績とは集計時点が異なり、今回の比較には混ぜません。</p><h2>部署ごとの最新予想（検証用）</h2><p><a href="all_department_predictions.html">全レース7部署の予想・提出状況を見る</a></p><p>取得時点の予想です。締切後に後付けせず、事前予想がないレースは未成立と表示します。購入候補がなくても着順予想は別途提出します。</p><div id="department-forecasts">読み込み中</div><script>
fetch('annual_department_predictions.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('load');return r.json()}).then(d=>{
 const target=document.getElementById('department-forecasts');target.replaceChildren();
 const labels={data_department:'データ部署',pace_department:'展開部署',line_department:'ライン部署',risk_department:'リスク部署'};
 for(const p of d.proposals||[]){
  const article=document.createElement('article');article.style.borderBottom='1px solid #ddd';
  const title=document.createElement('h3');title.textContent=p.venue+' '+p.race_no+'R ／ '+labels[p.department];
  const content=document.createElement('p');content.textContent=p.forecast_available===false?'事前予想未成立：'+(p.display_status||'時刻・出走情報未確認'):'着順予想 '+((p.top3_cars||[p.winner_car]).join('-'))+' ／ 本線候補 '+p.main_count+'点・穴候補 '+p.hole_count+'点'+(p.missing_reference_riders?'（一部選手はモデル補完）':'');
  const tickets=document.createElement('p');tickets.textContent=(p.tickets||[]).map(t=>((t.group==='main'||t.group==='本線')?'本線':'穴')+' '+t.buy+'（期待値 '+t.ev.toFixed(2)+'）').join(' ／ ')||'着順予想は提出済み・購入候補はなし（買い目見送り）';
  article.append(title,content,tickets);target.append(article);
 }
 if(!(d.proposals||[]).length)target.textContent='対象の発走前レースがありません。';
}).catch(()=>document.getElementById('department-forecasts').textContent='予想を読み込めませんでした。');
</script>"""
    page = page.replace('</main></html>', forecast_view + player_view + '</main></html>')
    page = page.replace('<h2>部署ごとの最新予想（検証用）</h2>', '<p><a href="annual_strategist_report.html">軍師の分析・部署の比較検証を見る</a></p><h2>部署ごとの最新予想（検証用）</h2>')
    (folder / "annual_department_report.html").write_text(page, encoding="utf-8")
    from strategist_validation import build_strategist_validation
    build_strategist_validation(output_dir)
    return report


if __name__ == "__main__":
    from common import TODAY_CSV, TODAY_ODDS_CSV
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    entries = frame_read(TODAY_CSV)
    backfill_observations(entries)
    knowledge = build_annual_profiles(now.date())
    market = frame_read(TODAY_ODDS_CSV)
    # Stamp after preparation; network enrichment must not backdate predictions.
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    if not entries.empty:
        entries["p_win"] = 1 / entries.groupby("race_id")["car_no"].transform("size")
        forecast_departments(entries, market, knowledge, now)
    audit_department_predictions()
    print(json.dumps({"archive_races": knowledge["total_archive_races"], "annual_races": knowledge["annual_races"],
                      "players": knowledge["players"]}, ensure_ascii=False))
