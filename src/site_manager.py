import re
import html as html_lib
from html.parser import HTMLParser
import json
import shutil
import subprocess
import importlib
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import TODAY_CSV, TODAY_ODDS_CSV, OUTPUT_DIR, RAW_DIR

STATUS_PATH = OUTPUT_DIR / "manager_status.json"
INCIDENT_HISTORY_PATH = OUTPUT_DIR / "manager_incident_history.jsonl"
LAST_GOOD_DIR = RAW_DIR / "last_good"
LAST_GOOD_ENTRIES = LAST_GOOD_DIR / "today_entries.csv"
LAST_GOOD_ODDS = LAST_GOOD_DIR / "today_odds.csv"
RACE_SCHEDULE_PATH = OUTPUT_DIR / "latest_race_schedule.csv"
MAX_REPAIR_ATTEMPTS = 3
RETRY_SECONDS = 5
RECURRENCE_WINDOW_MINUTES = 120
RECURRENCE_RESET_GAP_MINUTES = 30


class RiderButtonParser(HTMLParser):
    """Extract rendered rider identity without depending on HTML attribute order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.riders = {}

    def handle_starttag(self, tag, attrs):
        if str(tag).lower() != "button":
            return
        values = {
            str(key).lower(): html_lib.unescape(value or "").strip()
            for key, value in attrs
            if key
        }
        classes = set(values.get("class", "").split())
        if "rider" not in classes:
            return
        player_id = values.get("data-player-id", "")
        player_name = values.get("data-name", "")
        if player_id and player_name:
            self.riders[player_id] = player_name


def snapshot_last_good():
    """Persist only an already-audited source snapshot as rollback material."""
    LAST_GOOD_DIR.mkdir(parents=True, exist_ok=True)
    if TODAY_CSV.exists():
        shutil.copy2(TODAY_CSV, LAST_GOOD_ENTRIES)
    odds = RAW_DIR / "today_odds.csv"
    if odds.exists():
        shutil.copy2(odds, LAST_GOOD_ODDS)


def rollback_last_good():
    if not LAST_GOOD_ENTRIES.exists():
        return False
    shutil.copy2(LAST_GOOD_ENTRIES, TODAY_CSV)
    if LAST_GOOD_ODDS.exists():
        shutil.copy2(LAST_GOOD_ODDS, RAW_DIR / "today_odds.csv")
    return True


def restore_races_from_cache(problems):
    """Restore known-good individual races before touching the network."""
    cache_dir = RAW_DIR / "race_cache"
    if not cache_dir.exists() or not TODAY_CSV.exists():
        return False
    wanted = set()
    for p in problems or []:
        if p.get("type") in {"missing_entire_races", "missing_riders", "duplicate_riders", "missing_player_identity", "duplicate_player_identity"}:
            wanted.update(str(x) for x in (p.get("race_ids") or ([p.get("race_id")] if p.get("race_id") else [])))
    if not wanted:
        return False
    try:
        base = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
        odds = pd.read_csv(TODAY_ODDS_CSV, dtype={"race_id": str}) if TODAY_ODDS_CSV.exists() else pd.DataFrame()
    except Exception:
        return False
    restored = 0
    for rid in sorted(wanted):
        ep = cache_dir / f"{rid}_entries.csv"
        if not ep.exists():
            continue
        try:
            race = pd.read_csv(ep, dtype={"race_id": str, "player_id": str})
            expected = pd.to_numeric(race.get("entries_number"), errors="coerce").dropna()
            n = int(expected.iloc[0]) if len(expected) else len(race)
            cars = sorted(pd.to_numeric(race["car_no"], errors="coerce").dropna().astype(int).unique().tolist())
            complete, _, _ = importlib.import_module("fetch_today_entries")._entry_rows_complete(race.to_dict("records"))
            if not complete:
                continue
            base = base[~base["race_id"].astype(str).eq(rid)]
            base = pd.concat([base, race], ignore_index=True, sort=False)
            op = cache_dir / f"{rid}_odds.csv"
            if op.exists():
                ro = pd.read_csv(op, dtype={"race_id": str})
                if len(odds) and "race_id" in odds.columns:
                    odds = odds[~odds["race_id"].astype(str).eq(rid)]
                odds = pd.concat([odds, ro], ignore_index=True, sort=False)
            restored += 1
        except Exception:
            continue
    if not restored:
        return False
    fetch_mod = importlib.import_module("fetch_today_entries")
    fetch_mod.save_today_frames(base.to_dict("records"), odds.to_dict("records"))
    ok, gate_problems, _ = validate_source_gate()
    if not ok:
        print(f"race-cache restore failed source gate: {gate_problems}", flush=True)
        rollback_last_good()
        return False
    snapshot_last_good()
    print(f"restored {restored} races from per-race cache", flush=True)
    return True


def repair_missing_races_in_place(problems):
    """Recover only damaged/missing races and merge them into the last good day."""
    race_ids = set()
    for problem in problems or []:
        if problem.get("type") == "missing_entire_races":
            race_ids.update(str(x) for x in problem.get("race_ids", []))
        elif problem.get("type") in {"missing_riders", "duplicate_riders", "missing_player_identity", "duplicate_player_identity"}:
            if problem.get("race_id"):
                race_ids.add(str(problem["race_id"]))
    if not race_ids or not RACE_SCHEDULE_PATH.exists():
        return False

    try:
        fetch_mod = importlib.import_module("fetch_today_entries")
        schedule = pd.read_csv(RACE_SCHEDULE_PATH, dtype={"race_id": str})
        base_path = LAST_GOOD_ENTRIES if LAST_GOOD_ENTRIES.exists() else TODAY_CSV
        base = pd.read_csv(base_path, dtype={"race_id": str})
        odds_path = RAW_DIR / "today_odds.csv"
        base_odds = pd.read_csv(LAST_GOOD_ODDS if LAST_GOOD_ODDS.exists() else odds_path, dtype={"race_id": str}) if (LAST_GOOD_ODDS.exists() or odds_path.exists()) else pd.DataFrame()
        repaired_entries, repaired_odds = [], []

        for race_id in sorted(race_ids):
            rows = schedule[schedule["race_id"].astype(str).eq(race_id)]
            if rows.empty:
                return False
            url = rows.iloc[0].get("source_url")
            success = False
            # Focus retries on the broken race instead of refetching all 82.
            for _ in range(12):
                try:
                    entries, odds = fetch_mod.parse_race_page(url, completeness_attempts=1)
                    check = pd.DataFrame(entries)
                    cars = sorted(pd.to_numeric(check.get("car_no"), errors="coerce").dropna().astype(int).unique().tolist())
                    expected_values = pd.to_numeric(check.get("entries_number"), errors="coerce").dropna()
                    expected = int(expected_values.max()) if len(expected_values) else 0
                    complete, _, _ = fetch_mod._entry_rows_complete(check.to_dict("records"))
                    if complete:
                        # Do not trust a single apparently-complete response.
                        # Fetch once more and require the same car/player mapping.
                        time.sleep(2)
                        confirm_entries, confirm_odds = fetch_mod.parse_race_page(url, completeness_attempts=1)
                        confirm = pd.DataFrame(confirm_entries)
                        def signature(frame):
                            pairs = frame[["car_no", "player_id"]].copy()
                            pairs["car_no"] = pd.to_numeric(pairs["car_no"], errors="coerce")
                            pairs["player_id"] = pairs["player_id"].fillna("").astype(str)
                            return sorted((int(a), b) for a, b in pairs.dropna(subset=["car_no"]).itertuples(index=False, name=None))
                        if signature(check) != signature(confirm):
                            print(f"double-fetch mismatch race={race_id}; retrying", flush=True)
                            time.sleep(3)
                            continue
                        repaired_entries.extend(confirm_entries)
                        repaired_odds.extend(confirm_odds)
                        success = True
                        break
                except Exception as exc:
                    print(f"targeted repair failed race={race_id}: {exc}", flush=True)
                time.sleep(5)
            if not success:
                return False

        base = base[~base["race_id"].astype(str).isin(race_ids)]
        merged = pd.concat([base, pd.DataFrame(repaired_entries)], ignore_index=True, sort=False)
        if not base_odds.empty:
            base_odds = base_odds[~base_odds["race_id"].astype(str).isin(race_ids)]
            merged_odds = pd.concat([base_odds, pd.DataFrame(repaired_odds)], ignore_index=True, sort=False)
        else:
            merged_odds = pd.DataFrame(repaired_odds)

        # save_today_frames performs one atomic publication of the repaired set.
        fetch_mod.save_today_frames(merged.to_dict("records"), merged_odds.to_dict("records"))
        gate_ok, gate_problems, _ = validate_source_gate()
        if not gate_ok:
            print(f"targeted repair rejected by gate: {gate_problems}", flush=True)
            rollback_last_good()
            return False
        snapshot_last_good()
        return True
    except Exception as exc:
        print(f"targeted repair exception: {exc}", flush=True)
        rollback_last_good()
        return False


def validate_source_gate():
    """Publication gate: source data must pass both rider and whole-race audits."""
    entry_problems, stats = audit_entries()
    coverage_problems = audit_race_coverage()
    identity_problems = []
    try:
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
        if "player_id" in entries.columns:
            for race_id, group in entries.groupby("race_id"):
                ids = group["player_id"].fillna("").astype(str).str.strip()
                if ids.eq("").any() or ids.duplicated().any():
                    identity_problems.append({"type": "source_identity_gate_failed", "race_id": str(race_id)})
    except Exception as exc:
        identity_problems.append({"type": "source_identity_gate_unreadable", "detail": str(exc)})
    problems = entry_problems + coverage_problems + identity_problems
    return not problems, problems, stats


def audit_entries():
    problems = []
    try:
        df = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
    except Exception as exc:
        return [{"type": "entries_unreadable", "detail": str(exc)}], {}

    if df.empty:
        return [{"type": "entries_empty"}], {}

    required = {"race_id", "venue", "race_no", "car_no"}
    missing = sorted(required - set(df.columns))
    if missing:
        return [{"type": "missing_columns", "columns": missing}], {}

    stats = {}
    for race_id, group in df.groupby("race_id"):
        cars = sorted(pd.to_numeric(group["car_no"], errors="coerce").dropna().astype(int).unique().tolist())
        expected_values = pd.to_numeric(group.get("entries_number"), errors="coerce").dropna()
        expected = int(expected_values.max()) if len(expected_values) else 0
        if "declared_entries_number" in group.columns:
            declared_values = pd.to_numeric(group["declared_entries_number"], errors="coerce").dropna()
        else:
            declared_values = pd.Series(dtype=float)
        declared = int(declared_values.max()) if len(declared_values) else expected
        cancelled = set()
        if "cancelled_car_numbers" in group.columns:
            for value in group["cancelled_car_numbers"].dropna().astype(str).unique():
                cancelled.update(int(x) for x in re.findall(r"\d+", value))
        # Older cached snapshots may have active/declaration counts but an empty
        # cancelled_car_numbers field. If the active count is explicit and the
        # exact number of absent car slots equals declared-active, recover those
        # slots as withdrawals instead of falsely reporting missing riders.
        all_declared_cars = set(range(1, declared + 1))
        absent_slots = all_declared_cars - set(cars)
        if not cancelled and expected > 0 and declared > expected and len(cars) == expected and len(absent_slots) == declared - expected:
            cancelled = set(absent_slots)
        expected_cars = all_declared_cars - cancelled
        missing_cars = sorted(expected_cars - set(cars))
        duplicate_cars = sorted(group.loc[group.duplicated("car_no", keep=False), "car_no"].dropna().astype(int).unique().tolist())
        stats[str(race_id)] = {"cars": cars, "count": len(cars), "expected_entries": expected, "declared_entries": declared, "cancelled_cars": sorted(cancelled)}
        if expected <= 0 or declared <= 0:
            problems.append({"type": "missing_expected_field_size", "race_id": str(race_id), "cars": cars})
        elif len(cars) != expected or set(cars) != expected_cars or expected + len(cancelled) != declared:
            problems.append({"type": "missing_riders", "race_id": str(race_id), "expected": expected, "declared": declared, "cancelled_cars": sorted(cancelled), "missing_cars": missing_cars, "cars": cars})
        if duplicate_cars:
            problems.append({"type": "duplicate_riders", "race_id": str(race_id), "cars": duplicate_cars})
        if len(cars) < 5 or len(cars) > 9:
            problems.append({"type": "implausible_rider_count", "race_id": str(race_id), "count": len(cars), "cars": cars})
    return problems, stats


def audit_race_coverage():
    """Detect entire races disappearing from the daily snapshot.

    Coverage must be evaluated against the snapshot's own race date, not the
    wall-clock date. Around JST midnight, the previous day's final snapshot can
    remain valid while datetime.now() has already moved to the next day.
    """
    problems = []
    if not RACE_SCHEDULE_PATH.exists():
        return [{"type": "race_schedule_missing"}]
    try:
        schedule = pd.read_csv(RACE_SCHEDULE_PATH, dtype={"race_id": str})
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})

        audit_date = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
        if "date" in entries.columns:
            entry_dates = (
                entries["date"]
                .dropna()
                .astype(str)
                .str.slice(0, 10)
            )
            if not entry_dates.empty:
                modes = entry_dates.mode()
                audit_date = str(modes.iloc[0] if not modes.empty else entry_dates.iloc[0])

        if "date" in schedule.columns:
            schedule_dates = schedule["date"].astype(str).str.slice(0, 10)
            schedule = schedule[schedule_dates.eq(audit_date)]

        expected_ids = set(schedule["race_id"].dropna().astype(str))
        actual_ids = set(entries["race_id"].dropna().astype(str))

        # If the matching schedule slice is unavailable, do not mislabel every
        # valid race as "unexpected". The dedicated schedule-missing condition
        # is more accurate and avoids midnight false failures.
        if actual_ids and not expected_ids:
            problems.append({"type": "race_schedule_date_missing", "date": audit_date})
            return problems

        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing:
            problems.append({
                "type": "missing_entire_races",
                "expected_races": len(expected_ids),
                "actual_races": len(actual_ids),
                "race_ids": missing,
            })
        if extra:
            problems.append({"type": "unexpected_races", "race_ids": extra})
    except Exception as exc:
        problems.append({"type": "race_coverage_audit_failed", "detail": str(exc)})
    return problems


def audit_budget():
    problems = []
    path = OUTPUT_DIR / "latest_shadow_bets.csv"
    if not path.exists() or path.stat().st_size == 0:
        return problems
    try:
        bets = pd.read_csv(path, dtype={"race_id": str})
        if bets.empty or "stake_yen" not in bets.columns:
            return problems
        for race_id, group in bets.groupby("race_id"):
            total = int(pd.to_numeric(group["stake_yen"], errors="coerce").fillna(0).sum())
            if total != 10000:
                problems.append({"type": "race_budget_mismatch", "race_id": str(race_id), "total_yen": total})
    except Exception as exc:
        problems.append({"type": "budget_audit_failed", "detail": str(exc)})
    return problems


def audit_live_bets():
    """Detect the failure mode where predictions/bet candidates exist but the live bet output is empty."""
    problems = []
    shadow_path = OUTPUT_DIR / "latest_shadow_bets.csv"
    live_path = OUTPUT_DIR / "latest_bets.csv"
    if not shadow_path.exists() or shadow_path.stat().st_size == 0:
        return problems
    try:
        shadow = pd.read_csv(shadow_path, dtype={"race_id": str})
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return problems
    if shadow.empty:
        return problems
    try:
        live = pd.read_csv(live_path, dtype={"race_id": str}) if live_path.exists() else pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        live = pd.DataFrame()
    # The site intentionally renders shadow recommendations even while the
    # external-profit purchase gate is closed. Empty latest_bets must therefore
    # never masquerade as a prediction/data failure.
    if live.empty and "purchase_authorized" in shadow.columns:
        authorized = shadow["purchase_authorized"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
        if authorized:
            problems.append({"type": "authorized_bets_missing", "shadow_rows": int(len(shadow))})
    return problems


def audit_site_output():
    problems = []
    html = OUTPUT_DIR / "index.html"
    if not html.exists() or html.stat().st_size < 1000:
        return [{"type": "site_output_missing_or_too_small"}]
    try:
        content = html.read_text(encoding="utf-8")
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
        # The published HTML must contain every race/car currently present in
        # the validated prediction input. This catches stale pages that are
        # large enough to pass the old file-size-only health check.
        for race_id, group in entries.groupby("race_id"):
            race_id = str(race_id)
            if race_id not in content:
                problems.append({"type": "site_missing_race", "race_id": race_id})
                continue
            cars = sorted(pd.to_numeric(group["car_no"], errors="coerce").dropna().astype(int).unique().tolist())
            # Prediction audit is the authoritative car-level check; here require
            # the race itself to be represented in the actual published artifact.
            if not cars:
                problems.append({"type": "site_race_has_no_valid_cars", "race_id": race_id})
            if "player_name" in group.columns:
                marker = f'<article class="race" id="race-{race_id}"'
                start = content.find(marker)
                end = content.find('<article class="race"', start + len(marker)) if start >= 0 else -1
                race_html = content[start:] if start >= 0 and end < 0 else content[start:end]

                # Parse rider attributes independently of their HTML attribute
                # order. The old exact-substring audit falsely reported a name
                # mismatch whenever another attribute was inserted between
                # data-player-id and data-name even though the rendered rider
                # was correct.
                parser = RiderButtonParser()
                parser.feed(race_html)
                rider_attrs = parser.riders

                for _, rider in group.iterrows():
                    name = str(rider.get("player_name", "")).strip()
                    player_id = str(rider.get("player_id", "")).strip()
                    if not name or name in {"nan", "None"}:
                        continue
                    actual_name = rider_attrs.get(player_id)
                    if actual_name != name:
                        problems.append({
                            "type": "site_player_name_mismatch",
                            "race_id": race_id,
                            "player_id": player_id,
                            "player_name": name,
                            "site_player_name": actual_name,
                        })
    except Exception as exc:
        problems.append({"type": "site_output_unreadable", "detail": str(exc)})
    return problems


def audit_results():
    problems = []
    path = OUTPUT_DIR / "latest_results.json"
    try:
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str})
    except Exception:
        return problems

    now = datetime.now(ZoneInfo("Asia/Tokyo")).timestamp()
    waiting = set()
    overdue = set()

    # Two-stage result timing:
    #   0-10 min after start: normal race/result processing, no warning.
    #   10-20 min after start: official publication wait, not a fault.
    #   20+ min after start: genuine result delay and eligible for escalation.
    # Legacy rows without start_at fall back to close_at with an extra 5 min
    # because betting normally closes before the scheduled start.
    if "start_at" in entries.columns:
        start_at = pd.to_numeric(entries["start_at"], errors="coerce")
        waiting_mask = start_at.notna() & (start_at <= now - 10 * 60) & (start_at > now - 20 * 60)
        overdue_mask = start_at.notna() & (start_at <= now - 20 * 60)

        if "close_at" in entries.columns:
            close_at = pd.to_numeric(entries["close_at"], errors="coerce")
            legacy = start_at.isna() & close_at.notna()
            waiting_mask = waiting_mask | (
                legacy & (close_at <= now - 15 * 60) & (close_at > now - 25 * 60)
            )
            overdue_mask = overdue_mask | (legacy & (close_at <= now - 25 * 60))

        waiting = set(entries.loc[waiting_mask, "race_id"].astype(str))
        overdue = set(entries.loc[overdue_mask, "race_id"].astype(str))
    elif "close_at" in entries.columns:
        close_at = pd.to_numeric(entries["close_at"], errors="coerce")
        waiting = set(entries.loc[
            close_at.notna() & (close_at <= now - 15 * 60) & (close_at > now - 25 * 60),
            "race_id",
        ].astype(str))
        overdue = set(entries.loc[
            close_at.notna() & (close_at <= now - 25 * 60),
            "race_id",
        ].astype(str))

    if not waiting and not overdue:
        return problems

    if not path.exists() or path.stat().st_size == 0:
        if waiting:
            problems.append({"type": "results_waiting", "race_ids": sorted(waiting)})
        if overdue:
            problems.append({"type": "results_missing", "overdue_races": sorted(overdue)})
        return problems

    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        decided = {str(x.get("race_id")) for x in rows if x.get("official_result_available")}
        waiting_missing = sorted(waiting - decided)
        overdue_missing = sorted(overdue - decided)
        if waiting_missing:
            problems.append({"type": "results_waiting", "race_ids": waiting_missing})
        if overdue_missing:
            problems.append({"type": "results_overdue", "race_ids": overdue_missing})
    except Exception as exc:
        problems.append({"type": "results_unreadable", "detail": str(exc)})
    return problems


def audit_strategy_version():
    from betting_logic import STRATEGY_VERSION
    try:
        plans = json.loads((OUTPUT_DIR / "latest_race_strategy.json").read_text(encoding="utf-8"))
        current = bool(plans) and all(p.get("strategy_version") == STRATEGY_VERSION for p in plans)
    except (OSError, ValueError, TypeError, AttributeError):
        current = False
    return [] if current else [{"type": "prediction_strategy_schema_stale", "reason": "strategy_version_changed", "expected_strategy": STRATEGY_VERSION}]


def audit_prediction_outputs():
    problems = audit_strategy_version()
    latest = OUTPUT_DIR / "latest_predictions.csv"
    if not latest.exists() or latest.stat().st_size == 0:
        problems.append({"type": "prediction_missing"})
        return problems
    try:
        pred = pd.read_csv(latest, dtype={"race_id": str, "player_id": str})
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
        # Strategy-schema freshness gate: when live ranking logic changes, an
        # older but structurally valid CSV must not be treated as healthy.
        # Regenerate predictions so the published output actually uses the
        # current Top1 consensus guard.
        required_strategy_columns = {
            "p_win_pre_override",
            "top1_override_applied",
            "top1_override_reason",
            "score_first",
            "score_second",
            "score_third",
            "position_score_source",
            "popularity_source",
        }
        missing_strategy_columns = sorted(required_strategy_columns - set(pred.columns))
        if missing_strategy_columns:
            problems.append({
                "type": "prediction_strategy_schema_stale",
                "missing_columns": missing_strategy_columns,
            })
        if "player_name" in entries.columns:
            if "player_name" not in pred.columns:
                problems.append({"type": "prediction_player_name_missing"})
            else:
                names = pred["player_name"].fillna("").astype(str).str.strip()
                missing_names = int(names.isin({"", "nan", "None"}).sum())
                if missing_names:
                    problems.append({"type": "prediction_player_name_missing", "rows": missing_names})
        for race_id, group in entries.groupby("race_id"):
            source_cars = set(pd.to_numeric(group["car_no"], errors="coerce").dropna().astype(int))
            p = pred[pred["race_id"].astype(str).eq(str(race_id))]
            pred_cars = set(pd.to_numeric(p["car_no"], errors="coerce").dropna().astype(int)) if "car_no" in p else set()
            missing = sorted(source_cars - pred_cars)
            if missing:
                problems.append({"type": "prediction_missing_riders", "race_id": str(race_id), "missing_cars": missing})
    except Exception as exc:
        problems.append({"type": "prediction_unreadable", "detail": str(exc)})
    return problems


def audit_identity_and_prediction_quality():
    problems = []
    try:
        entries = pd.read_csv(TODAY_CSV, dtype={"race_id": str, "player_id": str})
    except Exception:
        return problems

    # A car must map to exactly one rider and a rider must not occupy multiple
    # cars in the same race. This catches malformed snapshots that still have
    # the correct row count.
    if "player_id" in entries.columns:
        for race_id, group in entries.groupby("race_id"):
            valid = group.dropna(subset=["player_id"]).copy()
            if valid["player_id"].astype(str).duplicated().any():
                dup = sorted(valid.loc[valid["player_id"].astype(str).duplicated(keep=False), "player_id"].astype(str).unique().tolist())
                problems.append({"type": "duplicate_player_identity", "race_id": str(race_id), "player_ids": dup})
            if valid["player_id"].astype(str).isin({"", "nan", "None"}).any() or len(valid) != len(group):
                problems.append({"type": "missing_player_identity", "race_id": str(race_id)})

    latest = OUTPUT_DIR / "latest_predictions.csv"
    if latest.exists() and latest.stat().st_size:
        try:
            pred = pd.read_csv(latest, dtype={"race_id": str, "player_id": str})
            if "p_win" in pred.columns:
                pred["p_win"] = pd.to_numeric(pred["p_win"], errors="coerce")
                for race_id, group in pred.groupby("race_id"):
                    probs = group["p_win"]
                    if probs.isna().any() or (~probs.between(0, 1)).any():
                        problems.append({"type": "invalid_prediction_probability", "race_id": str(race_id)})
                    total = float(probs.sum(skipna=True))
                    if len(group) and abs(total - 1.0) > 0.02:
                        problems.append({"type": "prediction_probability_not_normalized", "race_id": str(race_id), "sum": total})
        except Exception as exc:
            problems.append({"type": "prediction_quality_audit_failed", "detail": str(exc)})
    return problems


def audit_freshness():
    problems = []
    now = datetime.now(ZoneInfo("Asia/Tokyo")).timestamp()
    # During active racing, stale prediction artifacts can look structurally
    # perfect while actually serving old information.
    latest = OUTPUT_DIR / "latest_predictions.csv"
    try:
        schedule = pd.read_csv(RACE_SCHEDULE_PATH, dtype={"race_id": str})
        close_at = pd.to_numeric(schedule.get("close_at"), errors="coerce")
        active = close_at.notna() & close_at.gt(now) & close_at.le(now + 60 * 60)
        if active.any() and latest.exists():
            age = now - latest.stat().st_mtime
            if age > 45 * 60:
                problems.append({"type": "stale_predictions", "age_seconds": int(age)})
        elif active.any() and not latest.exists():
            problems.append({"type": "prediction_missing"})
    except Exception:
        pass
    return problems


def append_incident_history(problems, action=None):
    if not problems:
        return
    record = {
        "at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "problem_types": sorted({str(p.get("type")) for p in problems}),
        "problems": problems,
        "repair_action": action,
    }
    with INCIDENT_HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def incident_recurrence_counts(problems, lookback=200):
    """Count only the current recurrence streak for each problem/race key.

    Old incidents must not permanently poison escalation. A streak resets when
    the same key has been absent for RECURRENCE_RESET_GAP_MINUTES, and records
    older than RECURRENCE_WINDOW_MINUTES are ignored entirely.
    """
    keys = set()
    for p in problems or []:
        kind = str(p.get("type", "unknown"))
        race_ids = p.get("race_ids") or ([p.get("race_id")] if p.get("race_id") else ["*"])
        for race_id in race_ids:
            keys.add((kind, str(race_id)))

    counts = {key: 0 for key in keys}
    if not keys or not INCIDENT_HISTORY_PATH.exists():
        return counts

    occurrences = {key: [] for key in keys}
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    window_seconds = RECURRENCE_WINDOW_MINUTES * 60
    reset_gap_seconds = RECURRENCE_RESET_GAP_MINUTES * 60

    try:
        lines = INCIDENT_HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-lookback:]
        for line in lines:
            try:
                rec = json.loads(line)
                at = datetime.fromisoformat(str(rec.get("at_jst", "")))
                if at.tzinfo is None:
                    at = at.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
            except Exception:
                continue

            age_seconds = (now - at.astimezone(ZoneInfo("Asia/Tokyo"))).total_seconds()
            if age_seconds < 0 or age_seconds > window_seconds:
                continue

            for old in rec.get("problems", []) or []:
                kind = str(old.get("type", "unknown"))
                race_ids = old.get("race_ids") or ([old.get("race_id")] if old.get("race_id") else ["*"])
                for race_id in race_ids:
                    key = (kind, str(race_id))
                    if key in occurrences:
                        occurrences[key].append(at)

        for key, times in occurrences.items():
            if not times:
                continue
            times = sorted(times, reverse=True)
            streak = 1
            previous = times[0]
            for at in times[1:]:
                gap_seconds = (previous - at).total_seconds()
                if gap_seconds > reset_gap_seconds:
                    break
                streak += 1
                previous = at
            counts[key] = streak
    except OSError:
        pass

    return counts


def escalation_level(problems):
    """Escalate repeated failures instead of repeating the same repair forever."""
    counts = incident_recurrence_counts(problems)
    worst = max(counts.values(), default=0)
    if worst >= 5:
        return 3, counts
    if worst >= 3:
        return 2, counts
    if worst >= 1:
        return 1, counts
    return 0, counts


def summarize_health(problems, stats):
    kinds = [str(p.get("type")) for p in problems]
    severity = "critical" if any(k in {
        "missing_riders", "missing_entire_races", "entries_unreadable",
        "entries_empty", "prediction_missing", "prediction_missing_riders",
    } for k in kinds) else ("warning" if kinds else "ok")
    expected_riders = sum(int(v.get("expected_entries", 0) or 0) for v in stats.values())
    actual_riders = sum(int(v.get("count", 0) or 0) for v in stats.values())
    return {
        "severity": severity,
        "problem_types": sorted(set(kinds)),
        "race_count": len(stats),
        "expected_riders": expected_riders,
        "actual_riders": actual_riders,
        "rider_integrity_ok": bool(stats) and expected_riders == actual_riders and not any(
            k in {"missing_riders", "missing_entire_races", "duplicate_riders"} for k in kinds
        ),
    }


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def repair(problems=None):
    """Choose the smallest safe repair for the detected failure class."""
    problems = problems or []
    kinds = {str(p.get("type")) for p in problems}

    data_kinds = {
        "entries_unreadable", "entries_empty", "missing_columns",
        "missing_expected_field_size", "missing_riders", "duplicate_riders",
        "implausible_rider_count", "missing_entire_races", "unexpected_races",
        "race_schedule_missing", "race_schedule_date_missing", "race_coverage_audit_failed",
        "duplicate_player_identity", "missing_player_identity",
    }
    prediction_kinds = {
        "prediction_missing", "prediction_missing_riders", "prediction_player_name_missing", "prediction_unreadable",
        "prediction_strategy_schema_stale",
        "invalid_prediction_probability", "prediction_probability_not_normalized",
        "prediction_quality_audit_failed", "stale_predictions",
    }
    result_kinds = {"results_missing", "results_overdue", "results_unreadable"}
    site_kinds = {"site_output_missing_or_too_small", "site_missing_race", "site_race_has_no_valid_cars", "site_player_name_mismatch", "site_output_unreadable"}
    bet_kinds = {"race_budget_mismatch", "budget_audit_failed", "authorized_bets_missing"}

    actions = []
    level, recurrence = escalation_level(problems)
    if level:
        print(f"incident recurrence escalation level={level} counts={recurrence}", flush=True)

    # Any source-data integrity problem gets the strongest repair: an atomic
    # full-day rebuild. Never mask it with a near-close-only snapshot.
    if kinds & data_kinds:
        # Keep the last known-good source before touching production input.
        pre_ok, _, _ = validate_source_gate()
        if pre_ok:
            snapshot_last_good()

        targeted_types = {
            "missing_riders", "missing_entire_races", "duplicate_riders",
            "missing_player_identity", "duplicate_player_identity",
        }
        cache_ok = bool(kinds & targeted_types) and restore_races_from_cache(problems)
        # If cache cannot fully restore the day, try focused network recovery.
        # Level 2+ skips repeating a network strategy that already failed.
        targeted_ok = cache_ok or (level < 2 and bool(kinds & targeted_types) and repair_missing_races_in_place(problems))
        if targeted_ok:
            actions.append("race_cache_or_targeted_repair_gated")
        else:
            # Escalate after focused recovery fails or recurrence history says
            # the focused strategy is no longer effective.
            if run([sys.executable, "src/fetch_today_entries.py"]) != 0:
                rollback_last_good()
                return False, "targeted_and_full_day_rebuild_failed_rolled_back"
            gate_ok, gate_problems, _ = validate_source_gate()
            if not gate_ok:
                rolled_back = rollback_last_good()
                append_incident_history(gate_problems, "publication_gate_rejected")
                return False, "publication_gate_rejected_rolled_back" if rolled_back else "publication_gate_rejected_no_backup"
            snapshot_last_good()
            actions.append("full_day_data_rebuild_gated")

    # Prediction/site/bet faults are regenerated from the already validated
    # source snapshot. A successful command is not enough; the next audit loop
    # must independently prove every race/rider is present.
    if (kinds & (data_kinds | prediction_kinds | site_kinds | bet_kinds)) or not (OUTPUT_DIR / "latest_predictions.csv").exists():
        if run([sys.executable, "src/predict.py"]) != 0:
            return False, "predict_regeneration_failed"
        actions.append("prediction_regeneration")

    # Result failures use a separate escalation ladder. Repeating the same
    # settlement fetch forever is wasteful and can turn a provider publication
    # delay into a false system failure.
    result_fault = bool(kinds & result_kinds)
    result_overdue_only = kinds == {"results_overdue"}

    if result_overdue_only and level >= 3:
        # Circuit breaker: after repeated confirmed retries, stop hammering the
        # provider. The manager remains alive and the next scheduled cycle (and
        # quick-results workflow) will try again with fresh upstream state.
        actions.append("results_deferred_until_next_cycle")
    else:
        if result_fault and not (kinds & data_kinds) and level >= 2:
            # Escalation path: refresh race/source metadata first, validate it,
            # then retry settlement. This is deliberately stronger than simply
            # replaying settle_results.py against a stale source snapshot.
            pre_ok, _, _ = validate_source_gate()
            if pre_ok:
                snapshot_last_good()
            if run([sys.executable, "src/fetch_today_entries.py"]) != 0:
                return False, "results_source_refresh_failed"
            gate_ok, gate_problems, _ = validate_source_gate()
            if not gate_ok:
                rolled_back = rollback_last_good()
                append_incident_history(gate_problems, "results_source_gate_rejected")
                return False, "results_source_gate_rejected_rolled_back" if rolled_back else "results_source_gate_rejected_no_backup"
            snapshot_last_good()
            actions.append("results_source_refresh_gated")

        if kinds & (data_kinds | result_kinds):
            if run([sys.executable, "src/settle_results.py"]) != 0:
                return False, "settlement_refresh_failed"
            actions.append("settlement_refresh")

    if not actions:
        # Unknown failures get a conservative full integrity rebuild rather than
        # being silently ignored.
        if run([sys.executable, "src/fetch_today_entries.py"]) != 0:
            return False, "conservative_full_rebuild_fetch_failed"
        if run([sys.executable, "src/predict.py"]) != 0:
            return False, "conservative_full_rebuild_predict_failed"
        actions.extend(["conservative_full_day_rebuild", "prediction_regeneration"])
    return True, "+".join(actions)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    repaired = False
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        entry_problems, stats = audit_entries()
        prediction_problems = audit_prediction_outputs()
        problems = (
            entry_problems
            + audit_race_coverage()
            + prediction_problems
            + audit_identity_and_prediction_quality()
            + audit_freshness()
            + audit_budget()
            + audit_live_bets()
            + audit_site_output()
            + audit_results()
        )
        history.append({
            "attempt": attempt,
            "problems": problems,
            "health": summarize_health(problems, stats),
        })
        if not problems:
            # Only audited source snapshots are eligible to become rollback
            # checkpoints. This prevents a corrupt fetch from replacing backup.
            snapshot_last_good()
            status = "healthy" if not repaired else "repaired"
            break

        problem_types = {str(p.get("type")) for p in problems}
        if problem_types == {"results_waiting"}:
            # 10-20 minutes after race start is a normal official-publication
            # window. Do not burn repair attempts or refetch the whole day.
            status = "waiting_results"
            break

        if attempt >= MAX_REPAIR_ATTEMPTS:
            # Result-only delays must not take the whole site down. True source,
            # prediction, site or identity faults still fail hard.
            if problem_types and problem_types.issubset({"results_waiting", "results_overdue"}):
                status = "waiting_results"
            else:
                status = "unhealthy"
            break
        ok, action = repair(problems)
        append_incident_history(problems, action)
        repaired = repaired or ok
        history[-1]["repair_action"] = action
        if not ok:
            time.sleep(RETRY_SECONDS)
            continue
        time.sleep(RETRY_SECONDS)

    payload = {
        "checked_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "status": status,
        "repaired": repaired,
        "attempts": history,
        "race_stats": stats,
        "health": summarize_health(history[-1]["problems"] if history else [], stats),
        "recurrence": {
            "level": escalation_level(history[-1]["problems"] if history else [])[0],
            "counts": {
                f"{kind}:{race_id}": count
                for (kind, race_id), count in escalation_level(history[-1]["problems"] if history else [])[1].items()
            },
        },
    }
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status == "unhealthy":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
