"""Reject stale/partial NEXUS deployments before GitHub Pages upload.

Checks model code fingerprints, race coverage, and seven-department outputs.
This verification never generates forecasts or authorizes real purchases.
"""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "nexus-departments-v2"
SOURCES = (
    "src/predict.py",
    "src/annual_knowledge.py",
    "src/department_coverage.py",
    "src/mark_performance.py",
    "src/site_ui.py",
    "src/betting_logic.py",
    "src/release_guard.py",
)
DEPARTMENTS = {
    "data_department", "pace_department", "line_department", "risk_department",
    "prediction_department", "strategist_department", "high_payout_department",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def race_ids(path):
    if not path.is_file():
        raise ValueError(f"missing race data: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {str(row.get("race_id", "")).strip() for row in rows}
    ids.discard("")
    if not rows or not ids:
        raise ValueError(f"empty race data: {path}")
    return ids


def write_release_manifest(root=ROOT):
    root = Path(root)
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA,
        "generated_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "sources": {name: digest(root / name) for name in SOURCES},
        "races": sorted(race_ids(root / "data/raw/today_entries.csv")),
        "predicted_races": sorted(race_ids(out / "latest_predictions.csv")),
    }
    tmp = out / "release_manifest.json.tmp"
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out / "release_manifest.json")
    return report


class ReleaseHtmlParser(HTMLParser):
    """Check the parsed DOM, not the attribute order of the HTML serializer."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.schema = None
        self.races = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta" and values.get("name") == "nexus-render-schema":
            self.schema = values.get("content")
        elif tag == "article" and str(values.get("id", "")).startswith("race-"):
            self.races.add(str(values["id"])[5:])


def validate_release(root=ROOT, now=None):
    root = Path(root)
    out = root / "outputs"
    now = now or datetime.now(ZoneInfo("Asia/Tokyo"))
    errors = []
    try:
        manifest = json.loads((out / "release_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return [f"manifest missing/unreadable: {exc}"]
    if manifest.get("schema") != SCHEMA:
        errors.append("release manifest schema mismatch")
    for name in SOURCES:
        path = root / name
        if not path.is_file() or manifest.get("sources", {}).get(name) != digest(path):
            errors.append(f"code changed since generation: {name}")
    try:
        actual = race_ids(root / "data/raw/today_entries.csv")
        predicted = race_ids(out / "latest_predictions.csv")
        if actual != predicted or actual != set(manifest.get("races", [])):
            errors.append("source/prediction/manifest race set mismatch")
    except ValueError as exc:
        errors.append(str(exc))
        actual = set()
    try:
        page = (out / "index.html").read_text(encoding="utf-8")
        parser = ReleaseHtmlParser()
        parser.feed(page)
        if parser.schema != SCHEMA:
            errors.append("NEXUS HTML schema is stale")
        for race in actual:
            if race not in parser.races:
                errors.append(f"race absent from site HTML: {race}")
    except OSError:
        errors.append("index.html missing")
    try:
        report = json.loads((out / "company/all_department_predictions.json").read_text(encoding="utf-8"))
        per_race = defaultdict(list)
        for entry in report.get("predictions", []):
            per_race[str(entry.get("race_id"))].append(entry)
        if set(per_race) != actual:
            errors.append("department matrix race set mismatch")
        for race in actual:
            entries = per_race.get(race, [])
            if len(entries) != 7 or {e.get("department") for e in entries} != DEPARTMENTS:
                errors.append(f"missing/duplicate department rows: {race}")
                continue
            closes = [float(e["close_at"]) for e in entries if e.get("close_at") is not None]
            if closes and min(closes) > now.timestamp() + 300:
                if not all(e.get("forecast_available") for e in entries):
                    errors.append(f"forecast not produced before close: {race}")
        if report.get("coverage_status") != "complete":
            errors.append("department matrix marked incomplete")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"department matrix missing/corrupt: {exc}")
    for name in ("all_department_predictions.html", "all_department_results.html"):
        if not (out / "company" / name).is_file():
            errors.append(f"department UI page missing: {name}")
    return errors


if __name__ == "__main__":
    import sys
    failures = validate_release()
    if failures:
        print("PUBLICATION BLOCKED:\n" + "\n".join("- " + reason for reason in failures))
        sys.exit(2)
    print("PUBLICATION VALIDATED: code fingerprint, race coverage, and all seven departments")
