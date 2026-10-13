"""Verified physical bank features for nationwide race-method analysis.

Reference-only observations of geometry, never winner labels. Venue-specific
renovation boundaries are respected; unknown fields stay NaN.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from venue_time_learning import EXPECTED_VENUES, venue_name

ROOT = Path(__file__).resolve().parents[1]
BANK_FILE = ROOT / "data" / "reference" / "bank_geometry_43.csv"
ALIASES = {"松坂":"松阪", "伊東温泉":"伊東", "伊東温泉競輪":"伊東"}
PHYSICAL = ("lap_m", "straight_m", "cant_deg", "home_width_m", "back_width_m", "center_width_m")
PREFIXED = tuple("bank_"+c for c in PHYSICAL)
VALID_LAPS = {250, 333, 400, 500}


def canonical_name(value):
    name = venue_name(value)
    return ALIASES.get(name, name)


def angle_to_degrees(value):
    if pd.isna(value) or str(value).strip() == "":
        return float("nan")
    try:
        d, m, s = (int(z) for z in str(value).split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed bank angle: {value!r}") from exc
    if not (0 <= d < 90 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"Impossible bank angle: {value!r}")
    return d + m / 60 + s / 3600


def read_bank_geometry(path=BANK_FILE):
    frame = pd.read_csv(path)
    mandatory = {"venue", "lap_m", "straight_m", "cant_dms",
                 "home_width_m", "back_width_m", "center_width_m",
                 "race_system", "configuration_from", "verification_url"}
    if not mandatory.issubset(frame.columns):
        raise ValueError(f"Missing physical geometry columns: {sorted(mandatory - set(frame.columns))}")
    frame["venue"] = frame.venue.map(canonical_name)
    expected = set(EXPECTED_VENUES)
    if len(frame) != 43 or set(frame.venue) != expected or frame.venue.duplicated().any():
        raise ValueError("Geometry must cover each of the 43 listed velodromes exactly once")
    for col in ("lap_m", "straight_m", "home_width_m", "back_width_m", "center_width_m"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["cant_deg"] = frame.cant_dms.map(angle_to_degrees)
    if not frame.lap_m.isin(VALID_LAPS).all():
        raise ValueError("Unexpected velodrome circumference")
    expected_laps = {250:1,333:7,400:32,500:3}
    if frame.lap_m.value_counts().to_dict() != expected_laps:
        raise ValueError("Unverified track-length distribution")
    if frame.set_index("venue").loc["富山", "lap_m"] != 333:
        raise ValueError("Corrected Toyama 333m must be retained")
    if frame.set_index("venue").loc["平塚", "lap_m"] != 400:
        raise ValueError("Corrected Hiratsuka 400m must be retained")
    geometry = frame[frame.race_system.eq("KEIRIN")]
    for col in ("straight_m", "cant_deg", "home_width_m", "back_width_m", "center_width_m"):
        if geometry[col].isna().any():
            raise ValueError(f"Unexpected missing standard-bank geometry: {col}")
    if not geometry.straight_m.between(30,80).all() or not geometry.cant_deg.between(20,40).all():
        raise ValueError("Geometry outside physical plausibility bounds")
    # Surface exact source facts, but never treat them as observed finish results.
    frame["straight_fraction"] = frame.straight_m / frame.lap_m
    frame["width_drop_m"] = frame.home_width_m - frame.center_width_m
    frame["configuration_from"] = pd.to_datetime(frame.configuration_from, errors="coerce")
    return frame


def add_bank_geometry(races, geometry=None, *, date_col="date"):
    if "venue" not in races:
        raise ValueError("Race rows require venue")
    if "bank_lap_m" in races:
        raise ValueError("Bank geometry has already been joined; prevent duplicate-source ambiguity")
    geometry = read_bank_geometry() if geometry is None else geometry.copy()
    rows = races.copy()
    rows["_bank_venue_name"] = rows.venue.map(canonical_name)
    reference = geometry.rename(columns={"venue":"_bank_venue_name"})
    for c in ("lap_m","straight_m","cant_deg","home_width_m","back_width_m",
              "center_width_m","straight_fraction","width_drop_m","race_system",
              "configuration_from","correction_note"):
        if c in reference:
            reference = reference.rename(columns={c:"bank_"+c})
    selected = ["_bank_venue_name"] + [c for c in reference.columns if c.startswith("bank_")]
    rows = rows.merge(reference[selected], how="left", on="_bank_venue_name", validate="many_to_one")
    rows.drop(columns=["_bank_venue_name"], inplace=True)
    # Do not project the current Kumamoto 400m onto the pre-2024 old 500m course.
    if date_col in rows:
        date = pd.to_datetime(rows[date_col],errors="coerce")
        effective = rows["bank_configuration_from"]
        obsolete = effective.notna() & date.lt(effective)
        numeric = [c for c in rows if c.startswith("bank_") and c not in (
            "bank_race_system","bank_configuration_from","bank_correction_note")]
        for col in numeric:
            rows.loc[obsolete,col] = np.nan
        rows["bank_geometry_usable"] = rows["bank_lap_m"].notna() & ~obsolete
        rows["bank_historically_obsolete"] = obsolete
    else:
        rows["bank_geometry_usable"] = rows["bank_lap_m"].notna()
        rows["bank_historically_obsolete"] = False
    # PIST6 is not a KEIRIN comparable observation.
    rows.loc[rows.bank_race_system.ne("KEIRIN"),"bank_geometry_usable"] = False
    return rows
