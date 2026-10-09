"""Carry immutable pre-close evidence and validated feature cache across git resets."""
import argparse
import hashlib
import shutil
from pathlib import Path

from common import ROOT

LEDGERS = ["outputs/company/ticket_return_snapshots.jsonl",
           "outputs/company/company_decision_ledger.jsonl",
           "outputs/company/all_department_prediction_ledger.jsonl",
           "outputs/company/high_payout_axis_shadow_ledger.jsonl",
           "outputs/company/market_first_shadow_ledger.jsonl",
           "outputs/company/annual_department_prediction_ledger.jsonl",
           "outputs/company/annual_position_experiment_ledger.jsonl",
           "outputs/company/annual_position_experiment_outcomes.jsonl",
           "outputs/company/annual_position_v2_ledger.jsonl",
           "outputs/company/annual_position_v2_attempts.jsonl",
           "outputs/company/annual_position_v2_outcomes.jsonl",
           "outputs/company/annual_equation_ledger.jsonl",
           "outputs/company/annual_equation_models.jsonl",
           "outputs/company/annual_equation_attempts.jsonl",
           "outputs/company/annual_equation_observed_paths.jsonl",
           "outputs/company/annual_fusion_inputs.jsonl",
           "outputs/company/annual_fusion_models.jsonl",
           "outputs/company/annual_fusion_ledger.jsonl",
           "outputs/company/annual_fusion_attempts.jsonl",
           "outputs/company/annual_fusion_confirmations.jsonl",
           "outputs/company/annual_strategist_ledger.jsonl"]
OBSERVATIONS = "outputs/company/rider_official_observations.csv"
CACHES = ["data/raw/live_feature_base.csv", "data/raw/live_feature_base.meta.json",
          "outputs/company/annual_rider_knowledge.json"]


def save(directory, root=ROOT):
    directory = Path(directory)
    for name in LEDGERS + CACHES + [OBSERVATIONS]:
        path = root / name
        if path.exists():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def restore(directory, root=ROOT):
    directory = Path(directory)
    for name in LEDGERS:
        source = directory / name
        if not source.exists():
            continue
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
        seen = {hashlib.sha256(line.encode()).hexdigest() for line in existing}
        for line in source.read_text(encoding="utf-8").splitlines():
            key = hashlib.sha256(line.encode()).hexdigest()
            if line and key not in seen:
                existing.append(line)
                seen.add(key)
        target.write_text("\n".join(existing) + "\n", encoding="utf-8")
    observations = directory / OBSERVATIONS
    if observations.exists():
        import pandas as pd
        target = root / OBSERVATIONS
        saved = pd.read_csv(observations, dtype={"race_id": str, "player_id": str}).set_index(["race_id", "player_id"])
        if target.exists():
            current = pd.read_csv(target, dtype={"race_id": str, "player_id": str}).set_index(["race_id", "player_id"])
            saved = current.combine_first(saved)
        target.parent.mkdir(parents=True, exist_ok=True)
        saved.reset_index().to_csv(target, index=False)
    for name in CACHES:
        source = directory / name
        if source.exists():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["save", "restore"])
    parser.add_argument("directory")
    args = parser.parse_args()
    (save if args.mode == "save" else restore)(args.directory)
