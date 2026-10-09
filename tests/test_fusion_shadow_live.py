import pandas as pd
import pytest

from fusion_shadow_live import restore_race_metadata


def test_restore_close_time_and_race_number_after_model_projection():
    entries = pd.DataFrame({
        'race_id': ['race-a'] * 3,
        'close_at': [1_800_000_000] * 3,
        'race_no': [9] * 3,
    })
    projected = pd.DataFrame({
        'race_id': ['race-a'] * 3,
        'car_no': [1, 2, 3],
    })

    restored = restore_race_metadata(projected, entries)

    assert restored.close_at.tolist() == [1_800_000_000] * 3
    assert restored.race_no.tolist() == [9] * 3
    assert restored.car_no.tolist() == [1, 2, 3]


def test_reject_inconsistent_race_level_metadata():
    entries = pd.DataFrame({
        'race_id': ['race-a', 'race-a'],
        'close_at': [1_800_000_000, 1_800_000_001],
        'race_no': [9, 9],
    })
    projected = pd.DataFrame({'race_id': ['race-a'], 'car_no': [1]})

    with pytest.raises(ValueError, match='race-level'):
        restore_race_metadata(projected, entries)

