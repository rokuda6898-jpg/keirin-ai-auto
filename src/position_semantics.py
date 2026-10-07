"""Exact-place challenger from cumulative outputs; never automatic promotion."""
import numpy as np
import pandas as pd


def exact_place_challenger(frame, cumulative_second, cumulative_third):
    out = frame.copy()
    win = pd.to_numeric(out.get("p_core", out["p_win"]), errors="coerce")
    second = np.asarray(cumulative_second, dtype=float)
    third = np.asarray(cumulative_third, dtype=float)
    if len(second) != len(out) or len(third) != len(out):
        raise ValueError("cumulative place output length mismatch")
    if not np.isfinite(second).all() or not np.isfinite(third).all() or not np.isfinite(win).all():
        raise ValueError("nonfinite cumulative place output")
    # Projection handles independently estimated cumulative probabilities.
    # This is a comparison proposal, not evidence of calibration.
    top2 = np.maximum(win.to_numpy(), np.clip(second, 0, 1))
    top3 = np.maximum(top2, np.clip(third, 0, 1))
    for name, values in [("second", top2 - win.to_numpy()), ("third", top3 - top2)]:
        values = pd.Series(values, index=out.index)
        mass = values.groupby(out.race_id).transform("sum")
        out[f"p_{name}_exact_challenger"] = values.div(mass.where(mass.gt(0)))
    return out
