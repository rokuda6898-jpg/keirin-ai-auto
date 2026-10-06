"""Reject unusable observed trifecta quotes without fabricating legacy timestamps."""
import numpy as np
import pandas as pd

MAX_QUOTE_AGE_SECONDS=300


def validate_quotes(odds, now_epoch, close_epoch=None):
    frame=odds.copy()
    if 'odds_captured_at_jst' not in frame:
        return frame
    trifecta=frame.bet_type.eq('trifecta')
    captured=pd.to_datetime(frame.odds_captured_at_jst,utc=True,errors='coerce')
    epoch=captured.map(lambda value:value.timestamp() if pd.notna(value) else np.nan)
    observed=trifecta & frame.odds_captured_at_jst.notna()
    invalid=observed & (captured.isna() | epoch.gt(now_epoch) | (now_epoch-epoch).gt(MAX_QUOTE_AGE_SECONDS))
    if close_epoch is not None and np.isfinite(close_epoch):invalid |= observed & epoch.ge(close_epoch)
    if 'odds_verification_status' in frame:
        invalid |= trifecta & frame.odds_verification_status.eq('excluded')
    frame.loc[invalid,'odds_used']=np.nan
    frame.loc[invalid,'odds_verification_status']='excluded_time_or_source'
    return frame
