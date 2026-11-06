"""
features.py
Turns each customer's long row of daily usage into a small set of
meaningful numbers (features) that a model can learn from.

Used by: notebooks/04_features.ipynb  and later by the Streamlit app.
"""

import numpy as np
import pandas as pd


def safe_ratio(top, bottom, min_bottom=0.1, max_value=10.0):
    """Divide top by bottom, but avoid crazy values when bottom is ~0.
    If bottom is too small, return 1.0 (meaning 'no change')."""
    top = np.asarray(top, dtype=float)
    bottom = np.asarray(bottom, dtype=float)
    ratio = np.where(bottom > min_bottom, top / np.maximum(bottom, 1e-9), 1.0)
    return np.clip(ratio, 0, max_value)


def build_features(usage: pd.DataFrame) -> pd.DataFrame:
    """
    usage: rows = customers, columns = dates (datetime), values = kWh per day
    returns: rows = customers, columns = features
    """
    X = usage.to_numpy(dtype=float)
    n_customers, n_days = X.shape
    dates = pd.DatetimeIndex(usage.columns)
    f = pd.DataFrame(index=usage.index)

    # ---------- 1. Basic statistics ----------
    f["mean"] = X.mean(axis=1)
    f["median"] = np.median(X, axis=1)
    f["std"] = X.std(axis=1)
    f["max"] = X.max(axis=1)
    f["cv"] = safe_ratio(f["std"], f["mean"], min_bottom=0.01, max_value=50)

    # ---------- 2. When did the meter start? ----------
    has_usage = (X > 0).any(axis=1)
    start_day = np.where(has_usage, (X > 0).argmax(axis=1), n_days)
    f["start_day"] = start_day
    f["active_days"] = n_days - start_day

    # ---------- 3. Zeros AFTER the meter started (on/off pattern) ----------
    day_idx = np.arange(n_days)
    after_start = day_idx[None, :] >= start_day[:, None]
    Z = (X == 0) & after_start                      # suspicious zeros only

    f["zero_days_after_start"] = Z.sum(axis=1)
    f["zero_pct_after_start"] = safe_ratio(
        f["zero_days_after_start"], f["active_days"], min_bottom=0.5, max_value=1
    )

    # a "streak" starts where today is zero and yesterday was not
    prev = np.zeros_like(Z)
    prev[:, 1:] = Z[:, :-1]
    f["zero_streaks"] = (Z & ~prev).sum(axis=1)

    # longest run of zeros in a row
    c = np.cumsum(Z, axis=1)
    reset = np.maximum.accumulate(np.where(~Z, c, 0), axis=1)
    f["longest_zero_streak"] = (c - reset).max(axis=1)

    # ---------- 4. Sudden drops (month to month) ----------
    monthly = usage.T.groupby(dates.to_period("M")).mean().T.to_numpy()
    before, after = monthly[:, :-1], monthly[:, 1:]
    drop = np.where(before > 0.5, (before - after) / np.maximum(before, 1e-9), 0)
    f["max_month_drop"] = np.clip(drop, 0, 1).max(axis=1)        # 0.7 = 70% drop
    f["big_drops_count"] = (drop > 0.5).sum(axis=1)              # months with >50% fall
    f["monthly_std"] = monthly.std(axis=1)

    # ---------- 5. Long-term change ----------
    last_6m = X[:, dates >= dates.max() - pd.DateOffset(months=6)].mean(axis=1)
    f["recent_vs_overall"] = safe_ratio(last_6m, f["mean"])

    jan_oct = dates.month <= 10
    y2015 = X[:, (dates.year == 2015) & jan_oct].mean(axis=1)
    y2016 = X[:, (dates.year == 2016) & jan_oct].mean(axis=1)
    f["change_2016_vs_2015"] = safe_ratio(y2016, y2015)

    # ---------- 6. Seasonal and weekly habits ----------
    summer = X[:, dates.month.isin([6, 7, 8])].mean(axis=1)
    winter = X[:, dates.month.isin([12, 1, 2])].mean(axis=1)
    f["summer_vs_winter"] = safe_ratio(summer, winter)

    weekend = X[:, dates.dayofweek >= 5].mean(axis=1)
    weekday = X[:, dates.dayofweek < 5].mean(axis=1)
    f["weekend_vs_weekday"] = safe_ratio(weekend, weekday)

    return f.round(4)