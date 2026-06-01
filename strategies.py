"""
Strategy detectors  —  one pluggable scanner per Andrew Aziz setup.
===================================================================

Each detector is a function:

    detect(intraday, daily, cfg) -> dict | None

It returns the strategy-specific columns when the pattern is present on the
latest bar, or None when it isn't. The shared engine in top_reversal_scanner.py
handles everything else (universal "stocks in play" filters, data fetching, the
funnel, float-on-survivors, sorting). To add a strategy, write one function and
register it in STRATEGIES at the bottom — nothing else changes.

PRECISE vs HEURISTIC
--------------------
Some setups are purely computational and faithful to the definition:
  * Reversal   — count consecutive candles            (precise)
  * ORB        — break of the opening range           (precise)
  * VWAP       — price vs intraday VWAP                (precise)
Others are visual chart patterns that can only be *approximated* by a screener.
These are good at narrowing the list, but you must confirm the actual pattern on
the chart yourself:
  * ABCD       — pole / pullback / new-high heuristic  (approximate)
  * Bull Flag  — surge then tight consolidation         (approximate)
"""

from __future__ import annotations

import pandas as pd

# detect_reversal and bar_minutes live in the engine module so it stays
# standalone; we import them here and include reversal in the registry.
from top_reversal_scanner import ScanConfig, detect_reversal, bar_minutes


# --------------------------------------------------------------------------- #
# ORB  —  Opening Range Breakout  (Aziz beginner pick; precise)
# --------------------------------------------------------------------------- #
def detect_orb(intraday: pd.DataFrame, daily: pd.DataFrame, cfg: ScanConfig):
    """Price has broken the opening range (first cfg.orb_minutes of the session).

    direction 'up'   -> current price above the opening-range HIGH (long ORB)
    direction 'down' -> current price below the opening-range LOW  (short ORB)
    """
    if intraday.empty:
        return None
    n_open = max(1, cfg.orb_minutes // bar_minutes(cfg.bar_interval))
    if len(intraday) <= n_open:        # opening range not complete yet
        return None

    opening = intraday.iloc[:n_open]
    or_high = float(opening["High"].max())
    or_low = float(opening["Low"].min())
    price = float(intraday["Close"].iloc[-1])

    if cfg.direction == "up":
        if price <= or_high:
            return None
        return {"OR_High": round(or_high, 2),
                "Break%": round((price / or_high - 1) * 100, 2)}
    else:
        if price >= or_low:
            return None
        return {"OR_Low": round(or_low, 2),
                "Break%": round((1 - price / or_low) * 100, 2)}


# --------------------------------------------------------------------------- #
# ABCD Pattern  (Aziz beginner pick; HEURISTIC — confirm on the chart)
# --------------------------------------------------------------------------- #
def detect_abcd(intraday: pd.DataFrame, daily: pd.DataFrame, cfg: ScanConfig):
    """Approximate long ABCD on today's bars:

        A = session open            (move starts)
        B = highest high so far     (the peak)
        C = lowest low after B      (the pullback — must hold ABOVE A)
        D = price now back near B   (approaching a new high)

    Requires a real move A->B and a pullback that didn't give it all back.
    This is a screen, not proof: always eyeball the actual ABCD on the chart.
    """
    o = intraday
    if len(o) < 5:
        return None
    a = float(o["Open"].iloc[0])
    b_idx = int(o["High"].to_numpy().argmax())
    b = float(o["High"].iloc[b_idx])
    if b_idx >= len(o) - 1:            # peak is the last bar; no pullback yet
        return None
    c = float(o["Low"].iloc[b_idx:].min())
    price = float(o["Close"].iloc[-1])

    if a <= 0 or (b - a) / a < cfg.abcd_min_move_pct:   # need a real pole
        return None
    if c <= a:                                          # pullback must hold above A
        return None
    if price < b * (1 - cfg.abcd_new_high_tol):         # must be near a new high (D)
        return None
    return {"A": round(a, 2), "B": round(b, 2), "C": round(c, 2)}


# --------------------------------------------------------------------------- #
# Bull Flag / Momentum  (HEURISTIC — confirm on the chart)
# --------------------------------------------------------------------------- #
def detect_bull_flag(intraday: pd.DataFrame, daily: pd.DataFrame, cfg: ScanConfig):
    """Approximate bull flag: a strong surge (the 'pole') followed by a tight
    consolidation (the 'flag') that holds in the upper half of the pole.

    Pole  = all but the last 3 bars; must gain >= cfg.flag_min_pole_pct.
    Flag  = the last 3 bars; range must be tighter than cfg.flag_max_range_pct.
    """
    o = intraday
    flag_bars = 3
    if len(o) < flag_bars + 2:
        return None
    pole = o.iloc[:-flag_bars]
    flag = o.iloc[-flag_bars:]

    pole_low = float(pole["Low"].min())
    pole_high = float(pole["High"].max())
    if pole_low <= 0 or (pole_high - pole_low) / pole_low < cfg.flag_min_pole_pct:
        return None

    flag_high = float(flag["High"].max())
    flag_low = float(flag["Low"].min())
    price = float(o["Close"].iloc[-1])
    if price <= 0 or (flag_high - flag_low) / price > cfg.flag_max_range_pct:
        return None
    # flag should hold the upper half of the pole (still strong, not rolling over)
    if flag_low < pole_low + 0.5 * (pole_high - pole_low):
        return None
    return {"PoleLow": round(pole_low, 2), "PoleHigh": round(pole_high, 2)}


# --------------------------------------------------------------------------- #
# VWAP Pullback  (Aziz: VWAP is "essential"; precise)
# --------------------------------------------------------------------------- #
def _intraday_vwap(o: pd.DataFrame) -> float:
    """Volume-weighted average price for the session so far."""
    typical = (o["High"] + o["Low"] + o["Close"]) / 3
    cum_vol = o["Volume"].cumsum()
    return float((typical * o["Volume"]).cumsum().iloc[-1] / cum_vol.iloc[-1])


def detect_vwap_pullback(intraday: pd.DataFrame, daily: pd.DataFrame, cfg: ScanConfig):
    """Price trading near VWAP after holding the trend — a pullback entry zone.

    direction 'up'   -> price ABOVE VWAP but within cfg.vwap_band_pct of it
    direction 'down' -> price BELOW VWAP but within cfg.vwap_band_pct of it
    """
    o = intraday
    if o.empty or float(o["Volume"].sum()) <= 0:
        return None
    vwap = _intraday_vwap(o)
    if vwap <= 0:
        return None
    price = float(o["Close"].iloc[-1])
    dist = (price - vwap) / vwap        # signed distance from VWAP

    if cfg.direction == "up":
        if dist < 0 or dist > cfg.vwap_band_pct:
            return None
    else:
        if dist > 0 or -dist > cfg.vwap_band_pct:
            return None
    return {"VWAP": round(vwap, 2), "Dist%": round(dist * 100, 2)}


# --------------------------------------------------------------------------- #
# REGISTRY  —  the UI reads this to build its strategy picker.
# --------------------------------------------------------------------------- #
STRATEGIES = {
    "Reversal (Top/Bottom)": detect_reversal,
    "ORB — Opening Range Breakout": detect_orb,
    "ABCD Pattern": detect_abcd,
    "Bull Flag / Momentum": detect_bull_flag,
    "VWAP Pullback": detect_vwap_pullback,
}

# Which strategies are faithful vs heuristic (the UI shows a caveat for these).
HEURISTIC_STRATEGIES = {"ABCD Pattern", "Bull Flag / Momentum"}

# Aziz explicitly recommends these two for beginners.
BEGINNER_STRATEGIES = {"ORB — Opening Range Breakout", "ABCD Pattern"}


# --------------------------------------------------------------------------- #
# STOCKS-IN-PLAY PRESETS  —  Aziz's two scanners, verbatim from the book.
# --------------------------------------------------------------------------- #
# These set the universal "stocks in play" filters to his exact numbers. You
# still pick a STRATEGY (pattern) on top. "Custom" lets you set filters by hand.
#
# Each preset also records what free/delayed data CANNOT enforce, so the UI can
# be honest about the gap rather than pretend the match is perfect.

GAPPERS_PRESET = {
    "filters": dict(
        min_gap_pct=2.0,             # gapped up/down >= 2%
        min_gap_dollar=0.0,
        min_atr=0.50,                # ATR >= 50 cents
        min_rvol=0.0,                # RVOL is not a Gappers criterion
        min_avg_daily_volume=500_000,  # avg daily volume >= 500k
        min_volume_today=0,
        max_short_pct_float=30.0,    # avoid short interest > 30%
        min_price=1.0, max_price=1000.0,
    ),
    "cannot": [
        "Pre-market volume ≥ 50k — free feeds don't give reliable pre-market prints.",
        "Fundamental catalyst — no news feed on free data.",
        "The gap shown is OPEN-vs-prior-close, not the live pre-market gap.",
    ],
}

VOLUME_RADAR_PRESET = {
    "filters": dict(
        min_gap_dollar=1.0,          # gapped up/down >= $1
        min_gap_pct=0.0,
        min_atr=0.50,                # ATR > 50 cents
        min_rvol=1.5,                # relative volume >= 1.5
        min_avg_daily_volume=500_000,  # avg daily volume >= 500k
        min_volume_today=0,
        max_short_pct_float=None,
        min_price=1.0, max_price=1000.0,
    ),
    "cannot": [
        "RVOL here is naive (partial-day vs full-day avg), not time-of-day adjusted.",
        "The gap shown is OPEN-vs-prior-close, computed from delayed daily bars.",
    ],
}

STOCK_SELECTION_PRESETS = {
    "Pre-Market Gappers (Aziz)": GAPPERS_PRESET,
    "Intraday Volume Radar (Aziz)": VOLUME_RADAR_PRESET,
}

# Aziz's float/price -> favorite-strategy guidance (the book's table).
FLOAT_PRICE_GUIDE = [
    ("Low float (<20M)", "Under $10", "Momentum (long)"),
    ("Medium float (20–500M)", "$10–$100", "VWAP, Support/Resistance"),
    ("Large float (>500M)", "Usually $20+", "Moving Average, Reversal"),
]
