"""
V5 Top Reversal scanner  —  Andrew Aziz style, free-data edition.
=================================================================

THE STRATEGY (why these columns exist)
--------------------------------------
A "Top Reversal" hunts for a stock whose rally has gone too far, too fast, and
is likely to snap back down (a short setup). The tell is a run of consecutive
UP candles -- buyers exhausting themselves -- CONFIRMED by abnormally high
relative volume. Without the volume, an extended move is just noise; with it,
real money is involved and the reversal is tradeable. That is why, in the
screenshot, the finger points at the Rel Vol column.

Filters (all from Aziz's "stocks in play" criteria), mapped to the columns:
  Consec Cndls  -> N consecutive same-direction candles (the exhaustion signal)
  Price ($)     -> mid-priced names, not penny junk
  Flt (Shr)     -> float; lower = cleaner, faster moves (optional here)
  Avg True      -> ATR; enough daily range that a reversal is worth trading
  Vol Today     -> today's volume
  Rel Vol       -> today's volume vs normal (the make-or-break confirmation)

"Bottom Reversal" is the mirror image: set direction="down" to scan for
exhausted SELL-offs (consecutive down candles) as long setups.

DATA NOTE
---------
yfinance is free and needs no key, but it is DELAYED ~15 min and occasionally
flaky. That is fine for learning the logic. To go real-time later, swap only the
two functions in the "DATA LAYER" block below (e.g. Alpaca / Polygon websockets);
everything else stays exactly the same.

RUN
---
    pip install yfinance pandas
    python top_reversal_scanner.py
Best run during or just after US market hours (09:30-16:00 ET).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


# --------------------------------------------------------------------------- #
# CONFIG  —  these are your scanner's filter knobs. Tweak freely.
# --------------------------------------------------------------------------- #
@dataclass
class ScanConfig:
    # The reversal trigger
    direction: str = "up"            # "up" = Top Reversal, "down" = Bottom Reversal
    min_consecutive_candles: int = 4 # >=4 same-direction candles (screenshot shows 4-6)
    bar_interval: str = "5m"         # candle size for the consecutive count

    # Tradeability filters
    min_price: float = 5.0
    max_price: float = 100.0
    min_atr: float = 0.50            # Aziz: daily ATR >= $0.50
    atr_period: int = 14

    # Volume confirmation
    min_rvol: float = 2.0            # Aziz's #1 criterion: relative volume >= 2x
    rvol_lookback_days: int = 50
    min_volume_today: int = 100_000

    # Float (optional). None = ignore. Set e.g. 100_000_000 to focus on
    # low-float momentum plays the way Aziz does for his faster setups.
    max_float_shares: float | None = None

    request_pause: float = 0.3       # be polite to the free data endpoint


DEFAULT_WATCHLIST = [
    "TSLA", "NVDA", "AMD", "PLTR", "SOFI", "RIVN", "LCID", "MARA", "RIOT",
    "COIN", "HOOD", "AFRM", "UPST", "DKNG", "SNAP", "F", "NIO", "CCL",
    "AAL", "PLUG", "FSLR", "ENPH", "ROKU", "CVNA", "SMCI",
]


# --------------------------------------------------------------------------- #
# DATA LAYER  —  swap these two functions to change data source / go real-time.
# --------------------------------------------------------------------------- #
def fetch_intraday(ticker: yf.Ticker, interval: str) -> pd.DataFrame:
    """Today's intraday OHLCV bars. Columns: Open High Low Close Volume."""
    return ticker.history(period="1d", interval=interval)


def fetch_daily(ticker: yf.Ticker) -> pd.DataFrame:
    """~6 months of daily bars (covers ATR and the RVOL average)."""
    return ticker.history(period="6mo", interval="1d")


def fetch_float_shares(ticker: yf.Ticker) -> float | None:
    """Public float in shares (falls back to shares outstanding)."""
    try:
        info = ticker.info
    except Exception:
        return None
    return info.get("floatShares") or info.get("sharesOutstanding")


# --------------------------------------------------------------------------- #
# INDICATORS  —  pure functions, easy to unit-test.
# --------------------------------------------------------------------------- #
def count_consecutive_candles(bars: pd.DataFrame, direction: str = "up") -> int:
    """Number of consecutive same-direction candles ending at the latest bar."""
    if bars.empty:
        return 0
    up = (bars["Close"] > bars["Open"]).to_numpy()
    matched = up if direction == "up" else ~up
    count = 0
    for ok in reversed(matched):
        if ok:
            count += 1
        else:
            break
    return count


def average_true_range(daily: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR on daily bars."""
    if len(daily) < period + 1:
        return float("nan")
    high, low, prev_close = daily["High"], daily["Low"], daily["Close"].shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing == EMA with alpha = 1/period
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return float(atr.iloc[-1])


def relative_volume(daily: pd.DataFrame, lookback: int = 50) -> float:
    """Today's volume divided by the average daily volume over `lookback` days.

    NOTE: intraday this UNDER-reads, because today's bar is only a partial day
    compared against full-day averages. The professional fix is a time-of-day
    adjusted RVOL (compare to the average volume *by this time of day*). See the
    docstring at the bottom of this file for how to upgrade it.
    """
    if len(daily) < 2:
        return float("nan")
    today_volume = float(daily["Volume"].iloc[-1])
    prior = daily["Volume"].iloc[-(lookback + 1):-1]
    avg = float(prior.mean()) if len(prior) else float("nan")
    if not avg or avg <= 0:
        return float("nan")
    return today_volume / avg


# --------------------------------------------------------------------------- #
# SCAN
# --------------------------------------------------------------------------- #
def scan_symbol(symbol: str, cfg: ScanConfig) -> dict | None:
    """Return a result row for `symbol` if it passes every filter, else None."""
    ticker = yf.Ticker(symbol)
    try:
        intraday = fetch_intraday(ticker, cfg.bar_interval)
        daily = fetch_daily(ticker)
    except Exception:
        return None
    if intraday.empty or daily.empty:
        return None

    price = float(intraday["Close"].iloc[-1])
    consec = count_consecutive_candles(intraday, cfg.direction)
    atr = average_true_range(daily, cfg.atr_period)
    rvol = relative_volume(daily, cfg.rvol_lookback_days)
    vol_today = float(daily["Volume"].iloc[-1])
    flt = fetch_float_shares(ticker)

    # Filters. The `not (x >= y)` form makes NaN values fail safely (skip).
    if consec < cfg.min_consecutive_candles:
        return None
    if not (cfg.min_price <= price <= cfg.max_price):
        return None
    if not (atr >= cfg.min_atr):
        return None
    if not (rvol >= cfg.min_rvol):
        return None
    if not (vol_today >= cfg.min_volume_today):
        return None
    if cfg.max_float_shares is not None and flt is not None and flt > cfg.max_float_shares:
        return None

    return {
        "Symbol": symbol,
        "Consec": consec,
        "Price": round(price, 2),
        "Float(M)": round(flt / 1e6, 1) if flt else None,
        "ATR": round(atr, 2),
        "VolToday": int(vol_today),
        "RVOL": round(rvol, 2),
    }


def run_scan(
    symbols: list[str],
    cfg: ScanConfig | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Scan a watchlist and return matches sorted by RVOL (highest first).

    progress_callback(done: int, total: int, symbol: str) is called after each
    symbol, so a UI (e.g. Streamlit) can show a progress bar. Optional.
    """
    cfg = cfg or ScanConfig()
    rows = []
    total = len(symbols)
    for i, symbol in enumerate(symbols):
        row = scan_symbol(symbol, cfg)
        if row:
            rows.append(row)
        if progress_callback is not None:
            progress_callback(i + 1, total, symbol)
        time.sleep(cfg.request_pause)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("RVOL", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    config = ScanConfig()
    results = run_scan(DEFAULT_WATCHLIST, config)

    label = "TOP" if config.direction == "up" else "BOTTOM"
    print(f"\n=== V5 {label} Reversal  ({config.min_consecutive_candles}+ "
          f"consec candles, RVOL>={config.min_rvol}) ===\n")
    if results.empty:
        print("No matches right now.")
        print("Tips: run during US market hours, or loosen min_rvol / "
              "min_consecutive_candles in ScanConfig.")
    else:
        print(results.to_string(index=False))


# --------------------------------------------------------------------------- #
# NEXT STEP — time-of-day adjusted RVOL (the upgrade that makes RVOL real)
# --------------------------------------------------------------------------- #
# Plain RVOL compares a partial day to full days, so at 10:00 ET everything
# looks "low". The fix: build, per symbol, the *cumulative* volume profile by
# time-of-day from ~20 days of intraday bars, then compare today's cumulative
# volume at the current time against that historical same-time average:
#
#     rvol_tod = volume_so_far_today / avg_volume_by_this_time_of_day
#
# That is exactly what Trade Ideas' "Rel Vol" column does, and it is the single
# most valuable refinement once the basic version is working.
