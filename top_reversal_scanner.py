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

import re
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


# --------------------------------------------------------------------------- #
# SCALE-UP  —  scan the ENTIRE US market instead of a hand-typed watchlist.
# --------------------------------------------------------------------------- #
# The honest tradeoff: free Yahoo data is delayed and rate-limited, so a full
# ~6,000-stock scan is SLOW (minutes) and can return partial results when Yahoo
# throttles. The three tricks below make it as fast as free data allows:
#   1. fetch_universe()      -> the real list of US tickers, cleaned of junk
#   2. _batch_history()      -> download ~150 stocks per request, not 1 at a time
#   3. run_scan_universe()   -> fetch the slow "float" field ONLY for stocks that
#                               already passed every other filter (the winners)

# Official NASDAQ symbol directory (covers Nasdaq + NYSE + AMEX). Plain pipe-
# delimited text, refreshed through each trading day.
_NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
_OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


def _parse_symbol_file(text: str, symbol_col: str) -> list[str]:
    """Pull clean common-stock tickers out of a NASDAQ directory file.

    Drops ETFs, test issues, and anything that isn't a plain 1-5 letter symbol
    (which removes most warrants/units/rights/preferred share classes). Whatever
    junk slips through gets filtered later by the price and volume thresholds.
    """
    lines = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.startswith("File Creation Time")
    ]
    if not lines:
        return []
    header = lines[0].split("|")
    if symbol_col not in header:
        return []
    si = header.index(symbol_col)
    ei = header.index("ETF") if "ETF" in header else None
    ti = header.index("Test Issue") if "Test Issue" in header else None

    out = []
    for ln in lines[1:]:
        parts = ln.split("|")
        if len(parts) <= si:
            continue
        sym = parts[si].strip()
        if ei is not None and ei < len(parts) and parts[ei].strip() == "Y":
            continue  # skip ETFs
        if ti is not None and ti < len(parts) and parts[ti].strip() == "Y":
            continue  # skip test issues
        if not re.fullmatch(r"[A-Z]{1,5}", sym):
            continue  # skip warrants/units/rights/odd symbols
        out.append(sym)
    return out


def fetch_universe() -> list[str]:
    """Download and clean the full list of US-traded common stocks.

    Returns a sorted, de-duplicated list of tickers (typically a few thousand).
    Needs network access at runtime (runs fine on Streamlit Cloud / your laptop).
    """
    import requests  # provided transitively by yfinance

    headers = {"User-Agent": "Mozilla/5.0 (reversal-scanner)"}
    symbols: set[str] = set()

    nasdaq = requests.get(_NASDAQ_LISTED_URL, headers=headers, timeout=30)
    symbols |= set(_parse_symbol_file(nasdaq.text, "Symbol"))

    other = requests.get(_OTHER_LISTED_URL, headers=headers, timeout=30)
    symbols |= set(_parse_symbol_file(other.text, "ACT Symbol"))

    return sorted(symbols)


def _batch_history(
    symbols: list[str],
    period: str,
    interval: str,
    chunk_size: int = 150,
    pause: float = 0.5,
    progress=None,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for many symbols in batches. -> {symbol: DataFrame}.

    yfinance fetches a whole chunk in one HTTP request, which is the key speedup.
    Symbols that fail or come back empty are simply omitted.
    """
    out: dict[str, pd.DataFrame] = {}
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    done = 0
    for chunk in chunks:
        try:
            data = yf.download(
                chunk, period=period, interval=interval,
                group_by="ticker", threads=True, progress=False, auto_adjust=True,
            )
        except Exception:
            data = None

        if data is not None and not data.empty:
            if len(chunk) == 1:
                df = data.dropna(how="all")
                if not df.empty:
                    out[chunk[0]] = df
            else:
                for sym in chunk:
                    try:
                        df = data[sym].dropna(how="all")
                    except Exception:
                        continue
                    if not df.empty:
                        out[sym] = df

        done += len(chunk)
        if progress is not None:
            progress(done, len(symbols))
        time.sleep(pause)
    return out


def run_scan_universe(
    symbols: list[str],
    cfg: ScanConfig | None = None,
    max_symbols: int | None = None,
    fetch_float_for_hits: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """Batched scan over a large symbol list (e.g. the whole US market).

    Same filters and indicators as run_scan, but built for scale:
      Phase 1  batch-download intraday + daily bars (no per-symbol float lookup)
      Phase 2  apply the consecutive-candle / price / ATR / RVOL / volume filters
      Phase 3  fetch float ONLY for the survivors, then apply the float cap
    """
    cfg = cfg or ScanConfig()
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    n = len(symbols)
    if n == 0:
        return pd.DataFrame()

    # Phase 1: two batched downloads. Report progress across both as one bar.
    def intraday_progress(done, total):
        if progress_callback:
            progress_callback(done, total * 2, "Downloading intraday bars")

    def daily_progress(done, total):
        if progress_callback:
            progress_callback(total + done, total * 2, "Downloading daily bars")

    intraday = _batch_history(symbols, "1d", cfg.bar_interval, progress=intraday_progress)
    daily = _batch_history(symbols, "6mo", "1d", progress=daily_progress)

    # Phase 2: in-memory filtering (fast, no network).
    if progress_callback:
        progress_callback(n * 2, n * 2, "Applying filters")
    hits = []
    for sym in symbols:
        bars, days = intraday.get(sym), daily.get(sym)
        if bars is None or days is None or bars.empty or days.empty:
            continue
        price = float(bars["Close"].iloc[-1])
        consec = count_consecutive_candles(bars, cfg.direction)
        atr = average_true_range(days, cfg.atr_period)
        rvol = relative_volume(days, cfg.rvol_lookback_days)
        vol_today = float(days["Volume"].iloc[-1])

        if consec < cfg.min_consecutive_candles:
            continue
        if not (cfg.min_price <= price <= cfg.max_price):
            continue
        if not (atr >= cfg.min_atr):
            continue
        if not (rvol >= cfg.min_rvol):
            continue
        if not (vol_today >= cfg.min_volume_today):
            continue

        hits.append({
            "Symbol": sym,
            "Consec": consec,
            "Price": round(price, 2),
            "Float(M)": None,            # filled in Phase 3 if requested
            "ATR": round(atr, 2),
            "VolToday": int(vol_today),
            "RVOL": round(rvol, 2),
        })

    # Phase 3: float lookup for survivors only (the expensive .info call).
    if fetch_float_for_hits and hits:
        for row in hits:
            flt = fetch_float_shares(yf.Ticker(row["Symbol"]))
            row["Float(M)"] = round(flt / 1e6, 1) if flt else None
        if cfg.max_float_shares is not None:
            cap_m = cfg.max_float_shares / 1e6
            hits = [r for r in hits if r["Float(M)"] is None or r["Float(M)"] <= cap_m]

    df = pd.DataFrame(hits)
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
