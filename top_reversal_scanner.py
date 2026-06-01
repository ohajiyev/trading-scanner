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
    min_rvol: float = 2.0            # relative volume floor (Aziz Volume Radar = 1.5)
    rvol_lookback_days: int = 50
    min_volume_today: int = 100_000

    # Float (optional). None = ignore. Set e.g. 100_000_000 to focus on
    # low-float momentum plays the way Aziz does for his faster setups.
    max_float_shares: float | None = None

    # --- Aziz "stocks in play" selection filters (off by default; presets set
    #     them to his exact book numbers). ---
    min_avg_daily_volume: int = 0    # Aziz: average daily volume >= 500,000
    min_gap_pct: float = 0.0         # Aziz Gappers: gapped up/down >= 2%
    min_gap_dollar: float = 0.0      # Aziz Volume Radar: gapped up/down >= $1
    max_short_pct_float: float | None = None  # Aziz: avoid short interest > 30%

    # --- Strategy-specific knobs (only the chosen strategy reads its own) ---
    # ORB: how many minutes define the opening range (first N min of session).
    orb_minutes: int = 15
    # ABCD: B must be this far above A (a real "pole"), and price must be within
    # new_high_tol of B to count as approaching point D.
    abcd_min_move_pct: float = 0.04
    abcd_new_high_tol: float = 0.01
    # Bull Flag: pole must gain at least this %, then the flag (last few bars)
    # must stay tighter than max_range_pct.
    flag_min_pole_pct: float = 0.05
    flag_max_range_pct: float = 0.03
    # VWAP pullback: price must be within this band of VWAP (a pullback, not
    # an extended move) on the correct side.
    vwap_band_pct: float = 0.006

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


def average_daily_volume(daily: pd.DataFrame, lookback: int = 50) -> float:
    """Average of the prior `lookback` days' volume (today excluded).

    Aziz wants this >= 500,000 — it's a liquidity gate separate from RVOL.
    """
    if len(daily) < 2:
        return float("nan")
    prior = daily["Volume"].iloc[-(lookback + 1):-1]
    return float(prior.mean()) if len(prior) else float("nan")


def gap_from_prev_close(daily: pd.DataFrame) -> tuple[float, float]:
    """(gap_percent, gap_dollar) of today's OPEN vs the prior day's CLOSE.

    IMPORTANT: this is the regular-session opening gap. It is NOT the live
    pre-market gap Aziz screens for — free data can't reliably give pre-market
    prints — but the open-vs-prior-close gap is the closest faithful proxy.
    """
    if len(daily) < 2:
        return float("nan"), float("nan")
    today_open = float(daily["Open"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2])
    if prev_close <= 0:
        return float("nan"), float("nan")
    gap_dollar = today_open - prev_close
    return gap_dollar / prev_close * 100, gap_dollar


def fetch_float_and_short(ticker) -> tuple[float | None, float | None]:
    """One .info call returning (float_shares, short_percent_of_float_%).

    Short interest is reported only ~twice a month by the exchanges, so this
    field is inherently stale even in professional tools — not a Claude/free
    limitation.
    """
    try:
        info = ticker.info
    except Exception:
        return None, None
    flt = info.get("floatShares") or info.get("sharesOutstanding")
    sp = info.get("shortPercentOfFloat")
    return flt, (sp * 100 if sp is not None else None)


# --------------------------------------------------------------------------- #
# STRATEGY ENGINE
# --------------------------------------------------------------------------- #
# Each strategy is just a "detector": a function
#     detect(intraday, daily, cfg) -> dict | None
# It returns the strategy-specific columns (e.g. {"Consec": 5}) when the pattern
# is present, or None when it isn't. The engine below handles everything shared:
# the universal "stocks in play" filters (price / ATR / RVOL / volume / float),
# data fetching, the funnel, and sorting. Add a new strategy by writing one
# detector — no need to touch the engine. (More detectors live in strategies.py.)


def bar_minutes(interval: str) -> int:
    """Minutes per bar for an interval string like '5m' / '15m'. Defaults to 5."""
    m = re.fullmatch(r"(\d+)m", interval.strip())
    return int(m.group(1)) if m else 5


def detect_reversal(intraday: pd.DataFrame, daily: pd.DataFrame, cfg: ScanConfig):
    """Top/Bottom Reversal: N consecutive same-direction candles (exhaustion)."""
    consec = count_consecutive_candles(intraday, cfg.direction)
    if consec < cfg.min_consecutive_candles:
        return None
    return {"Consec": consec}


def _passes_universal_filters(price, atr, rvol, vol_today, avg_vol,
                              gap_pct, gap_dollar, cfg: ScanConfig) -> bool:
    """The Aziz 'stocks in play' gates every strategy must clear.

    `not (x >= y)` form makes NaN values fail safely (the symbol is skipped).
    Each gate is active only when its threshold is set (> 0 / not None), so the
    base config behaves exactly as before and presets switch the rest on.
    """
    if not (cfg.min_price <= price <= cfg.max_price):
        return False
    if not (atr >= cfg.min_atr):
        return False
    if cfg.min_rvol > 0 and not (rvol >= cfg.min_rvol):
        return False
    if not (vol_today >= cfg.min_volume_today):
        return False
    if cfg.min_avg_daily_volume > 0 and not (avg_vol >= cfg.min_avg_daily_volume):
        return False
    if cfg.min_gap_pct > 0 and not (abs(gap_pct) >= cfg.min_gap_pct):
        return False
    if cfg.min_gap_dollar > 0 and not (abs(gap_dollar) >= cfg.min_gap_dollar):
        return False
    return True


def _evaluate(symbol, intraday, daily, cfg, strategy) -> dict | None:
    """Apply universal filters + the chosen strategy. Float and short interest
    are NOT fetched here (the engine fills them for survivors only).
    Returns a row dict or None."""
    if intraday is None or daily is None or intraday.empty or daily.empty:
        return None
    price = float(intraday["Close"].iloc[-1])
    atr = average_true_range(daily, cfg.atr_period)
    rvol = relative_volume(daily, cfg.rvol_lookback_days)
    vol_today = float(daily["Volume"].iloc[-1])
    avg_vol = average_daily_volume(daily, cfg.rvol_lookback_days)
    gap_pct, gap_dollar = gap_from_prev_close(daily)
    if atr != atr or rvol != rvol:        # NaN check (NaN != NaN)
        return None
    if not _passes_universal_filters(price, atr, rvol, vol_today, avg_vol,
                                     gap_pct, gap_dollar, cfg):
        return None
    extra = strategy(intraday, daily, cfg)
    if extra is None:                     # pattern not present
        return None
    row = {
        "Symbol": symbol,
        "Price": round(price, 2),
        "Gap%": round(gap_pct, 2) if gap_pct == gap_pct else None,
        "ATR": round(atr, 2),
        "RVOL": round(rvol, 2),
        "VolToday": int(vol_today),
        "AvgVol": int(avg_vol) if avg_vol == avg_vol else None,
        "Float(M)": None,                 # filled later for survivors only
        "Short%": None,                   # filled later for survivors only
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# SCAN  (single symbol + small watchlist)
# --------------------------------------------------------------------------- #
def scan_symbol(symbol: str, cfg: ScanConfig | None = None, strategy=None) -> dict | None:
    """Return a result row for `symbol` if it matches the strategy, else None."""
    cfg = cfg or ScanConfig()
    strategy = strategy or detect_reversal
    ticker = yf.Ticker(symbol)
    try:
        intraday = fetch_intraday(ticker, cfg.bar_interval)
        daily = fetch_daily(ticker)
    except Exception:
        return None
    row = _evaluate(symbol, intraday, daily, cfg, strategy)
    if row is not None:
        flt, short_pct = fetch_float_and_short(ticker)
        row["Float(M)"] = round(flt / 1e6, 1) if flt else None
        row["Short%"] = round(short_pct, 1) if short_pct is not None else None
        if (cfg.max_float_shares is not None and flt is not None
                and flt > cfg.max_float_shares):
            return None
        if (cfg.max_short_pct_float is not None and short_pct is not None
                and short_pct > cfg.max_short_pct_float):
            return None
    return row


def run_scan(
    symbols: list[str],
    cfg: ScanConfig | None = None,
    strategy=None,
    progress_callback=None,
) -> pd.DataFrame:
    """Scan a watchlist and return matches sorted by RVOL (highest first).

    strategy: a detector function (defaults to detect_reversal).
    progress_callback(done, total, symbol) is called after each symbol. Optional.
    """
    cfg = cfg or ScanConfig()
    strategy = strategy or detect_reversal
    rows = []
    total = len(symbols)
    for i, symbol in enumerate(symbols):
        row = scan_symbol(symbol, cfg, strategy)
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


def _prefilter_by_price(
    symbols: list[str], cfg: ScanConfig, progress=None
) -> list[str]:
    """Cheap first pass: keep only symbols whose latest price is in range.

    Downloads just today's single daily bar per stock (one light batch), so it
    runs fast across the whole market. Price is robust even mid-session, which
    makes it the ideal one-filter funnel before the heavy downloads.
    """
    snapshot = _batch_history(symbols, "1d", "1d", progress=progress)
    kept = []
    for sym, df in snapshot.items():
        try:
            price = float(df["Close"].iloc[-1])
        except Exception:
            continue
        if cfg.min_price <= price <= cfg.max_price:
            kept.append(sym)
    return kept


def run_scan_universe(
    symbols: list[str],
    cfg: ScanConfig | None = None,
    strategy=None,
    max_symbols: int | None = None,
    quick_prefilter: bool = True,
    fetch_float_for_hits: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """Batched scan over a large symbol list (e.g. the whole US market).

    Same universal filters as run_scan, applied via the shared strategy engine,
    but built for scale as a funnel:
      Phase 0  QUICK price pre-filter — one light download, drops most stocks
      Phase 1  heavy download (intraday + 6mo daily) for SURVIVORS only
      Phase 2  universal filters + the chosen strategy detector
      Phase 3  fetch float ONLY for the final winners, then apply the float cap

    strategy: a detector function (defaults to detect_reversal).
    Phase 0 is the speed win: instead of downloading 6 months of data for 6,000
    stocks, it cheaply narrows to a few hundred first. Set quick_prefilter=False
    to scan every stock the slow way.
    """
    cfg = cfg or ScanConfig()
    strategy = strategy or detect_reversal
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    if not symbols:
        return pd.DataFrame()

    # Phase 0: quick price pre-filter (the funnel's wide end).
    if quick_prefilter:
        before = len(symbols)

        def prefilter_progress(done, total):
            if progress_callback:
                progress_callback(done, total, "Quick price filter")

        symbols = _prefilter_by_price(symbols, cfg, progress=prefilter_progress)
        if progress_callback:
            progress_callback(
                len(symbols), before,
                f"Price filter kept {len(symbols)} of {before}",
            )
        if not symbols:
            return pd.DataFrame()

    # Phase 1: heavy batched downloads — only for the survivors now.
    def intraday_progress(done, total):
        if progress_callback:
            progress_callback(done, total, "Downloading intraday (survivors)")

    def daily_progress(done, total):
        if progress_callback:
            progress_callback(done, total, "Downloading daily (survivors)")

    intraday = _batch_history(symbols, "1d", cfg.bar_interval, progress=intraday_progress)
    daily = _batch_history(symbols, "6mo", "1d", progress=daily_progress)

    # Phase 2: universal filters + strategy detector (fast, no network).
    hits = []
    for sym in symbols:
        row = _evaluate(sym, intraday.get(sym), daily.get(sym), cfg, strategy)
        if row is not None:
            hits.append(row)

    # Phase 3: float + short interest for survivors only (one .info call each).
    if fetch_float_for_hits and hits:
        for row in hits:
            flt, short_pct = fetch_float_and_short(yf.Ticker(row["Symbol"]))
            row["Float(M)"] = round(flt / 1e6, 1) if flt else None
            row["Short%"] = round(short_pct, 1) if short_pct is not None else None
        if cfg.max_float_shares is not None:
            cap_m = cfg.max_float_shares / 1e6
            hits = [r for r in hits if r["Float(M)"] is None or r["Float(M)"] <= cap_m]
        if cfg.max_short_pct_float is not None:
            hits = [r for r in hits if r["Short%"] is None
                    or r["Short%"] <= cfg.max_short_pct_float]

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
