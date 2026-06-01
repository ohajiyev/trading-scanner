"""
Streamlit UI for the Aziz multi-strategy scanner  —  runs in any phone browser.
===============================================================================

Run locally:    streamlit run streamlit_app.py
Deploy (free):  push this folder to GitHub, then deploy on Streamlit
                Community Cloud (see the chat for the 5-minute steps).

All scan logic lives in top_reversal_scanner.py + strategies.py — this file is
ONLY the interface, so the rules stay in exactly one place.
"""

import datetime as dt
import re

import streamlit as st

from top_reversal_scanner import (
    ScanConfig,
    run_scan,
    run_scan_universe,
    fetch_universe,
    DEFAULT_WATCHLIST,
)
from strategies import (
    STRATEGIES, HEURISTIC_STRATEGIES, BEGINNER_STRATEGIES,
    STOCK_SELECTION_PRESETS, FLOAT_PRICE_GUIDE,
)

st.set_page_config(
    page_title="Aziz Scanner",
    page_icon="📈",
    layout="centered",          # 'centered' reads better than 'wide' on phones
)

st.title("📈 Aziz Day-Trading Scanner")
st.caption(
    "Andrew Aziz–style scans on free, ~15-min-delayed data. "
    "A learning tool, not trading advice."
)

# --------------------------------------------------------------------------- #
# SIDEBAR = your ScanConfig control panel. On mobile, tap the ›/hamburger.
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Strategy")
    strategy_name = st.selectbox(
        "Scanner", list(STRATEGIES.keys()),
        help="Each strategy is its own scanner. They share the same 'stocks in "
             "play' filters below; only the pattern differs.",
    )
    strategy_fn = STRATEGIES[strategy_name]
    if strategy_name in BEGINNER_STRATEGIES:
        st.caption("⭐ Aziz recommends this one for beginners.")
    if strategy_name in HEURISTIC_STRATEGIES:
        st.caption("⚠️ Pattern is approximated — confirm it on the chart yourself.")

    st.divider()
    st.header("Filters")

    direction_label = st.radio(
        "Direction",
        ["Up (long-side)", "Down (short-side)"],
        help="Applies to Reversal, ORB, and VWAP. "
             "Up = exhausted rally / break above range / above VWAP. "
             "Down = the mirror. (ABCD and Bull Flag are long-only.)",
    )
    direction = "up" if direction_label.startswith("Up") else "down"

    # --- strategy-specific controls (only shown when relevant) ---
    min_consec = 4
    orb_minutes = 15
    if strategy_name == "Reversal (Top/Bottom)":
        min_consec = st.slider(
            "Min consecutive candles", 2, 10, 4,
            help="The exhaustion signal. Aziz looks for roughly 4–6.",
        )
    if strategy_name == "ORB — Opening Range Breakout":
        orb_minutes = st.selectbox(
            "Opening range (minutes)", [5, 15, 30], index=1,
            help="The session's first N minutes define the range to break.",
        )

    bar_interval = st.selectbox("Candle size", ["1m", "2m", "5m", "15m"], index=2)

    # --- Stock selection: Aziz preset, or set filters by hand ---
    selection_mode = st.selectbox(
        "Stock selection (stocks in play)",
        ["Custom"] + list(STOCK_SELECTION_PRESETS.keys()),
        index=1,
        help="The Aziz presets lock in his exact book criteria. "
             "Custom lets you set the filters by hand.",
    )

    if selection_mode == "Custom":
        price_min, price_max = st.slider("Price range ($)", 0, 500, (5, 100))
        min_atr = st.number_input(
            "Min ATR ($)", value=0.50, step=0.10, min_value=0.0,
            help="Daily average true range. Aziz: ≥ $0.50.",
        )
        min_rvol = st.number_input(
            "Min RVOL", value=2.0, step=0.5, min_value=0.0,
            help="Relative volume. Aziz's Volume Radar uses 1.5.",
        )
        min_vol = st.number_input(
            "Min volume today", value=100_000, step=50_000, min_value=0,
        )
        float_cap_m = st.number_input(
            "Max float (M shares, 0 = ignore)", value=0, step=10, min_value=0,
            help="Lower float = faster, cleaner moves. 0 disables.",
        )
        filter_kwargs = dict(
            min_price=float(price_min), max_price=float(price_max),
            min_atr=float(min_atr), min_rvol=float(min_rvol),
            min_volume_today=int(min_vol),
            max_float_shares=None if float_cap_m == 0 else float(float_cap_m) * 1e6,
        )
    else:
        preset = STOCK_SELECTION_PRESETS[selection_mode]
        filter_kwargs = dict(preset["filters"])
        f = preset["filters"]
        crit = []
        if f.get("min_gap_pct"):
            crit.append(f"Gap ≥ {f['min_gap_pct']:.0f}%")
        if f.get("min_gap_dollar"):
            crit.append(f"Gap ≥ ${f['min_gap_dollar']:.0f}")
        crit.append(f"ATR ≥ ${f['min_atr']:.2f}")
        if f.get("min_rvol"):
            crit.append(f"RVOL ≥ {f['min_rvol']}")
        if f.get("min_avg_daily_volume"):
            crit.append(f"Avg daily vol ≥ {f['min_avg_daily_volume']:,}")
        if f.get("max_short_pct_float") is not None:
            crit.append(f"Short interest ≤ {f['max_short_pct_float']:.0f}%")
        st.success("Aziz criteria: " + " · ".join(crit))
        st.caption("Can't enforce on free data — " + " ".join(preset["cannot"]))

    st.divider()
    universe_mode = st.radio(
        "Scan universe",
        ["My watchlist", "All US stocks (slow)"],
        help="Watchlist = the tickers below (fast). "
             "All US stocks = the whole market via batched downloads (minutes, "
             "and may return partial results when Yahoo rate-limits).",
    )

    if universe_mode == "My watchlist":
        watchlist_raw = st.text_area(
            "Watchlist (comma or space separated)",
            ", ".join(DEFAULT_WATCHLIST),
            height=120,
        )
        max_symbols = None
    else:
        watchlist_raw = ""
        st.warning(
            "Free data is ~15 min delayed and rate-limited. A full scan of "
            "thousands of stocks takes several minutes and can come back partial. "
            "Start with the cap below and raise it once you've seen it work."
        )
        max_symbols = st.number_input(
            "Max stocks to scan (0 = no cap, the full market)",
            value=500, step=250, min_value=0,
            help="A safety limit so a first run doesn't take 10+ minutes. "
                 "Set 0 to scan everything once you're comfortable.",
        )
        max_symbols = None if max_symbols == 0 else int(max_symbols)

        quick_prefilter = st.checkbox(
            "⚡ Quick price pre-filter", value=True,
            help="Run the Price range filter FIRST on a light download, then only "
                 "pull heavy data for survivors. Big speedup. Uncheck to scan every "
                 "stock the slow way.",
        )

# --------------------------------------------------------------------------- #
# Build the config from the controls
# --------------------------------------------------------------------------- #
symbols = sorted({s.upper() for s in re.split(r"[\s,]+", watchlist_raw) if s.strip()})

cfg = ScanConfig(
    **filter_kwargs,
    direction=direction,
    min_consecutive_candles=min_consec,
    bar_interval=bar_interval,
    orb_minutes=orb_minutes,
)

@st.cache_data(ttl=3600, show_spinner="Fetching the US stock list…")
def load_universe():
    """Download the full US ticker list once per hour (cached)."""
    return fetch_universe()


if universe_mode == "My watchlist":
    st.write(
        f"**{len(symbols)} symbols** · {strategy_name} · "
        f"{direction.upper()} · {selection_mode}"
    )
else:
    cap_text = "full market" if max_symbols is None else f"first {max_symbols}"
    st.write(
        f"**All US stocks** ({cap_text}) · {strategy_name} · "
        f"{direction.upper()} · {selection_mode}"
    )

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if st.button("🔍 Run scan", type="primary", use_container_width=True):
    bar = st.progress(0.0, text="Starting…")

    def on_progress(done, total, label):
        pct = min(done / total, 1.0) if total else 0.0
        bar.progress(pct, text=f"{label}  ({done}/{total})")

    if universe_mode == "My watchlist":
        if not symbols:
            st.error("Add at least one ticker to the watchlist.")
            st.stop()
        results = run_scan(symbols, cfg, strategy=strategy_fn,
                           progress_callback=on_progress)
    else:
        all_symbols = load_universe()
        st.caption(f"Universe loaded: {len(all_symbols):,} US common stocks.")
        results = run_scan_universe(
            all_symbols, cfg, strategy=strategy_fn, max_symbols=max_symbols,
            quick_prefilter=quick_prefilter, progress_callback=on_progress,
        )

    bar.empty()

    if results.empty:
        st.warning(
            "No matches right now. Most of these patterns need a live, active "
            "session — outside 09:30–16:00 ET the signals go stale. Try loosening "
            "**Min RVOL** or the **Price range**, or pick a different strategy. "
            "(A full-market scan can also come back empty if Yahoo rate-limited "
            "the downloads — just retry.)"
        )
    else:
        st.success(f"{len(results)} match(es) · scanned {dt.datetime.now():%H:%M:%S}")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
            column_config={
                # ProgressColumn echoes the screenshot's blue RVOL bar
                "RVOL": st.column_config.ProgressColumn(
                    "RVOL",
                    min_value=0.0,
                    max_value=float(max(results["RVOL"].max(), 5.0)),
                    format="%.2f",
                ),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "VolToday": st.column_config.NumberColumn("Vol Today", format="%d"),
                "Float(M)": st.column_config.NumberColumn("Float (M)", format="%.1f"),
                "ATR": st.column_config.NumberColumn("ATR", format="$%.2f"),
            },
        )
else:
    st.info("Set your filters in the sidebar (tap **›** on mobile), then **Run scan**.")
