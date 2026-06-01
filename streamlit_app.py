"""
Streamlit UI for the V5 Reversal scanner  —  runs in any phone browser.
=======================================================================

Run locally:    streamlit run streamlit_app.py
Deploy (free):  push this folder to GitHub, then deploy on Streamlit
                Community Cloud (see the chat for the 5-minute steps).

All scan logic lives in top_reversal_scanner.py — this file is ONLY the
interface, so the filter rules stay in exactly one place.
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

st.set_page_config(
    page_title="Reversal Scanner",
    page_icon="📉",
    layout="centered",          # 'centered' reads better than 'wide' on phones
)

st.title("📉 V5 Reversal Scanner")
st.caption(
    "Andrew Aziz–style reversal scan on free, ~15-min-delayed data. "
    "A learning tool, not trading advice."
)

# --------------------------------------------------------------------------- #
# SIDEBAR = your ScanConfig control panel. On mobile, tap the ›/hamburger.
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Filters")

    direction_label = st.radio(
        "Setup",
        ["Top Reversal (shorts)", "Bottom Reversal (longs)"],
        help="Top = exhausted rally (consecutive UP candles, short setup). "
             "Bottom = exhausted sell-off (consecutive DOWN candles, long setup).",
    )
    direction = "up" if direction_label.startswith("Top") else "down"

    min_consec = st.slider(
        "Min consecutive candles", 2, 10, 4,
        help="The exhaustion signal. Aziz looks for roughly 4–6.",
    )
    bar_interval = st.selectbox("Candle size", ["1m", "2m", "5m", "15m"], index=2)

    price_min, price_max = st.slider("Price range ($)", 0, 500, (5, 100))

    min_atr = st.number_input(
        "Min ATR ($)", value=0.50, step=0.10, min_value=0.0,
        help="Daily average true range — enough room to be worth trading. Aziz: ≥ $0.50.",
    )
    min_rvol = st.number_input(
        "Min RVOL", value=2.0, step=0.5, min_value=0.0,
        help="Relative volume — the confirmation. Aziz's single most important filter.",
    )
    min_vol = st.number_input(
        "Min volume today", value=100_000, step=50_000, min_value=0,
    )
    float_cap_m = st.number_input(
        "Max float (M shares, 0 = ignore)", value=0, step=10, min_value=0,
        help="Lower float = faster, cleaner moves. Leave at 0 to disable.",
    )

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

# --------------------------------------------------------------------------- #
# Build the config from the controls
# --------------------------------------------------------------------------- #
symbols = sorted({s.upper() for s in re.split(r"[\s,]+", watchlist_raw) if s.strip()})

cfg = ScanConfig(
    direction=direction,
    min_consecutive_candles=min_consec,
    bar_interval=bar_interval,
    min_price=float(price_min),
    max_price=float(price_max),
    min_atr=float(min_atr),
    min_rvol=float(min_rvol),
    min_volume_today=int(min_vol),
    max_float_shares=None if float_cap_m == 0 else float(float_cap_m) * 1e6,
)

setup_name = "Top" if direction == "up" else "Bottom"


@st.cache_data(ttl=3600, show_spinner="Fetching the US stock list…")
def load_universe():
    """Download the full US ticker list once per hour (cached)."""
    return fetch_universe()


if universe_mode == "My watchlist":
    st.write(
        f"**{len(symbols)} symbols** · {setup_name} Reversal · "
        f"{min_consec}+ candles · RVOL ≥ {min_rvol}"
    )
else:
    cap_text = "full market" if max_symbols is None else f"first {max_symbols}"
    st.write(
        f"**All US stocks** ({cap_text}) · {setup_name} Reversal · "
        f"{min_consec}+ candles · RVOL ≥ {min_rvol}"
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
        results = run_scan(symbols, cfg, progress_callback=on_progress)
    else:
        all_symbols = load_universe()
        st.caption(f"Universe loaded: {len(all_symbols):,} US common stocks.")
        results = run_scan_universe(
            all_symbols, cfg, max_symbols=max_symbols, progress_callback=on_progress
        )

    bar.empty()

    if results.empty:
        st.warning(
            "No matches right now. Outside US market hours the consecutive-candle "
            "count goes stale — try lowering **Min RVOL** or **Min consecutive "
            "candles**, or scan during 09:30–16:00 ET. (A full-market scan can also "
            "come back empty if Yahoo rate-limited the downloads — just retry.)"
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
