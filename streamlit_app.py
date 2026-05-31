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

from top_reversal_scanner import ScanConfig, run_scan, DEFAULT_WATCHLIST

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
    watchlist_raw = st.text_area(
        "Watchlist (comma or space separated)",
        ", ".join(DEFAULT_WATCHLIST),
        height=120,
    )

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
st.write(
    f"**{len(symbols)} symbols** · {setup_name} Reversal · "
    f"{min_consec}+ candles · RVOL ≥ {min_rvol}"
)

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if st.button("🔍 Run scan", type="primary", use_container_width=True):
    if not symbols:
        st.error("Add at least one ticker to the watchlist.")
    else:
        bar = st.progress(0.0, text="Starting…")

        def on_progress(done, total, symbol):
            bar.progress(done / total, text=f"Scanning {symbol}  ({done}/{total})")

        results = run_scan(symbols, cfg, progress_callback=on_progress)
        bar.empty()

        if results.empty:
            st.warning(
                "No matches right now. Outside US market hours the consecutive-candle "
                "count goes stale — try lowering **Min RVOL** or **Min consecutive "
                "candles**, or scan during 09:30–16:00 ET."
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
