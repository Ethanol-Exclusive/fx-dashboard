"""
daily_dashboard.py

Run this every morning (or on a schedule). It pulls fresh data for all
configured instruments, runs all strategy setups (Daily PDH/PDL, NY 5AM CRT,
and Major Liquidity), and renders a single mobile-friendly HTML page showing
exactly where each instrument stands: waiting / swept / MSS confirmed /
entry+SL+TP1+TP2 levels + confluence tags.

Usage:
    python daily_dashboard.py

Output:
    index.html   <- open this on your phone
"""

import datetime
import json
import os
import traceback
import pandas as pd
import yfinance as yf
import requests

from strategy_v2 import (
    analyze_daily_setup,
    analyze_ny_crt_setup,
    run_major_liquidity_across_timeframes,
    detect_5m_mss_bos,
    SetupState,
    MSSSignal,
    NY_TZ,
)

SYMBOLS = {
    "GOLD":   "GC=F",
    "NAS100": "^NDX",
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "BTCUSD": "BTC-USD",
    "AUDUSD": "AUDUSD=X",
    "US30":   "^DJI",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "GER30":  "^GDAXI",
}

# Most instruments' CRT candle opens at 5AM NY. Indices (and BTCUSD) have
# been observed to sometimes shift to a different hour depending on the day
# (likely DST/session-calendar drift), so these list every hour worth
# trying - the first one that yields a real (non-empty) result is used.
CRT_HOUR_CANDIDATES = {
    "NAS100": [5, 6],
    "US30": [5, 6],
    "GER30": [5, 6],
    "BTCUSD": [4, 5],
}

# Deep link straight to the workflow run page - update this if your username/repo differ
REPO_ACTIONS_URL = "https://github.com/Ethanol-Exclusive/fx-dashboard/actions/workflows/update-dashboard.yml"

STATUS_COLORS = {
    "waiting": "#555",
    "sweep_only": "#c9a227",
    "mss_confirmed": "#1e9e5a",
    "no_data": "#999",
}

STATUS_LABELS = {
    "waiting": "WAITING",
    "sweep_only": "SWEPT — awaiting MSS",
    "mss_confirmed": "SIGNAL LIVE",
    "no_data": "NO DATA",
}

STATE_FILE = "last_signals.json"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def load_previous_signals():
    """Loads the set of signal keys we've already notified about, so we
    don't spam the same confirmed signal every time the workflow re-runs."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_signals(signals):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(signals, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save {STATE_FILE}: {e}")


def signal_key(symbol, setup_name, state):
    """A unique fingerprint for a specific confirmed signal - includes entry
    price so that if the SAME setup reconfirms at a genuinely new level
    later, it's treated as new, but re-running the workflow on an unchanged
    signal won't re-notify."""
    entry = f"{state.entry_price:.5f}" if state.entry_price is not None else "na"
    return f"{symbol}|{setup_name}|{state.direction}|{entry}"


def mss_signal_key(signal: MSSSignal) -> str:
    """Fingerprint for a fully-confirmed 5m NY CRT signal. Uses entry price
    as the unique anchor — same logic as the Setup A/B signal_key — so
    re-running the workflow on an unchanged signal won't re-notify."""
    price = f"{signal.entry_price:.5f}"
    return f"{signal.symbol}|5m_CRT_confirmed|{signal.direction}|{price}"


def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured (missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets) - skipping alert.")
        return

    # Weekday-only guard (belt-and-suspenders alongside the cron schedule's
    # own weekday restriction) - no weekend trading, so never alert on a
    # Saturday or Sunday, even if the workflow gets triggered manually or
    # runs late into a weekend boundary. Uses NY time since that's the
    # relevant trading calendar for these setups.
    ny_now = pd.Timestamp.now(tz="America/New_York")
    if ny_now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        print(f"Skipping Telegram alert - it's the weekend in NY ({ny_now.strftime('%A')}), no weekend trading.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }, timeout=10)
        if resp.status_code != 200:
            print(f"Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Telegram send error: {e}")


def build_alert_message(symbol, state: SetupState) -> str:
    conf = ", ".join(state.confluence) if state.confluence else "none"
    direction_word = "BUY" if state.direction == "long" else "SELL"
    emoji = "🟢" if state.direction == "long" else "🔴"
    trend_line = "\n⚠️ *AGAINST 4H TREND*\n" if state.against_trend else ""

    tp1_line = f"TP1 (→ BE): `{state.tp1:.5f}`\n" if state.tp1 is not None else "TP1: no liquidity found yet\n"
    tp2_line = f"TP2: `{state.tp2:.5f}`\n" if state.tp2 is not None else ""

    return (
        f"{emoji} *{direction_word} — {symbol}*\n"
        f"_{state.setup_name}_\n"
        f"{trend_line}\n"
        f"Entry: `{state.entry_price:.5f}`\n"
        f"Stop Loss: `{state.stop_loss:.5f}`\n"
        f"{tp1_line}"
        f"{tp2_line}"
        f"Confluence: {conf}"
    )


def build_mss_alert_message(signal: MSSSignal) -> str:
    """
    Telegram message for a fully-confirmed 5m NY CRT signal.
    Only sent after sweep + MSS + confluence + in-range entry are all met —
    so the format mirrors the Setup A/B alert exactly.
    """
    conf = ", ".join(signal.confluence) if signal.confluence else "none"
    direction_word = "BUY" if signal.direction == "long" else "SELL"
    emoji = "🟢" if signal.direction == "long" else "🔴"
    trend_line = "\n⚠️ *AGAINST 4H TREND*\n" if signal.against_trend else ""
    time_str = signal.signal_time.strftime("%H:%M NY") if signal.signal_time else ""
    time_line = f"Time: {time_str}\n" if time_str else ""

    tp1_line = f"TP1 (→ BE): `{signal.tp1:.5f}`\n" if signal.tp1 is not None else "TP1: no liquidity found yet\n"
    tp2_line = f"TP2: `{signal.tp2:.5f}`\n" if signal.tp2 is not None else ""

    return (
        f"{emoji} *{direction_word} — {signal.symbol}*\n"
        f"_NY 5AM CRT (5m confirmed)_\n"
        f"{trend_line}\n"
        f"Entry: `{signal.entry_price:.5f}`\n"
        f"Stop Loss: `{signal.stop_loss:.5f}`\n"
        f"{tp1_line}"
        f"{tp2_line}"
        f"{time_line}"
        f"Confluence: {conf}"
    )


# OANDA is the PRIMARY data source for all instruments - real spot/CFD
# pricing that tracks brokers far more closely than Yahoo's mix of futures
# (GC=F) and cash index (^NDX, ^DJI) tickers. Yahoo Finance is kept only as
# a backup for whenever OANDA is unavailable (rate limit, outage, missing
# instrument, or no token configured).
OANDA_API_TOKEN = os.environ.get("OANDA_API_TOKEN", "")
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"
OANDA_INSTRUMENTS = {
    "GOLD":   "XAU_USD",
    "NAS100": "NAS100_USD",
    "EURUSD": "EUR_USD",
    "USDJPY": "USD_JPY",
    "GBPUSD": "GBP_USD",
    "BTCUSD": "BTC_USD",
    "AUDUSD": "AUD_USD",
    "US30":   "US30_USD",
    "USDCAD": "USD_CAD",
    "NZDUSD": "NZD_USD",
    "GER30":  "DE30_EUR",
}


def fetch_oanda(name, granularity, count):
    """Pulls OHLC candles from OANDA's v20 API.
    granularity: OANDA codes - "D" (daily), "H1" (1 hour), "M15" (15 min),
                               "M30" (30 min), "M5" (5 min)"""
    instrument = OANDA_INSTRUMENTS.get(name)
    if not instrument or not OANDA_API_TOKEN:
        return pd.DataFrame()

    try:
        url = f"{OANDA_BASE_URL}/v3/instruments/{instrument}/candles"
        headers = {"Authorization": f"Bearer {OANDA_API_TOKEN}"}
        params = {"granularity": granularity, "count": count, "price": "M"}
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"OANDA fetch failed for {name} ({granularity}): {resp.status_code} {resp.text[:200]}")
            return pd.DataFrame()
        candles = resp.json().get("candles", [])
    except Exception as e:
        print(f"OANDA fetch error for {name} ({granularity}): {e}")
        return pd.DataFrame()

    rows = []
    for c in candles:
        if not c.get("complete", True):
            continue
        try:
            t = pd.Timestamp(c["time"])
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            mid = c["mid"]
            rows.append({"time": t, "Open": float(mid["o"]), "High": float(mid["h"]),
                         "Low": float(mid["l"]), "Close": float(mid["c"])})
        except (KeyError, ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("time").sort_index()


def fetch_yahoo(ticker, interval, period):
    """Backup fetch via Yahoo Finance, used only when OANDA returns empty."""
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def fetch(name, interval, period):
    """Pulls OHLC data for one instrument, trying OANDA first (primary) and
    falling back to Yahoo Finance if OANDA has nothing usable (no token
    configured, rate limited, instrument unavailable, etc). Returns an
    empty DataFrame (not an exception) if both sources fail, so callers
    can check .empty rather than needing a try/except at every call site.

    interval: "1d", "1h", "30m", "15m", "5m" (mapped internally to each source's codes)
    """
    oanda_map = {
        "1d":  ("D",   30),
        "1h":  ("H1",  500),
        "30m": ("M30", 2000),
        "15m": ("M15", 2000),
        "5m":  ("M5",  500),
    }
    oanda_granularity, oanda_count = oanda_map.get(interval, ("M15", 2000))

    df = fetch_oanda(name, oanda_granularity, oanda_count)
    if not df.empty:
        return df

    # OANDA unavailable/empty - fall back to Yahoo (Yahoo doesn't have 30m,
    # so we resample from 15m in that case)
    ticker = SYMBOLS.get(name)
    if ticker:
        print(f"OANDA returned nothing for {name} ({interval}) - trying Yahoo Finance backup.")
        if interval == "30m":
            df_15 = fetch_yahoo(ticker, "15m", "60d")
            if not df_15.empty:
                try:
                    return df_15.resample("30min").agg(
                        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
                    ).dropna()
                except Exception:
                    pass
            return pd.DataFrame()
        yahoo_interval = {"1d": "1d", "1h": "1h", "15m": "15m", "5m": "5m"}.get(interval, "15m")
        yahoo_period = {"1d": "30d", "1h": "60d", "15m": "60d", "5m": "5d"}.get(interval, "60d")
        return fetch_yahoo(ticker, yahoo_interval, yahoo_period)

    return pd.DataFrame()


def safe_setup_state(setup_name, symbol, note):
    """Builds a clearly-labeled error/placeholder state so a single bad
    symbol never takes down the whole dashboard build."""
    return SetupState(setup_name, symbol, float("nan"), float("nan"), "N/A",
                       status="no_data", notes=note)


def build_4h_anchored_to_hour(df_1h: pd.DataFrame, target_hour: int) -> pd.DataFrame:
    """
    Resamples 1H OHLC data into 4H candles whose bin edges are anchored to
    target_hour in NEW YORK time (e.g. bins at 5,9,13,17,21,1 AM NY for
    target_hour=5, or 6,10,14,18,22,2 AM NY for target_hour=6).

    A plain df.resample("4h") uses whatever boundary pandas defaults to
    based on the data's own timezone/origin, which only coincidentally
    lines up with 5AM NY for some instruments and not others (e.g. index
    CFDs like NAS100/US30 that run on a session grid offset by an hour).
    This anchors explicitly so the right instrument gets the right candle.
    """
    if df_1h.empty:
        return pd.DataFrame()

    idx = df_1h.index
    if idx.tz is None:
        df_ny = df_1h.copy()
        df_ny.index = idx.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        df_ny = df_1h.copy()
        df_ny.index = idx.tz_convert(NY_TZ)

    try:
        resampled = df_ny.resample("4h", offset=f"{target_hour}h", origin="start_day").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
    except Exception:
        return pd.DataFrame()

    return resampled


def analyze_symbol(name, ticker):
    results = {}
    try:
        df_daily = fetch(name, "1d", "30d")
        df_15m = fetch(name, "15m", "60d")
        df_30m = fetch(name, "30m", "60d")
        df_1h = fetch(name, "1h", "60d")   # resampled to 4H below, anchored per-instrument
        df_5m = fetch(name, "5m", "5d")    # for 5m MSS/BOS during NY session

        # Setup A - Daily PDH/PDL
        if not df_daily.empty and not df_15m.empty:
            try:
                results["setup_a"] = analyze_daily_setup(df_daily, df_15m, name)
            except Exception as e:
                results["setup_a"] = safe_setup_state("Daily PDH/PDL", name, f"Analysis error: {e}")
        else:
            results["setup_a"] = safe_setup_state("Daily PDH/PDL", name, "No data available for this symbol yet.")

        # Setup B - NY CRT. Try each candidate hour for this instrument
        # (some instruments' CRT candle shifts hour day-to-day, e.g. DST
        # drift) and use whichever candidate produces the best result.
        candidate_hours = CRT_HOUR_CANDIDATES.get(name, [5])
        best_state = None

        def _candidate_score(s):
            # A genuinely live signal (confirmed AND has an entry price)
            # ranks above a merely-confirmed-but-filtered one (e.g. blocked
            # by the HTF bias/confluence filters) - that filtered state
            # isn't more useful than a plain "waiting" from another hour.
            if s.status == "mss_confirmed" and s.entry_price is not None:
                return 4
            if s.status == "mss_confirmed":
                return 2  # confirmed but filtered - still informative, but not a live signal
            return {"sweep_only": 3, "waiting": 1, "no_data": 0}.get(s.status, 0)

        for hour in candidate_hours:
            df_4h = build_4h_anchored_to_hour(df_1h, hour)
            if df_4h.empty or df_15m.empty:
                candidate_state = safe_setup_state("NY 5AM CRT", name, "No data available for this symbol yet.")
            else:
                try:
                    candidate_state = analyze_ny_crt_setup(df_4h, df_15m, name, target_hour=hour, df_daily=df_daily)
                except Exception as e:
                    candidate_state = safe_setup_state("NY 5AM CRT", name, f"Analysis error: {e}")

            if best_state is None or _candidate_score(candidate_state) > _candidate_score(best_state):
                best_state = candidate_state

        results["setup_b"] = best_state

        # 5m fully-confirmed NY CRT signal — only fires when ALL gates pass:
        # CRT range sweep → MSS → confluence → entry inside the range.
        # Only attempted during the live NY session with valid 5m data.
        ny_now = pd.Timestamp.now(tz=NY_TZ)
        current_hour_ny = ny_now.hour + ny_now.minute / 60
        if 9 <= current_hour_ny < 17 and not df_5m.empty:
            crt_high = best_state.range_high if best_state and best_state.range_high == best_state.range_high else None
            crt_low  = best_state.range_low  if best_state and best_state.range_low  == best_state.range_low  else None
            # Pass the same 4H bias used by Setup B so the trend tag is consistent
            try:
                htf_bias_5m = best_state.notes  # not the bias itself — compute it properly below
            except Exception:
                htf_bias_5m = "neutral"
            # Recompute bias from 4H data (same method as Setup B)
            try:
                from strategy_v2 import compute_htf_bias
                df_4h_for_bias = build_4h_anchored_to_hour(df_1h, candidate_hours[0])
                htf_bias_5m = compute_htf_bias(df_4h_for_bias) if not df_4h_for_bias.empty else "neutral"
            except Exception:
                htf_bias_5m = "neutral"
            try:
                results["mss_5m"] = detect_5m_mss_bos(
                    df_5m, name, crt_high, crt_low, htf_bias=htf_bias_5m
                )
            except Exception as e:
                print(f"5m confirmed signal detection error for {name}: {e}")
                results["mss_5m"] = None
        else:
            results["mss_5m"] = None

        # Setup C - Major Liquidity (15m / 30m / 1h)
        try:
            results["setup_c"] = run_major_liquidity_across_timeframes(
                df_15m=df_15m if not df_15m.empty else None,
                df_30m=df_30m if not df_30m.empty else None,
                df_1h=df_1h if not df_1h.empty else None,
                symbol=name,
            )
        except Exception as e:
            results["setup_c"] = safe_setup_state("Major Liquidity", name, f"Analysis error: {e}")

    except Exception as e:
        err_state = safe_setup_state("Error", name, f"Unexpected error: {e}")
        results["setup_a"] = err_state
        results["setup_b"] = err_state
        results["setup_c"] = err_state
        results["mss_5m"] = None

    return results


def fmt(value, decimals=4):
    """Safely formats a possibly-None numeric value for display."""
    if value is None:
        return "—"
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def render_setup_card(state: SetupState) -> str:
    # A "live" signal needs BOTH mss_confirmed status AND an actual entry
    # price - the new displacement/HTF-bias/confluence filters can leave
    # status as "mss_confirmed" while entry_price stays None (filtered out),
    # which should NOT display as a green "SIGNAL LIVE" / BUY-SELL card.
    is_live_signal = state.status == "mss_confirmed" and state.entry_price is not None
    is_filtered = state.status == "mss_confirmed" and state.entry_price is None

    if is_live_signal:
        color, label = STATUS_COLORS["mss_confirmed"], STATUS_LABELS["mss_confirmed"]
        if state.against_trend:
            label = "SIGNAL LIVE ⚠"
    elif is_filtered:
        color, label = "#555", "FILTERED — no entry"
    else:
        color = STATUS_COLORS.get(state.status, "#555")
        label = STATUS_LABELS.get(state.status, (state.status or "unknown").upper())

    levels_html = ""
    if is_live_signal:
        conf = ", ".join(state.confluence) if state.confluence else "—"
        dir_color = "#1e9e5a" if state.direction == "long" else "#c0392b"
        direction_label = "BUY" if state.direction == "long" else "SELL" if state.direction == "short" else "—"
        trend_warning = '<div class="row" style="color:#c9a227"><span>⚠ Against 4H trend</span><b></b></div>' if state.against_trend else ""
        levels_html = f"""
        <div class="levels">
          <div class="dir" style="color:{dir_color}">{direction_label}</div>
          {trend_warning}
          <div class="row"><span>Entry</span><b>{fmt(state.entry_price)}</b></div>
          <div class="row"><span>Stop Loss</span><b>{fmt(state.stop_loss)}</b></div>
          <div class="row"><span>TP1 (→ move SL to BE)</span><b>{fmt(state.tp1)}</b></div>
          <div class="row"><span>TP2</span><b>{fmt(state.tp2)}</b></div>
          <div class="row"><span>Confluence</span><b>{conf}</b></div>
        </div>
        """
    elif state.status == "sweep_only":
        side = "HIGH" if state.sweep_side == "high" else "LOW"
        levels_html = f"""
        <div class="levels">
          <div class="row"><span>Swept</span><b>{state.range_label} {side}</b></div>
          <div class="row"><span>Sweep price</span><b>{fmt(state.sweep_price)}</b></div>
        </div>
        """

    range_html = ""
    if state.range_high is not None and state.range_high == state.range_high:  # not NaN
        range_html = f'<div class="range">{state.range_label}: {fmt(state.range_low)} — {fmt(state.range_high)}</div>'

    notes = state.notes or ""

    return f"""
    <div class="setup-card">
      <div class="setup-header">
        <span class="setup-name">{state.setup_name}</span>
        <span class="status-pill" style="background:{color}">{label}</span>
      </div>
      {range_html}
      {levels_html}
      <div class="notes">{notes}</div>
    </div>
    """


def render_mss_5m_card(signal: MSSSignal) -> str:
    """
    Renders a full signal card for a confirmed 5m NY CRT setup.
    Only shown when ALL gates passed (sweep + MSS + confluence + in-range
    entry), so the card always carries a complete entry/SL/TP layout —
    identical in structure to the Setup A and Setup B signal cards.
    """
    if signal is None:
        return ""

    dir_color = "#1e9e5a" if signal.direction == "long" else "#c0392b"
    direction_label = "BUY" if signal.direction == "long" else "SELL"
    conf = ", ".join(signal.confluence) if signal.confluence else "—"
    time_str = signal.signal_time.strftime("%H:%M NY") if signal.signal_time else ""
    trend_warning = (
        '<div class="row" style="color:#c9a227"><span>⚠ Against 4H trend</span><b></b></div>'
        if signal.against_trend else ""
    )

    return f"""
    <div class="setup-card mss-card">
      <div class="setup-header">
        <span class="setup-name">NY 5AM CRT (5m confirmed)</span>
        <span class="status-pill" style="background:#1e9e5a">SIGNAL LIVE</span>
      </div>
      <div class="range">CRT Range: {fmt(signal.crt_range_low)} — {fmt(signal.crt_range_high)}</div>
      <div class="levels">
        <div class="dir" style="color:{dir_color}">{direction_label}</div>
        {trend_warning}
        <div class="row"><span>Entry</span><b>{fmt(signal.entry_price)}</b></div>
        <div class="row"><span>Stop Loss</span><b>{fmt(signal.stop_loss)}</b></div>
        <div class="row"><span>TP1 (→ move SL to BE)</span><b>{fmt(signal.tp1)}</b></div>
        <div class="row"><span>TP2</span><b>{fmt(signal.tp2)}</b></div>
        <div class="row"><span>Confluence</span><b>{conf}</b></div>
        {"<div class='row'><span>Time</span><b>" + time_str + "</b></div>" if time_str else ""}
      </div>
      <div class="notes">{signal.notes}</div>
    </div>
    """


def render_instrument_section(name, results) -> str:
    mss_card = render_mss_5m_card(results.get("mss_5m"))
    return f"""
    <div class="instrument">
      <h2>{name}</h2>
      {render_setup_card(results['setup_a'])}
      {render_setup_card(results['setup_b'])}
      {mss_card}
      {render_setup_card(results['setup_c'])}
    </div>
    """


def build_dashboard():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = []
    previous_signals = load_previous_signals()
    current_signals = {}

    for name, ticker in SYMBOLS.items():
        try:
            results = analyze_symbol(name, ticker)
        except Exception as e:
            err_state = safe_setup_state("Error", name, f"Fatal error analyzing {name}: {e}")
            results = {
                "setup_a": err_state,
                "setup_b": err_state,
                "setup_c": err_state,
                "mss_5m": None,
            }
        sections.append(render_instrument_section(name, results))

        # check setup_a, setup_b, setup_c for newly-confirmed signals
        for setup_key in ("setup_a", "setup_b", "setup_c"):
            state = results.get(setup_key)
            if state and state.status == "mss_confirmed" and state.entry_price is not None:
                key = signal_key(name, state.setup_name, state)
                current_signals[key] = True
                if key not in previous_signals:
                    print(f"New signal detected: {key} - sending Telegram alert")
                    send_telegram_alert(build_alert_message(name, state))

        # check 5m MSS/BOS alert
        mss_sig = results.get("mss_5m")
        if mss_sig is not None:
            key = mss_signal_key(mss_sig)
            current_signals[key] = True
            if key not in previous_signals:
                print(f"New 5m confirmed signal detected for {name} ({mss_sig.direction}) - sending Telegram alert")
                send_telegram_alert(build_mss_alert_message(mss_sig))

    save_signals(current_signals)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exclusive FX — Daily Dashboard</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 16px;
    background: #0d0d0f; color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}
  header {{ margin-bottom: 20px; }}
  header h1 {{ font-size: 20px; margin: 0 0 4px 0; text-align: center; }}
  header .ts {{ font-size: 12px; color: #888; }}
  .instrument {{ margin-bottom: 24px; }}
  .instrument h2 {{
    font-size: 16px; margin: 0 0 8px 0; color: #fff;
    border-bottom: 1px solid #222; padding-bottom: 6px;
  }}
  .setup-card {{
    background: #16161a; border-radius: 10px; padding: 12px 14px;
    margin-bottom: 10px; border: 1px solid #232327;
  }}
  .setup-card.mss-card {{
    border-color: #2a2a1a;
    background: #18180e;
  }}
  .setup-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
  }}
  .setup-name {{ font-size: 13px; color: #ccc; font-weight: 600; }}
  .status-pill {{
    font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 20px;
    color: #fff; letter-spacing: 0.3px;
  }}
  .range {{ font-size: 11px; color: #888; margin-bottom: 6px; }}
  .levels {{ background: #0f0f12; border-radius: 8px; padding: 8px 10px; margin: 6px 0; }}
  .levels .dir {{ font-size: 12px; font-weight: 800; margin-bottom: 4px; letter-spacing: 0.5px; }}
  .levels .row {{
    display: flex; justify-content: space-between; font-size: 12px;
    padding: 2px 0; color: #bbb;
  }}
  .levels .row b {{ color: #fff; }}
  .notes {{ font-size: 11px; color: #777; margin-top: 6px; line-height: 1.4; }}
  footer {{ font-size: 10px; color: #555; text-align: center; margin-top: 24px; padding-bottom: 20px; }}
  .refresh-btn {{
    display: flex; align-items: center; justify-content: center; gap: 8px;
    background: #1a1a1f; border: 1px solid #2a2a30; color: #eee;
    text-decoration: none; font-size: 13px; font-weight: 600;
    padding: 12px 16px; border-radius: 10px; margin-top: 10px;
  }}
  .refresh-btn:active {{ background: #232328; }}
  .refresh-note {{ font-size: 10px; color: #666; text-align: center; margin-top: 6px; line-height: 1.4; }}
</style>
</head>
<body>
<header>
  <h1>Exclusive FX — Daily Setup Dashboard</h1>
  <div class="ts">Updated {now}</div>
  <a class="refresh-btn" href="{REPO_ACTIONS_URL}" target="_blank" rel="noopener">
    ⟳ Refresh Analysis
  </a>
  <div class="refresh-note">Opens GitHub Actions — tap the green "Run workflow" button there, wait ~30s, then come back and reload this page.</div>
</header>
{''.join(sections)}
<footer>
  Setup A = Daily PDH/PDL sweep · Setup B = NY 5AM candle range (CRT) · Setup C = Major Liquidity (15m/30m/1h)<br>
  5m MSS/BOS shown during NY session (9AM–5PM NY) as early structure alert<br>
  TP1 = nearest unswept liquidity → move SL to breakeven · TP2 = next liquidity beyond<br>
  Not financial advice. Demo-test before live.
</footer>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    out_path = "index.html"
    try:
        html = build_dashboard()
        with open(out_path, "w") as f:
            f.write(html)
        print(f"Dashboard written to {out_path}")
    except Exception as e:
        print("FATAL ERROR building dashboard:")
        traceback.print_exc()
        fallback_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dashboard Error</title></head>
<body style="background:#0d0d0f;color:#eee;font-family:sans-serif;padding:20px;">
<h1>Dashboard build failed</h1>
<p>{str(e)}</p>
<pre style="white-space:pre-wrap;font-size:11px;color:#999;">{traceback.format_exc()}</pre>
</body></html>"""
        with open(out_path, "w") as f:
            f.write(fallback_html)
        raise
