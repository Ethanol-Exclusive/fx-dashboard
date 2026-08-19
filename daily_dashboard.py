"""
daily_dashboard.py

Run this every morning (or on a schedule). It pulls fresh data for all
configured instruments, runs both strategy setups (Daily PDH/PDL and
NY 5AM CRT), and renders a single mobile-friendly HTML page showing
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

from strategy_v2 import analyze_daily_setup, analyze_ny_crt_setup, SetupState, NY_TZ

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


def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured (missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets) - skipping alert.")
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

    tp1_line = f"TP1 (→ BE): `{state.tp1:.5f}`\n" if state.tp1 is not None else "TP1: no liquidity found yet\n"
    tp2_line = f"TP2: `{state.tp2:.5f}`\n" if state.tp2 is not None else ""

    return (
        f"{emoji} *{direction_word} — {symbol}*\n"
        f"_{state.setup_name}_\n\n"
        f"Entry: `{state.entry_price:.5f}`\n"
        f"Stop Loss: `{state.stop_loss:.5f}`\n"
        f"{tp1_line}"
        f"{tp2_line}"
        f"Confluence: {conf}"
    )


# Secondary/fallback data source (Twelve Data) - only used when the primary
# source (Yahoo Finance) returns nothing for a given fetch, e.g. a delayed
# or missing candle causing a real setup to go undetected. Inactive unless
# TWELVE_DATA_API_KEY is set, so this is a no-op until configured.
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
TWELVE_DATA_SYMBOLS = {
    "GOLD":   "XAU/USD",
    "NAS100": "NDX",
    "EURUSD": "EUR/USD",
    "USDJPY": "USD/JPY",
    "GBPUSD": "GBP/USD",
    "BTCUSD": "BTC/USD",
    "AUDUSD": "AUD/USD",
    "US30":   "DJI",
    "USDCAD": "USD/CAD",
    "NZDUSD": "NZD/USD",
    "GER30":  "DAX",
}
YFINANCE_TO_NAME = {v: k for k, v in SYMBOLS.items()}


def fetch_twelvedata_fallback(name, interval, outputsize):
    """Fallback fetch via Twelve Data, used only when Yahoo returns empty.
    interval: Twelve Data codes - "1day", "1h", "15min" """
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()
    td_symbol = TWELVE_DATA_SYMBOLS.get(name)
    if not td_symbol:
        return pd.DataFrame()
    try:
        resp = requests.get(f"{TWELVE_DATA_BASE_URL}/time_series", params={
            "symbol": td_symbol, "interval": interval, "outputsize": outputsize,
            "apikey": TWELVE_DATA_API_KEY, "timezone": "UTC",
        }, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    if data.get("status") == "error":
        return pd.DataFrame()

    rows = []
    for v in data.get("values", []):
        try:
            t = pd.Timestamp(v["datetime"])
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            rows.append({"time": t, "Open": float(v["open"]), "High": float(v["high"]),
                         "Low": float(v["low"]), "Close": float(v["close"])})
        except (KeyError, ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("time").sort_index()


def fetch(symbol, interval, period):
    """Pulls OHLC data for one symbol from Yahoo Finance (primary). If that
    comes back empty, falls back to Twelve Data (if TWELVE_DATA_API_KEY is
    configured) rather than silently missing a setup because of one
    source's gap or delay. Returns an empty DataFrame (not an exception) if
    both sources fail, so callers can check .empty rather than needing a
    try/except at every call site."""
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    except Exception:
        df = None

    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        required = {"Open", "High", "Low", "Close"}
        if required.issubset(set(df.columns)):
            cleaned = df.dropna(subset=["Open", "High", "Low", "Close"])
            if not cleaned.empty:
                return cleaned

    # Primary source came back empty/unusable - try the fallback
    name = YFINANCE_TO_NAME.get(symbol)
    if name and TWELVE_DATA_API_KEY:
        td_interval = {"1d": "1day", "1h": "1h", "15m": "15min"}.get(interval)
        td_outputsize = {"1d": 30, "1h": 500, "15m": 2000}.get(interval, 500)
        if td_interval:
            print(f"Yahoo Finance returned nothing for {symbol} ({interval}) - trying Twelve Data fallback.")
            fallback_df = fetch_twelvedata_fallback(name, td_interval, td_outputsize)
            if not fallback_df.empty:
                return fallback_df

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
        df_daily = fetch(ticker, "1d", "30d")
        df_15m = fetch(ticker, "15m", "60d")
        df_1h = fetch(ticker, "1h", "60d")  # resampled to 4H below, anchored per-instrument

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
        status_priority = {"mss_confirmed": 3, "sweep_only": 2, "waiting": 1, "no_data": 0}
        best_state = None

        for hour in candidate_hours:
            df_4h = build_4h_anchored_to_hour(df_1h, hour)
            if df_4h.empty or df_15m.empty:
                candidate_state = safe_setup_state("NY 5AM CRT", name, "No data available for this symbol yet.")
            else:
                try:
                    candidate_state = analyze_ny_crt_setup(df_4h, df_15m, name, target_hour=hour)
                except Exception as e:
                    candidate_state = safe_setup_state("NY 5AM CRT", name, f"Analysis error: {e}")

            if best_state is None:
                best_state = candidate_state
            else:
                current_priority = status_priority.get(candidate_state.status, 0)
                best_priority = status_priority.get(best_state.status, 0)
                if current_priority > best_priority:
                    best_state = candidate_state

        results["setup_b"] = best_state

    except Exception as e:
        err_state = safe_setup_state("Error", name, f"Unexpected error: {e}")
        results["setup_a"] = err_state
        results["setup_b"] = err_state

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
    color = STATUS_COLORS.get(state.status, "#555")
    label = STATUS_LABELS.get(state.status, (state.status or "unknown").upper())

    levels_html = ""
    if state.status == "mss_confirmed":
        conf = ", ".join(state.confluence) if state.confluence else "—"
        dir_color = "#1e9e5a" if state.direction == "long" else "#c0392b"
        direction_label = "BUY" if state.direction == "long" else "SELL" if state.direction == "short" else "—"
        levels_html = f"""
        <div class="levels">
          <div class="dir" style="color:{dir_color}">{direction_label}</div>
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


def render_instrument_section(name, results) -> str:
    return f"""
    <div class="instrument">
      <h2>{name}</h2>
      {render_setup_card(results['setup_a'])}
      {render_setup_card(results['setup_b'])}
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
            results = {"setup_a": err_state, "setup_b": err_state}
        sections.append(render_instrument_section(name, results))

        # check both setups for newly-confirmed signals worth notifying about
        for setup_key in ("setup_a", "setup_b"):
            state = results.get(setup_key)
            if state and state.status == "mss_confirmed" and state.entry_price is not None:
                key = signal_key(name, state.setup_name, state)
                current_signals[key] = True
                if key not in previous_signals:
                    print(f"New signal detected: {key} - sending Telegram alert")
                    send_telegram_alert(build_alert_message(name, state))

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
  Setup A = Daily PDH/PDL sweep · Setup B = NY 5AM candle range (CRT)<br>
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
