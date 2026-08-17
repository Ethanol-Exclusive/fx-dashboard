"""
daily_dashboard.py

Run this every morning (or on a schedule). It pulls fresh data for
Gold, NAS100, EURUSD, USDJPY, runs both strategy setups (Daily PDH/PDL
and NY 5AM CRT), and renders a single mobile-friendly HTML page showing
exactly where each instrument stands: waiting / swept / MSS confirmed /
entry+SL+TP1+TP2 levels + confluence tags.

Usage:
    python daily_dashboard.py

Output:
    /mnt/user-data/outputs/dashboard.html   <- open this on your phone
"""

import datetime
import pandas as pd
import yfinance as yf

from strategy_v2 import analyze_daily_setup, analyze_ny_crt_setup, SetupState

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


def fetch(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def analyze_symbol(name, ticker):
    results = {}
    try:
        df_daily = fetch(ticker, "1d", "30d")
        df_15m = fetch(ticker, "15m", "60d")
        df_4h = fetch(ticker, "1h", "60d")  # resample to 4H below
        if not df_4h.empty:
            df_4h = df_4h.resample("4h").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            ).dropna()

        if not df_daily.empty and not df_15m.empty:
            results["setup_a"] = analyze_daily_setup(df_daily, df_15m, name)
        else:
            results["setup_a"] = SetupState("Daily PDH/PDL", name, float("nan"), float("nan"), "Previous Day", status="no_data")

        if not df_4h.empty and not df_15m.empty:
            results["setup_b"] = analyze_ny_crt_setup(df_4h, df_15m, name)
        else:
            results["setup_b"] = SetupState("NY 5AM CRT", name, float("nan"), float("nan"), "5AM NY Candle", status="no_data")

    except Exception as e:
        err_state = SetupState("Error", name, float("nan"), float("nan"), "N/A", status="no_data", notes=str(e))
        results["setup_a"] = err_state
        results["setup_b"] = err_state
    return results


def render_setup_card(state: SetupState) -> str:
    color = STATUS_COLORS.get(state.status, "#555")
    label = STATUS_LABELS.get(state.status, state.status.upper())

    levels_html = ""
    if state.status == "mss_confirmed":
        conf = ", ".join(state.confluence) if state.confluence else "—"
        dir_color = "#1e9e5a" if state.direction == "long" else "#c0392b"
        tp1_str = f"{state.tp1:.4f}" if state.tp1 is not None else "— (no liquidity found yet)"
        tp2_str = f"{state.tp2:.4f}" if state.tp2 is not None else "—"
        levels_html = f"""
        <div class="levels">
          <div class="dir" style="color:{dir_color}">{state.direction.upper() if state.direction else ''}</div>
          <div class="row"><span>Entry</span><b>{state.entry_price:.4f}</b></div>
          <div class="row"><span>Stop Loss</span><b>{state.stop_loss:.4f}</b></div>
          <div class="row"><span>TP1 (→ move SL to BE)</span><b>{tp1_str}</b></div>
          <div class="row"><span>TP2</span><b>{tp2_str}</b></div>
          <div class="row"><span>Confluence</span><b>{conf}</b></div>
        </div>
        """
    elif state.status == "sweep_only":
        side = "HIGH" if state.sweep_side == "high" else "LOW"
        sweep_price_str = f"{state.sweep_price:.4f}" if state.sweep_price is not None else "—"
        levels_html = f"""
        <div class="levels">
          <div class="row"><span>Swept</span><b>{state.range_label} {side}</b></div>
          <div class="row"><span>Sweep price</span><b>{sweep_price_str}</b></div>
        </div>
        """

    range_html = ""
    if state.range_high == state.range_high:  # not NaN
        range_html = f'<div class="range">{state.range_label}: {state.range_low:.4f} — {state.range_high:.4f}</div>'

    return f"""
    <div class="setup-card">
      <div class="setup-header">
        <span class="setup-name">{state.setup_name}</span>
        <span class="status-pill" style="background:{color}">{label}</span>
      </div>
      {range_html}
      {levels_html}
      <div class="notes">{state.notes}</div>
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
    for name, ticker in SYMBOLS.items():
        results = analyze_symbol(name, ticker)
        sections.append(render_instrument_section(name, results))

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
  header h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
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
    html = build_dashboard()
    out_path = "index.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to {out_path}")
