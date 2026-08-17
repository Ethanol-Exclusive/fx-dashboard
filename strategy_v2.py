"""
strategy_v2.py

Implements Exclusive's actual trading rules, as specified:

SETUP A - Daily PDH/PDL sweep
  1. Mark previous day's high (PDH) and low (PDL)
  2. Wait for a liquidity sweep of PDH or PDL (wick through, close back inside)
  3. Wait for a Market Structure Shift (MSS) confirming reversal
  4. Enter on MSS (confluence: FVG or breaker block improves the entry)
  5. Stop loss at the sweep extreme
  6. No fixed take-profit. Instead:
       TP1 = nearest UNSWEPT liquidity in trade direction
             (nearest prior swing high/low OR session high/low, whichever is closer)
             -> when hit, move stop loss to breakeven
       TP2 = next liquidity level beyond TP1
             -> trail / take remainder here or beyond

SETUP B - NY session CRT (Candle Range Theory)
  1. On the 4H chart, mark the high/low of the 5:00 AM (New York time) candle
  2. Wait for a sweep of that range on either side
  3. Wait for MSS
  4. Enter (refine entry timing on 5m/15m chart)
  5. Same SL at sweep extreme, same dynamic TP1 (breakeven)/TP2 liquidity logic

Both setups mirror for buy/sell.

This module is the shared "brain" used by both:
  - daily_dashboard.py  (live/current-state analysis you check every morning)
  - backtest_v2.py      (historical validation of these exact rules)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import pandas as pd
import numpy as np
import pytz

Direction = Literal["long", "short"]
NY_TZ = pytz.timezone("America/New_York")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class SwingPoint:
    index: int
    time: pd.Timestamp
    price: float
    kind: Literal["high", "low"]
    swept: bool = False


@dataclass
class FVG:
    index: int
    top: float
    bottom: float
    direction: Direction
    mitigated: bool = False


@dataclass
class BreakerBlock:
    index: int          # index of the origin candle (last opposite candle before impulse)
    top: float
    bottom: float
    direction: Direction   # direction breaker supports (long/short)


@dataclass
class LiquidityLevel:
    price: float
    kind: str            # "swing_high" / "swing_low" / "session_high" / "session_low" / "PDH" / "PDL"
    swept: bool = False


@dataclass
class SetupState:
    """Current live state of a strategy setup for one instrument/timeframe."""
    setup_name: str
    symbol: str
    range_high: float
    range_low: float
    range_label: str            # e.g. "Previous Day" or "5AM NY Candle"
    sweep_detected: bool = False
    sweep_side: Optional[str] = None       # "high" or "low"
    sweep_index: Optional[int] = None
    sweep_price: Optional[float] = None
    mss_confirmed: bool = False
    mss_index: Optional[int] = None
    direction: Optional[Direction] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    confluence: List[str] = field(default_factory=list)   # e.g. ["FVG", "Breaker Block"]
    status: str = "waiting"     # waiting / sweep_only / mss_confirmed / in_trade / tp1_hit / invalidated
    notes: str = ""


# --------------------------------------------------------------------------- #
# Core detection functions (shared building blocks)
# --------------------------------------------------------------------------- #
def _get_time_series(df: pd.DataFrame):
    """Robustly pulls a time series regardless of whether df has a DatetimeIndex
    or the index was reset into a column (which pandas may name 'index',
    'Date', 'Datetime', or 'Time' depending on the source)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    for candidate in ("Time", "Datetime", "Date", "index"):
        if candidate in df.columns:
            return df[candidate].values
    return np.arange(len(df))  # fallback: no real time info available


def find_swings(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    swings = []
    highs, lows = df["High"].values, df["Low"].values
    times = _get_time_series(df)
    for i in range(lookback, len(df) - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max():
            swings.append(SwingPoint(i, times[i], highs[i], "high"))
        if lows[i] == window_l.min():
            swings.append(SwingPoint(i, times[i], lows[i], "low"))
    return swings


def find_fvgs(df: pd.DataFrame) -> List[FVG]:
    fvgs = []
    highs, lows = df["High"].values, df["Low"].values
    for i in range(1, len(df) - 1):
        if highs[i - 1] < lows[i + 1]:
            fvgs.append(FVG(i, top=lows[i + 1], bottom=highs[i - 1], direction="long"))
        if lows[i - 1] > highs[i + 1]:
            fvgs.append(FVG(i, top=lows[i - 1], bottom=highs[i + 1], direction="short"))
    return fvgs


def find_breaker_blocks(df: pd.DataFrame, swings: List[SwingPoint]) -> List[BreakerBlock]:
    """
    Simplified breaker block detection: the last opposite-colored candle
    immediately before price impulsively breaks a swing point.
    - Bullish breaker: last down-candle before an up-move breaks a swing high
    - Bearish breaker: last up-candle before a down-move breaks a swing low
    """
    blocks = []
    opens, closes = df["Open"].values, df["Close"].values
    highs, lows = df["High"].values, df["Low"].values

    for s in swings:
        i = s.index
        if s.kind == "high":
            # look forward for a close beyond this high
            for j in range(i + 1, min(i + 20, len(df))):
                if closes[j] > s.price:
                    # walk back from j to find the last down-candle before the break
                    for k in range(j - 1, max(j - 6, 0), -1):
                        if closes[k] < opens[k]:
                            blocks.append(BreakerBlock(k, top=highs[k], bottom=lows[k], direction="long"))
                            break
                    break
        else:
            for j in range(i + 1, min(i + 20, len(df))):
                if closes[j] < s.price:
                    for k in range(j - 1, max(j - 6, 0), -1):
                        if closes[k] > opens[k]:
                            blocks.append(BreakerBlock(k, top=highs[k], bottom=lows[k], direction="short"))
                            break
                    break
    return blocks


def detect_sweep_and_mss(
    df: pd.DataFrame,
    range_high: float,
    range_low: float,
    swings: List[SwingPoint],
    search_start: int = 0,
) -> dict:
    """
    Scans forward from search_start looking for sweeps of range_high/range_low
    and subsequent Market Structure Shifts.

    IMPORTANT: this tracks the MOST RECENT sweep, not the first one of the day.
    Price often sweeps one side, fails to follow through, then later sweeps the
    other side and reverses for real - the dashboard should reflect whichever
    setup is currently live/actionable, not a stale one from earlier in the
    session. Each new sweep (either side) overwrites any prior untracked state.
    """
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    result = {
        "sweep_detected": False, "sweep_side": None, "sweep_index": None, "sweep_price": None,
        "mss_confirmed": False, "mss_index": None, "direction": None,
    }

    for i in range(search_start, len(df)):
        # Only count this as a NEW sweep if the previous candle hadn't already
        # poked past the level - otherwise a slow grind through the range
        # (e.g. a strong trending continuation) would re-trigger "sweep" on
        # almost every candle and falsely flip the reported direction.
        prev_high_ok = i == 0 or highs[i - 1] <= range_high
        prev_low_ok = i == 0 or lows[i - 1] >= range_low

        # Once MSS has confirmed a direction, the original range has done its
        # job - don't let a later retest of that same range flip the signal.
        # Only a sweep of the OPPOSITE side (a genuine new setup) should override.
        if result["mss_confirmed"]:
            if result["direction"] == "long" and lows[i] < range_low and closes[i] > range_low and prev_low_ok:
                pass  # same-side retest, ignore
            elif result["direction"] == "short" and highs[i] > range_high and closes[i] < range_high and prev_high_ok:
                pass  # same-side retest, ignore
            elif result["direction"] == "long" and highs[i] > range_high and closes[i] < range_high and prev_high_ok:
                result.update(sweep_detected=True, sweep_side="high", sweep_index=i, sweep_price=highs[i],
                               mss_confirmed=False, mss_index=None, direction=None)
            elif result["direction"] == "short" and lows[i] < range_low and closes[i] > range_low and prev_low_ok:
                result.update(sweep_detected=True, sweep_side="low", sweep_index=i, sweep_price=lows[i],
                               mss_confirmed=False, mss_index=None, direction=None)
        else:
            if lows[i] < range_low and closes[i] > range_low and prev_low_ok:
                result.update(sweep_detected=True, sweep_side="low", sweep_index=i, sweep_price=lows[i],
                               mss_confirmed=False, mss_index=None, direction=None)
            elif highs[i] > range_high and closes[i] < range_high and prev_high_ok:
                result.update(sweep_detected=True, sweep_side="high", sweep_index=i, sweep_price=highs[i],
                               mss_confirmed=False, mss_index=None, direction=None)

        if result["sweep_detected"] and not result["mss_confirmed"]:
            sweep_i = result["sweep_index"]
            if i <= sweep_i:
                continue
            if result["sweep_side"] == "low":
                prior_highs = [s for s in swings if s.kind == "high" and s.index < i]
                if prior_highs:
                    structure_level = prior_highs[-1].price
                    if closes[i] > structure_level:
                        result.update(mss_confirmed=True, mss_index=i, direction="long")
            else:
                prior_lows = [s for s in swings if s.kind == "low" and s.index < i]
                if prior_lows:
                    structure_level = prior_lows[-1].price
                    if closes[i] < structure_level:
                        result.update(mss_confirmed=True, mss_index=i, direction="short")
            # NOTE: no `break` here anymore - we keep scanning to the end of the
            # data so a later opposite sweep can still override this if it happens

    return result


def nearest_unswept_liquidity(
    current_price: float,
    direction: Direction,
    swings: List[SwingPoint],
    session_high: Optional[float],
    session_low: Optional[float],
    up_to_index: int,
) -> List[LiquidityLevel]:
    """
    Builds a sorted list of unswept liquidity levels beyond current_price
    in the trade's direction (closest first). Includes swing highs/lows
    AND session high/low as candidate liquidity, per Exclusive's rule.
    """
    candidates: List[LiquidityLevel] = []

    relevant_swings = [s for s in swings if s.index <= up_to_index]
    if direction == "long":
        for s in relevant_swings:
            if s.kind == "high" and s.price > current_price and not s.swept:
                candidates.append(LiquidityLevel(s.price, "swing_high"))
        if session_high and session_high > current_price:
            candidates.append(LiquidityLevel(session_high, "session_high"))
        candidates.sort(key=lambda c: c.price)
    else:
        for s in relevant_swings:
            if s.kind == "low" and s.price < current_price and not s.swept:
                candidates.append(LiquidityLevel(s.price, "swing_low"))
        if session_low and session_low < current_price:
            candidates.append(LiquidityLevel(session_low, "session_low"))
        candidates.sort(key=lambda c: c.price, reverse=True)

    # De-duplicate levels that are effectively the same price - overlapping
    # swing-detection windows can flag several adjacent candles as separate
    # "swing highs/lows" when they're really the same liquidity level, which
    # previously caused TP1 and TP2 to show identical numbers.
    deduped: List[LiquidityLevel] = []
    epsilon = max(abs(current_price) * 0.0001, 1e-6)  # ~0.01% of price, scales across instruments
    for c in candidates:
        if not any(abs(c.price - d.price) < epsilon for d in deduped):
            deduped.append(c)
    candidates = deduped

    return candidates


def check_fvg_breaker_confluence(
    entry_index: int,
    direction: Direction,
    fvgs: List[FVG],
    breakers: List[BreakerBlock],
    lookback: int = 15,
) -> List[str]:
    """Checks whether an FVG or breaker block sits near the entry, for confluence tagging."""
    tags = []
    recent_fvgs = [f for f in fvgs if f.direction == direction and entry_index - lookback <= f.index <= entry_index]
    if recent_fvgs:
        tags.append("FVG")
    recent_breakers = [b for b in breakers if b.direction == direction and entry_index - lookback <= b.index <= entry_index]
    if recent_breakers:
        tags.append("Breaker Block")
    return tags


# --------------------------------------------------------------------------- #
# High-level: build a SetupState for "Setup A" (Daily PDH/PDL)
# --------------------------------------------------------------------------- #
def analyze_daily_setup(df_daily: pd.DataFrame, df_intraday: pd.DataFrame, symbol: str) -> SetupState:
    """
    df_daily: daily OHLC (DatetimeIndex), used to get PDH/PDL
    df_intraday: finer timeframe (e.g. 15m) OHLC used to detect sweep/MSS live
    """
    if len(df_daily) < 2:
        return SetupState("Daily PDH/PDL", symbol, np.nan, np.nan, "Previous Day", status="no_data")

    prev_day = df_daily.iloc[-2]
    pdh, pdl = prev_day["High"], prev_day["Low"]

    state = SetupState("Daily PDH/PDL", symbol, pdh, pdl, "Previous Day")

    # restrict intraday df to candles that occurred after prev day's close (today's session)
    today_start = df_daily.index[-1]

    # align timezone-awareness between today_start and df_intraday's index
    # before comparing, otherwise pandas raises on tz-naive vs tz-aware
    intraday_idx = df_intraday.index
    if intraday_idx.tz is None and today_start.tzinfo is not None:
        today_start_cmp = today_start.tz_localize(None)
    elif intraday_idx.tz is not None and today_start.tzinfo is None:
        today_start_cmp = today_start.tz_localize(intraday_idx.tz)
    elif intraday_idx.tz is not None and today_start.tzinfo is not None:
        today_start_cmp = today_start.tz_convert(intraday_idx.tz)
    else:
        today_start_cmp = today_start

    intraday_today = df_intraday[df_intraday.index >= today_start_cmp]
    if len(intraday_today) < 10:
        state.status = "waiting"
        state.notes = "Not enough of today's session data yet."
        return state

    intraday_today = intraday_today.reset_index()
    swings = find_swings(intraday_today, lookback=3)
    fvgs = find_fvgs(intraday_today)
    breakers = find_breaker_blocks(intraday_today, swings)

    scan = detect_sweep_and_mss(intraday_today, pdh, pdl, swings)

    state.sweep_detected = scan["sweep_detected"]
    state.sweep_side = scan["sweep_side"]
    state.sweep_index = scan["sweep_index"]
    state.sweep_price = scan["sweep_price"]
    state.mss_confirmed = scan["mss_confirmed"]
    state.mss_index = scan["mss_index"]
    state.direction = scan["direction"]

    if not state.sweep_detected:
        state.status = "waiting"
        state.notes = "No sweep of PDH/PDL yet today."
        return state

    if not state.mss_confirmed:
        state.status = "sweep_only"
        state.notes = f"Sweep of PD{'L' if state.sweep_side=='low' else 'H'} detected - waiting for MSS confirmation."
        return state

    # MSS confirmed -> entry logic
    entry_idx = state.mss_index
    entry_price = intraday_today["Close"].values[entry_idx]
    state.entry_price = entry_price
    state.stop_loss = state.sweep_price
    state.confluence = check_fvg_breaker_confluence(entry_idx, state.direction, fvgs, breakers)

    liquidity = nearest_unswept_liquidity(
        entry_price, state.direction, swings, session_high=None, session_low=None, up_to_index=entry_idx
    )
    if len(liquidity) >= 1:
        state.tp1 = liquidity[0].price
    if len(liquidity) >= 2:
        state.tp2 = liquidity[1].price

    state.status = "mss_confirmed"
    state.notes = (
        f"MSS confirmed {state.direction.upper()}. Entry ~{entry_price:.4f}, "
        f"SL {state.stop_loss:.4f}, TP1 {state.tp1}, TP2 {state.tp2}. "
        f"Confluence: {', '.join(state.confluence) if state.confluence else 'none'}."
    )
    return state


# --------------------------------------------------------------------------- #
# High-level: build a SetupState for "Setup B" (NY 5AM CRT)
# --------------------------------------------------------------------------- #
def get_5am_ny_candle_range(df_4h: pd.DataFrame) -> Optional[dict]:
    """
    df_4h: 4H OHLC with a tz-aware or naive DatetimeIndex (assumed UTC if naive).
    Finds the most recent 4H candle whose NY-time hour is 5 (the 5AM NY candle),
    and returns its high/low along with the time it FINISHES printing (its close,
    ~9AM NY for a 4H candle starting at 5AM) - entries should only be hunted for
    after that point, once the range is actually locked in.
    """
    idx = df_4h.index
    if idx.tz is None:
        idx_ny = idx.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        idx_ny = idx.tz_convert(NY_TZ)

    hours = idx_ny.hour
    candidates = df_4h[(hours >= 4) & (hours <= 7)]  # 4H bucket containing 5AM NY
    if candidates.empty:
        return None
    last = candidates.iloc[-1]
    open_time = candidates.index[-1]
    close_time = open_time + pd.Timedelta(hours=4)  # when this 4H candle finishes printing
    return {"high": last["High"], "low": last["Low"], "time": open_time, "close_time": close_time}


def analyze_ny_crt_setup(df_4h: pd.DataFrame, df_intraday: pd.DataFrame, symbol: str) -> SetupState:
    """
    df_4h: 4H OHLC, used to find the 5AM NY candle range
    df_intraday: 5m or 15m OHLC used to detect the sweep/MSS and refine entry

    Per the rule: mark the 5AM-9AM NY range, but only start looking for a
    sweep/MSS entry AFTER that candle has finished printing (i.e. from 9AM
    NY onward) - not while it's still forming.
    """
    candle = get_5am_ny_candle_range(df_4h)
    if candle is None:
        return SetupState("NY 5AM CRT", symbol, np.nan, np.nan, "5AM NY Candle", status="no_data")

    range_high, range_low = candle["high"], candle["low"]
    state = SetupState("NY 5AM CRT", symbol, range_high, range_low, "5AM NY Candle")

    session_start = candle["close_time"]  # entries only hunted from candle CLOSE (~9AM NY), not open (5AM)

    # align timezone-awareness between session_start and df_intraday's index
    # before comparing, otherwise pandas raises on tz-naive vs tz-aware
    intraday_idx = df_intraday.index
    if intraday_idx.tz is None and session_start.tzinfo is not None:
        session_start_cmp = session_start.tz_localize(None)
    elif intraday_idx.tz is not None and session_start.tzinfo is None:
        session_start_cmp = session_start.tz_localize(intraday_idx.tz)
    elif intraday_idx.tz is not None and session_start.tzinfo is not None:
        session_start_cmp = session_start.tz_convert(intraday_idx.tz)
    else:
        session_start_cmp = session_start

    intraday_after = df_intraday[df_intraday.index >= session_start_cmp]
    if len(intraday_after) < 5:
        state.status = "waiting"
        state.notes = "5AM NY candle range marked (5-9AM NY) - waiting for the candle to finish printing before looking for entries."
        return state

    intraday_after = intraday_after.reset_index()
    swings = find_swings(intraday_after, lookback=3)
    fvgs = find_fvgs(intraday_after)
    breakers = find_breaker_blocks(intraday_after, swings)

    scan = detect_sweep_and_mss(intraday_after, range_high, range_low, swings)

    state.sweep_detected = scan["sweep_detected"]
    state.sweep_side = scan["sweep_side"]
    state.sweep_index = scan["sweep_index"]
    state.sweep_price = scan["sweep_price"]
    state.mss_confirmed = scan["mss_confirmed"]
    state.mss_index = scan["mss_index"]
    state.direction = scan["direction"]

    if not state.sweep_detected:
        state.status = "waiting"
        state.notes = "No sweep of the 5AM NY candle range yet."
        return state

    if not state.mss_confirmed:
        state.status = "sweep_only"
        state.notes = "Sweep of 5AM range detected - waiting for MSS confirmation (check 5m/15m for entry timing)."
        return state

    entry_idx = state.mss_index
    entry_price = intraday_after["Close"].values[entry_idx]
    state.entry_price = entry_price
    state.stop_loss = state.sweep_price
    state.confluence = check_fvg_breaker_confluence(entry_idx, state.direction, fvgs, breakers)

    liquidity = nearest_unswept_liquidity(
        entry_price, state.direction, swings,
        session_high=range_high, session_low=range_low, up_to_index=entry_idx
    )
    if len(liquidity) >= 1:
        state.tp1 = liquidity[0].price
    if len(liquidity) >= 2:
        state.tp2 = liquidity[1].price

    state.status = "mss_confirmed"
    state.notes = (
        f"MSS confirmed {state.direction.upper()} off 5AM NY range. Entry ~{entry_price:.4f}, "
        f"SL {state.stop_loss:.4f}, TP1 {state.tp1}, TP2 {state.tp2}. "
        f"Confluence: {', '.join(state.confluence) if state.confluence else 'none'}."
    )
    return state
