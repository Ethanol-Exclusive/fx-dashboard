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
class OrderBlock:
    index: int          # index of the origin candle (last opposite candle before impulsive displacement)
    top: float
    bottom: float
    direction: Direction   # direction order block supports (long/short)


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
    against_trend: bool = False   # True if the signal fired despite opposing the 4H bias
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


def find_order_blocks(df: pd.DataFrame, min_displacement_ratio: float = 0.5) -> List[OrderBlock]:
    """
    General order block detection: the last opposite-colored candle
    immediately before an impulsive displacement move, regardless of
    whether that move breaks a specific swing point (unlike breaker
    blocks, which specifically require a structure break).
    - Bullish order block: last down-candle before a strong up-displacement candle
    - Bearish order block: last up-candle before a strong down-displacement candle

    A move counts as "impulsive displacement" using the same body-vs-recent-
    average-range measure used for the MSS displacement filter.
    """
    blocks = []
    opens, closes = df["Open"].values, df["Close"].values
    highs, lows = df["High"].values, df["Low"].values

    for i in range(1, len(df)):
        body_size = abs(closes[i] - opens[i])
        lookback_start = max(0, i - 14)
        recent_ranges = highs[lookback_start:i] - lows[lookback_start:i]
        avg_range = recent_ranges.mean() if len(recent_ranges) > 0 else 0
        has_displacement = avg_range > 0 and body_size >= min_displacement_ratio * avg_range * 2  # stronger bar than plain MSS displacement

        if not has_displacement:
            continue

        is_bullish_displacement = closes[i] > opens[i]
        origin = i - 1

        if is_bullish_displacement and closes[origin] < opens[origin]:
            blocks.append(OrderBlock(origin, top=highs[origin], bottom=lows[origin], direction="long"))
        elif not is_bullish_displacement and closes[origin] > opens[origin]:
            blocks.append(OrderBlock(origin, top=highs[origin], bottom=lows[origin], direction="short"))

    return blocks


def detect_sweep_and_mss(
    df: pd.DataFrame,
    range_high: float,
    range_low: float,
    swings: List[SwingPoint],
    search_start: int = 0,
    min_displacement_ratio: float = 0.5,
) -> dict:
    """
    Scans forward from search_start looking for sweeps of range_high/range_low
    and subsequent Market Structure Shifts.

    IMPORTANT: this tracks the MOST RECENT sweep, not the first one of the day.
    Price often sweeps one side, fails to follow through, then later sweeps the
    other side and reverses for real - the dashboard should reflect whichever
    setup is currently live/actionable, not a stale one from earlier in the
    session. Each new sweep (either side) overwrites any prior untracked state.

    min_displacement_ratio: an MSS candle must have a real body, not just a
    marginal close past structure - its body size (abs(close-open)) must be
    at least this fraction of the recent average candle range (a simple
    ATR-like measure over the prior 14 candles). A weak, barely-there close
    is a much lower-conviction break of structure than one with genuine
    displacement behind it. Set to 0 to disable this filter.
    """
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    opens = df["Open"].values if "Open" in df.columns else closes.copy()
    result = {
        "sweep_detected": False, "sweep_side": None, "sweep_index": None, "sweep_price": None,
        "mss_confirmed": False, "mss_index": None, "direction": None, "invalidated": False,
    }

    for i in range(search_start, len(df)):
        if result["invalidated"]:
            break  # opposing liquidity already reached - this range is done for the session

        # Only count this as a NEW sweep if the previous candle hadn't already
        # poked past the level - otherwise a slow grind through the range
        # (e.g. a strong trending continuation) would re-trigger "sweep" on
        # almost every candle and falsely flip the reported direction.
        prev_high_ok = i == 0 or highs[i - 1] <= range_high
        prev_low_ok = i == 0 or lows[i - 1] >= range_low

        # Once MSS has confirmed a direction, we're targeting the OPPOSING
        # liquidity (e.g. sweep of PDH -> short -> target PDL). Once price
        # actually reaches that opposing level, the setup has played out -
        # stop looking for anything further on this range, rather than
        # starting a brand new setup in the other direction.
        if result["mss_confirmed"]:
            if result["direction"] == "long" and highs[i] >= range_high:
                result["invalidated"] = True
            elif result["direction"] == "short" and lows[i] <= range_low:
                result["invalidated"] = True
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

            # displacement check: this candle's body vs recent average range
            body_size = abs(closes[i] - opens[i])
            lookback_start = max(0, i - 14)
            recent_ranges = highs[lookback_start:i] - lows[lookback_start:i]
            avg_range = recent_ranges.mean() if len(recent_ranges) > 0 else 0
            has_displacement = (avg_range == 0) or (body_size >= min_displacement_ratio * avg_range)

            if result["sweep_side"] == "low":
                prior_highs = [s for s in swings if s.kind == "high" and s.index < i]
                if prior_highs:
                    structure_level = prior_highs[-1].price
                    if closes[i] > structure_level and has_displacement:
                        result.update(mss_confirmed=True, mss_index=i, direction="long")
            else:
                prior_lows = [s for s in swings if s.kind == "low" and s.index < i]
                if prior_lows:
                    structure_level = prior_lows[-1].price
                    if closes[i] < structure_level and has_displacement:
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


def compute_htf_bias(df_daily: pd.DataFrame, lookback: int = 10) -> str:
    """
    A simple higher-timeframe trend bias from daily closes: compares the
    most recent CLOSED day's close against the average close over the
    prior `lookback` days. Used to filter out counter-trend setups - e.g.
    skip a long signal if the daily trend is clearly bearish.

    Returns "bullish", "bearish", or "neutral" (not enough data, or too
    close to call either way).
    """
    if len(df_daily) < 3:
        return "neutral"

    closes = df_daily["Close"].values
    recent_closes = closes[-(lookback + 1):-1] if len(closes) > lookback else closes[:-1]
    if len(recent_closes) == 0:
        return "neutral"

    last_close = closes[-1]
    avg_close = recent_closes.mean()
    if avg_close == 0:
        return "neutral"

    pct_diff = (last_close - avg_close) / avg_close
    if pct_diff > 0.001:   # >0.1% above the recent average
        return "bullish"
    elif pct_diff < -0.001:
        return "bearish"
    return "neutral"


def bias_allows_direction(bias: str, direction: Direction) -> bool:
    """A long is filtered out only when bias is clearly bearish, and vice
    versa - "neutral" doesn't block either direction, since it just means
    there's no strong opposing trend to worry about."""
    if bias == "bearish" and direction == "long":
        return False
    if bias == "bullish" and direction == "short":
        return False
    return True


def find_entry_after_range_reentry(
    df: pd.DataFrame,
    mss_index: int,
    direction: Direction,
    range_low: float,
    range_high: float,
):
    """
    Per the rule: entries should be taken INSIDE the PDH/PDL or CRT range,
    relying on the 15m MSS/BOS for confirmation. If the MSS/BOS candle's
    close is already outside the range (price broke away before structure
    even confirmed), don't enter out there - wait for price to come back
    inside the range first, then use that as the entry point.

    Returns (entry_index, entry_price) if a valid in-range entry point is
    found, or (None, None) if the MSS close was already inside the range
    (use it directly) or if price hasn't come back inside yet.
    """
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    mss_close = closes[mss_index]
    if range_low <= mss_close <= range_high:
        return None, None  # already inside the range - caller uses mss_index directly, no change needed

    # MSS confirmed while price was already outside the range - scan forward
    # for the first candle where price comes back inside the range
    for j in range(mss_index + 1, len(df)):
        if lows[j] <= range_high and highs[j] >= range_low:
            return j, closes[j]

    return "pending", None  # sentinel: MSS confirmed, but still waiting for re-entry


def check_fvg_breaker_confluence(
    entry_index: int,
    direction: Direction,
    fvgs: List[FVG],
    breakers: List[BreakerBlock],
    lookback: int = 15,
    order_blocks: Optional[List[OrderBlock]] = None,
) -> List[str]:
    """Checks whether an FVG, breaker block, or order block sits near the entry, for confluence tagging."""
    tags = []
    recent_fvgs = [f for f in fvgs if f.direction == direction and entry_index - lookback <= f.index <= entry_index]
    if recent_fvgs:
        tags.append("FVG")
    recent_breakers = [b for b in breakers if b.direction == direction and entry_index - lookback <= b.index <= entry_index]
    if recent_breakers:
        tags.append("Breaker Block")
    if order_blocks:
        recent_order_blocks = [o for o in order_blocks if o.direction == direction and entry_index - lookback <= o.index <= entry_index]
        if recent_order_blocks:
            tags.append("Order Block")
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

    # Which row is "yesterday" depends on whether df_daily's last row is
    # today's still-forming candle or already a fully-closed day - this
    # differs by data source. Yahoo Finance includes today's in-progress
    # candle as the last row (so yesterday = iloc[-2]). OANDA's feed only
    # includes fully-closed candles (so its last row IS already yesterday,
    # making yesterday = iloc[-1] - using iloc[-2] there would silently
    # grab TWO days ago instead).
    #
    # Rather than compare exact calendar dates (fragile - depends on exactly
    # how each source labels/timezones its daily timestamp), check how
    # recently the last row's timestamp occurred: if it's within the last
    # ~26 hours, it's still today's in-progress candle; if it's older than
    # that, it must already be a fully-closed day.
    last_row_time = df_daily.index[-1]
    now_utc = pd.Timestamp.now(tz="UTC")
    last_row_time_utc = last_row_time.tz_convert("UTC") if last_row_time.tzinfo else last_row_time.tz_localize("UTC")
    hours_since_last_row = (now_utc - last_row_time_utc).total_seconds() / 3600

    if hours_since_last_row < 26:
        prev_day = df_daily.iloc[-2]   # last row is recent/still forming - yesterday is one back
        last_row_is_today = True
    else:
        prev_day = df_daily.iloc[-1]   # last row is already an old, closed day - that IS yesterday
        last_row_is_today = False

    pdh, pdl = prev_day["High"], prev_day["Low"]

    state = SetupState("Daily PDH/PDL", symbol, pdh, pdl, "Previous Day")

    # restrict intraday df to candles that occurred after prev day's close
    # (today's session). If the daily series' last row is already a closed
    # past day (OANDA case), "today" starts the day AFTER that row, not at
    # that row's own timestamp.
    if last_row_is_today:
        today_start = df_daily.index[-1]
    else:
        today_start = prev_day.name + pd.Timedelta(days=1)

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
    order_blocks = find_order_blocks(intraday_today)

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

    # HTF bias: derived from the 4H timeframe (built from the full intraday
    # history) rather than daily, per instruction to weight 4H more heavily
    # than daily for trend bias. A signal against the bias is NOT filtered
    # out - it still fires, but gets marked "Against Trend".
    try:
        df_4h_bias = df_intraday.resample("4h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        bias = compute_htf_bias(df_4h_bias)
    except Exception:
        bias = "neutral"
    state.against_trend = not bias_allows_direction(bias, state.direction)

    # MSS confirmed -> entry logic. Per the rule, entries are taken INSIDE
    # the PDH/PDL range - if the MSS candle closed already outside the
    # range, wait for price to come back inside before entering.
    reentry_idx, reentry_price = find_entry_after_range_reentry(
        intraday_today, state.mss_index, state.direction, state.range_low, state.range_high
    )

    if reentry_idx == "pending":
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()}, but price broke outside the PDH/PDL range before "
            f"confirming - waiting for price to return inside the range before entering."
        )
        return state
    elif reentry_idx is not None:
        entry_idx = reentry_idx
        entry_price = reentry_price
    else:
        entry_idx = state.mss_index
        entry_price = intraday_today["Close"].values[entry_idx]

    confluence = check_fvg_breaker_confluence(entry_idx, state.direction, fvgs, breakers, order_blocks=order_blocks)
    if not confluence:
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()} and price is inside the range, but no FVG or breaker "
            f"block confluence found yet - waiting for confluence before entering."
        )
        return state

    state.entry_price = entry_price
    state.stop_loss = state.sweep_price
    state.confluence = confluence

    liquidity = nearest_unswept_liquidity(
        entry_price, state.direction, swings, session_high=None, session_low=None, up_to_index=entry_idx
    )
    if len(liquidity) >= 1:
        state.tp1 = liquidity[0].price
    if len(liquidity) >= 2:
        state.tp2 = liquidity[1].price

    state.status = "mss_confirmed"
    trend_tag = " [AGAINST TREND]" if state.against_trend else ""
    state.notes = (
        f"MSS confirmed {state.direction.upper()}{trend_tag}. Entry ~{entry_price:.4f}, "
        f"SL {state.stop_loss:.4f}, TP1 {state.tp1}, TP2 {state.tp2}. "
        f"Confluence: {', '.join(state.confluence) if state.confluence else 'none'}."
    )
    return state


# --------------------------------------------------------------------------- #
# High-level: build a SetupState for "Setup B" (NY 5AM CRT)
# --------------------------------------------------------------------------- #
def get_5am_ny_candle_range(df_4h: pd.DataFrame, target_hour: int = 5) -> Optional[dict]:
    """
    df_4h: 4H OHLC with a tz-aware or naive DatetimeIndex (assumed UTC if naive).
    Finds the most recent 4H candle whose NY-time hour matches target_hour
    (defaults to 5AM for forex; some index CFDs like NAS100/US30 run on a
    different session grid where the equivalent candle opens at 6AM instead -
    pass target_hour=6 for those), and returns its high/low along with the
    time it FINISHES printing (its close, target_hour+4 NY) - entries should
    only be hunted for after that point, once the range is actually locked in.
    """
    idx = df_4h.index
    if idx.tz is None:
        idx_ny = idx.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        idx_ny = idx.tz_convert(NY_TZ)

    hours = idx_ny.hour
    # widen the search window slightly around the target hour to tolerate
    # minor candle-boundary drift between data sources
    candidates = df_4h[(hours >= target_hour - 1) & (hours <= target_hour + 2)]
    if candidates.empty:
        return None
    last = candidates.iloc[-1]
    open_time = candidates.index[-1]
    close_time = open_time + pd.Timedelta(hours=4)  # when this 4H candle finishes printing
    return {"high": last["High"], "low": last["Low"], "time": open_time, "close_time": close_time}


def analyze_ny_crt_setup(df_4h: pd.DataFrame, df_intraday: pd.DataFrame, symbol: str, target_hour: int = 5, df_daily: Optional[pd.DataFrame] = None) -> SetupState:
    """
    df_4h: 4H OHLC, used to find the CRT candle range
    df_intraday: 5m or 15m OHLC used to detect the sweep/MSS and refine entry
    target_hour: NY hour the CRT candle opens on. Defaults to 5AM (forex).
        Some index CFDs (NAS100/US30) run on a session grid offset by an
        hour - pass target_hour=6 for those instruments.
    df_daily: daily OHLC, used for the higher-timeframe bias filter. If not
        provided, the bias filter is skipped (treated as neutral).

    Per the rule: mark the CRT range, but only start looking for a
    sweep/MSS entry AFTER that candle has finished printing (4 hours after
    it opens), and only while it is CURRENTLY the NY trading session
    (candle-close through 5PM NY) in real time - not just historically.
    Outside the live NY session right now, this returns a clear
    "outside session" status instead of surfacing a stale prior-day signal.
    """
    session_open_hour = (target_hour + 4) % 24  # e.g. 9AM for a 5AM candle
    session_close_hour = 17  # 5PM NY, standard forex session close

    now_ny = pd.Timestamp.now(tz=NY_TZ)
    current_hour = now_ny.hour + now_ny.minute / 60

    in_live_session = session_open_hour <= current_hour < session_close_hour
    if not in_live_session:
        return SetupState(
            "NY 5AM CRT", symbol, np.nan, np.nan, "5AM NY Candle",
            status="waiting",
            notes=f"Outside NY session right now ({now_ny.strftime('%H:%M')} NY) - "
                  f"this setup only looks for entries between {session_open_hour}:00 and {session_close_hour}:00 NY."
        )

    candle = get_5am_ny_candle_range(df_4h, target_hour=target_hour)
    if candle is None:
        return SetupState("NY 5AM CRT", symbol, np.nan, np.nan, "5AM NY Candle", status="no_data")

    range_high, range_low = candle["high"], candle["low"]
    label = f"{target_hour}AM NY Candle" if target_hour < 12 else f"{target_hour}AM NY Candle"
    state = SetupState("NY 5AM CRT", symbol, range_high, range_low, label)

    session_start = candle["close_time"]  # entries only hunted from candle CLOSE, not open

    # NY session end - entries should only be hunted for during the NY
    # trading session, not the whole rest of the day/night. Using 5PM NY
    # (17:00) as the standard NY forex session close.
    session_end = session_start.normalize() + pd.Timedelta(hours=17)
    if session_end <= session_start:
        session_end += pd.Timedelta(days=1)

    # align timezone-awareness between session_start/session_end and
    # df_intraday's index before comparing, otherwise pandas raises on
    # tz-naive vs tz-aware
    intraday_idx = df_intraday.index
    if intraday_idx.tz is None and session_start.tzinfo is not None:
        session_start_cmp = session_start.tz_localize(None)
        session_end_cmp = session_end.tz_localize(None)
    elif intraday_idx.tz is not None and session_start.tzinfo is None:
        session_start_cmp = session_start.tz_localize(intraday_idx.tz)
        session_end_cmp = session_end.tz_localize(intraday_idx.tz)
    elif intraday_idx.tz is not None and session_start.tzinfo is not None:
        session_start_cmp = session_start.tz_convert(intraday_idx.tz)
        session_end_cmp = session_end.tz_convert(intraday_idx.tz)
    else:
        session_start_cmp = session_start
        session_end_cmp = session_end

    intraday_after = df_intraday[
        (df_intraday.index >= session_start_cmp) & (df_intraday.index <= session_end_cmp)
    ]
    if len(intraday_after) < 5:
        state.status = "waiting"
        close_hour = (target_hour + 4) % 24
        state.notes = f"{target_hour}AM NY candle range marked ({target_hour}AM-{close_hour}AM NY) - waiting for the candle to finish printing / for NY session entries."
        return state

    intraday_after = intraday_after.reset_index()
    swings = find_swings(intraday_after, lookback=3)
    fvgs = find_fvgs(intraday_after)
    breakers = find_breaker_blocks(intraday_after, swings)
    order_blocks = find_order_blocks(intraday_after)

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

    # HTF bias: derived from the 4H timeframe (df_4h is already the CRT's
    # native 4H data) rather than daily, per instruction to weight 4H more
    # heavily than daily. A signal against the bias is NOT filtered out -
    # it still fires, but gets marked "Against Trend".
    bias = compute_htf_bias(df_4h)
    state.against_trend = not bias_allows_direction(bias, state.direction)

    # MSS confirmed -> entry logic. Per the rule, entries are ONLY taken
    # INSIDE the CRT range - if the MSS candle closed already outside the
    # range, wait for price to come back inside before entering.
    reentry_idx, reentry_price = find_entry_after_range_reentry(
        intraday_after, state.mss_index, state.direction, range_low, range_high
    )

    if reentry_idx == "pending":
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()}, but price broke outside the CRT range before "
            f"confirming - waiting for price to return inside the range before entering."
        )
        return state
    elif reentry_idx is not None:
        entry_idx = reentry_idx
        entry_price = reentry_price
    else:
        entry_idx = state.mss_index
        entry_price = intraday_after["Close"].values[entry_idx]

    confluence = check_fvg_breaker_confluence(entry_idx, state.direction, fvgs, breakers, order_blocks=order_blocks)
    if not confluence:
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()} and price is inside the CRT range, but no FVG or "
            f"breaker block confluence found yet - waiting for confluence before entering."
        )
        return state

    state.entry_price = entry_price
    state.stop_loss = state.sweep_price
    state.confluence = confluence

    liquidity = nearest_unswept_liquidity(
        entry_price, state.direction, swings,
        session_high=range_high, session_low=range_low, up_to_index=entry_idx
    )
    if len(liquidity) >= 1:
        state.tp1 = liquidity[0].price
    if len(liquidity) >= 2:
        state.tp2 = liquidity[1].price

    state.status = "mss_confirmed"
    trend_tag = " [AGAINST TREND]" if state.against_trend else ""
    state.notes = (
        f"MSS confirmed {state.direction.upper()} off 5AM NY range{trend_tag}. Entry ~{entry_price:.4f}, "
        f"SL {state.stop_loss:.4f}, TP1 {state.tp1}, TP2 {state.tp2}. "
        f"Confluence: {', '.join(state.confluence) if state.confluence else 'none'}."
    )
    return state
