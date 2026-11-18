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


def classify_structure_breaks(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    use_high_low: bool = True,
) -> List[dict]:
    """
    Ports the BOS vs MSS state machine from the "Multi Timeframe Break of
    Structure(BOS) & Market Structure Shift(MSS)" Pine Script indicator
    (Lenny_Kiruthu). This is a stricter, more correct distinction than
    "any close past the last opposite swing point":

    - Tracks the most recent unbroken swing high and swing low.
    - When price closes (or wicks, if use_high_low=True) beyond one of
      those levels, that's a structure break. Whether it's labeled BOS
      or MSS depends on the direction of the PREVIOUS break:
        - Same direction as the previous break -> BOS (continuation)
        - Opposite direction from the previous break -> MSS (reversal)
        - No previous break yet -> unclassified (matches the source,
          which doesn't plot/alert on the very first break either)

    Returns a list of dicts: {"index": i, "type": "BOS"/"MSS", "direction": "long"/"short"}
    """
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    price_for_high = highs if use_high_low else closes
    price_for_low = lows if use_high_low else closes

    swing_highs_by_index = {s.index: s.price for s in swings if s.kind == "high"}
    swing_lows_by_index = {s.index: s.price for s in swings if s.kind == "low"}

    prev_high = None
    prev_low = None
    high_present = False
    low_present = False
    prev_breakout_type = 0  # 0 = none yet, 1 = last break was bullish, -1 = last break was bearish

    breaks: List[dict] = []

    for i in range(len(df)):
        if i in swing_highs_by_index:
            prev_high = swing_highs_by_index[i]
            high_present = True
        if i in swing_lows_by_index:
            prev_low = swing_lows_by_index[i]
            low_present = True

        if high_present and prev_high is not None and price_for_high[i] > prev_high:
            if prev_breakout_type == 1:
                breaks.append({"index": i, "type": "BOS", "direction": "long"})
            elif prev_breakout_type == -1:
                breaks.append({"index": i, "type": "MSS", "direction": "long"})
            high_present = False
            prev_breakout_type = 1

        if low_present and prev_low is not None and price_for_low[i] < prev_low:
            if prev_breakout_type == -1:
                breaks.append({"index": i, "type": "BOS", "direction": "short"})
            elif prev_breakout_type == 1:
                breaks.append({"index": i, "type": "MSS", "direction": "short"})
            low_present = False
            prev_breakout_type = -1

    return breaks


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


def find_precise_breaker_block(
    df: pd.DataFrame,
    sweep_index: int,
    mss_index: int,
    direction: Direction,
    only_body: bool = False,
) -> Optional[BreakerBlock]:
    """
    A more precise, trade-specific breaker block, ported from LuxAlgo's
    "Breaker Blocks with Signals" indicator. Rather than searching a fixed
    handful of candles backward from the MSS break (find_breaker_blocks
    above), this searches FORWARD from the swept structure point toward
    the confirming MSS break, and takes the FIRST candle of the correct
    color in that window - this ties the breaker block specifically to
    THIS trade's sweep-to-MSS move, not just any nearby candle.
    - Bullish breaker: first GREEN candle between the sweep and the MSS break
    - Bearish breaker: first RED candle between the sweep and the MSS break

    only_body: if True, use the candle's open/close (body) for the block's
    top/bottom instead of its full high/low wick range.

    Returns None if no matching-colored candle is found in that window.
    """
    opens, closes = df["Open"].values, df["Close"].values
    highs, lows = df["High"].values, df["Low"].values

    start = sweep_index + 1
    end = mss_index
    if start >= end:
        return None

    for i in range(start, end):
        is_green = closes[i] > opens[i]
        is_red = closes[i] < opens[i]

        if direction == "long" and is_green:
            top = max(closes[i], opens[i]) if only_body else highs[i]
            bottom = min(closes[i], opens[i]) if only_body else lows[i]
            return BreakerBlock(i, top=top, bottom=bottom, direction="long")
        elif direction == "short" and is_red:
            top = max(closes[i], opens[i]) if only_body else highs[i]
            bottom = min(closes[i], opens[i]) if only_body else lows[i]
            return BreakerBlock(i, top=top, bottom=bottom, direction="short")

    return None


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

    MSS confirmation uses the proper BOS-vs-MSS state machine (ported from
    the "Multi Timeframe Break of Structure & Market Structure Shift" Pine
    Script by Lenny_Kiruthu): a structure break only counts as a genuine
    MSS if it REVERSES the direction of the prior structure break. A break
    that continues the same direction as the prior one is a BOS
    (continuation), not an MSS, and does NOT confirm a reversal trade here.
    """
    highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values
    opens = df["Open"].values if "Open" in df.columns else closes.copy()
    result = {
        "sweep_detected": False, "sweep_side": None, "sweep_index": None, "sweep_price": None,
        "mss_confirmed": False, "mss_index": None, "direction": None, "invalidated": False,
    }

    structure_breaks = classify_structure_breaks(df, swings, use_high_low=True)
    mss_by_index_direction = {
        (b["index"], b["direction"]): True for b in structure_breaks if b["type"] == "MSS"
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
                # a low sweep sets up a LONG reversal - needs a genuine MSS
                # (not just BOS continuation) breaking to the upside here
                if (i, "long") in mss_by_index_direction and has_displacement:
                    result.update(mss_confirmed=True, mss_index=i, direction="long")
            else:
                if (i, "short") in mss_by_index_direction and has_displacement:
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


def min_stop_distance_ok(entry_price: float, stop_loss: float, direction: Direction,
                         min_pips: float = 1.5) -> bool:
    """Reject signals whose stop is closer to the entry than `min_pips`.

    A stop that sits inside the broker's bid/ask spread (a fraction of a
    pip away, as happens when the re-entry candle's close lands almost
    exactly on the swept level) can never be triggered by normal price
    movement, so the trade is un-runnable. Enforcing a floor keeps every
    emitted signal actually placeable.
    """
    if entry_price is None or stop_loss is None:
        return False
    distance = abs(stop_loss - entry_price)
    # Price bands, not a single threshold: JPY/GOLD/BTC quote at 0.01-per-pip,
    # most FX pairs at 0.0001. Anything below $1 (some crypto) also uses 0.0001.
    price = abs(entry_price)
    pip = 0.01 if price >= 30 else 0.0001
    return distance >= min_pips * pip


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

    # Reject signals whose stop sits inside the broker spread - they can
    # never be triggered, so there is no trade to take.
    if not min_stop_distance_ok(entry_price, state.stop_loss, state.direction):
        state.entry_price = None
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()} but the stop is too close to the "
            f"entry (inside the broker spread) - not a placeable trade."
        )
        return state

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

    # CRT-specific improvement: also check for a precise, trade-specific
    # breaker block (LuxAlgo's "Breaker Blocks with Signals" definition -
    # the first correctly-colored candle scanning forward from the sweep
    # toward the MSS break, rather than just a nearby candle). This is
    # additive - it can add "Precise Breaker Block" as extra confluence,
    # or on its own satisfy the confluence requirement even if the
    # simpler FVG/breaker/order-block checks above found nothing.
    precise_breaker = find_precise_breaker_block(intraday_after, state.sweep_index, state.mss_index, state.direction)
    if precise_breaker is not None:
        confluence.append("Precise Breaker Block")

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

    # Reject signals whose stop sits inside the broker spread - they can
    # never be triggered, so there is no trade to take.
    if not min_stop_distance_ok(entry_price, state.stop_loss, state.direction):
        state.entry_price = None
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()} but the stop is too close to the "
            f"entry (inside the broker spread) - not a placeable trade."
        )
        return state

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


# --------------------------------------------------------------------------- #
# Major Liquidity Setup (Setup C)
# --------------------------------------------------------------------------- #
#
# This setup identifies "major liquidity grab" scenarios:
#
#   1. Price forms a recognisable consolidation range (a cluster of candles
#      whose highs/lows stay within a relatively tight band). This creates a
#      pool of resting buy orders above the range high (equal highs / stop
#      orders) and resting sell orders below the range low.
#
#   2. Price spikes THROUGH one side of that range on a wick, then CLOSES
#      back inside (or rapidly reverses) — this is the liquidity grab. It
#      represents smart money sweeping the resting orders before reversing.
#
#   3. After the grab an MSS (Market Structure Shift) in the OPPOSITE
#      direction confirms the reversal. Entry, SL and TP logic then mirrors
#      the existing Daily CRT / NY CRT rules.
#
# Best timeframes: 15m, 30m, 1h — per user specification and because on
# lower timeframes (5m) the consolidations are too noisy, and on higher
# timeframes (4h+) a "grab" looks like a normal daily candle.
#
# Consolidation detection logic
# ─────────────────────────────
# We walk back from the most recent candle and collect a window of candles
# whose highs cluster within `range_threshold_pct` percent of one another,
# and whose lows likewise cluster, for at least `min_consolidation_candles`.
# The consolidation range = max(cluster highs) / min(cluster lows).
# A "major" level is one that has been TESTED multiple times (≥ 2 touches
# on either the high or the low side), making it a visually obvious level
# that the market repeatedly respected — these are the levels where the
# most resting liquidity builds up.
#
# After the consolidation is identified, we scan for a candle that:
#   - Wicks BEYOND the range high or low by more than a minimal amount
#   - Closes BACK INSIDE the range (classic wick-sweep / liquidity grab)
#     OR closes beyond the range but reverses sharply enough in the next
#     1-3 candles that price is back inside (a brief spike-through).
#
# Once a grab is detected, the same `detect_sweep_and_mss` function used
# by the other setups confirms the MSS, and entry/SL/TP are computed with
# the same helpers.


def find_consolidation_range(
    df: pd.DataFrame,
    lookback: int = 60,
    min_candles: int = 8,
    range_threshold_pct: float = 0.5,
    min_touches: int = 2,
) -> Optional[dict]:
    """
    Scans the last `lookback` candles for a consolidation zone — a band
    of candles whose highs and lows stay within `range_threshold_pct`%
    of each other.

    Returns a dict:
        {
          "high": float,      # top of the consolidation range
          "low":  float,      # bottom of the consolidation range
          "start_index": int, # first candle of the consolidation (in df)
          "end_index": int,   # last candle before the grab candle
          "high_touches": int,
          "low_touches":  int,
        }
    or None if no suitable consolidation was found.

    Strategy:
      - Slide a window of `min_candles` → `lookback` candles from recent
        history. For each window, compute the range (max_high / min_low).
      - Accept the window if its range is tight (< range_threshold_pct %)
        AND each of the high and low has been approached (within 10% of
        the range height) at least `min_touches` times.
      - Of all qualifying windows, prefer the one CLOSEST to the present
        (most recent) so we surface the latest relevant consolidation.
    """
    if len(df) < min_candles + 5:
        return None

    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    # Only look within the last `lookback` candles
    start_scan = max(0, n - lookback)

    best = None
    best_end = -1  # prefer the most-recent qualifying window

    # Try every window of at least min_candles candles in the lookback region
    for end in range(start_scan + min_candles, n):
        for start in range(start_scan, end - min_candles + 1):
            window_h = highs[start:end]
            window_l = lows[start:end]
            range_high = window_h.max()
            range_low = window_l.min()

            if range_low == 0:
                continue

            # Check tightness
            range_pct = (range_high - range_low) / range_low * 100
            if range_pct > range_threshold_pct:
                continue

            range_height = range_high - range_low
            touch_band = range_height * 0.15  # within 15% of range height = "touching"

            high_touches = int(np.sum(window_h >= range_high - touch_band))
            low_touches = int(np.sum(window_l <= range_low + touch_band))

            if high_touches < min_touches or low_touches < min_touches:
                continue

            # Prefer the most recent qualifying window
            if end > best_end:
                best_end = end
                best = {
                    "high": float(range_high),
                    "low": float(range_low),
                    "start_index": start,
                    "end_index": end - 1,
                    "high_touches": high_touches,
                    "low_touches": low_touches,
                }

    return best


def detect_liquidity_grab(
    df: pd.DataFrame,
    range_high: float,
    range_low: float,
    consolidation_end: int,
    min_wick_pct: float = 0.1,
    max_grab_candles_lookforward: int = 10,
) -> Optional[dict]:
    """
    After a consolidation range is identified, scans the candles AFTER
    consolidation_end for a liquidity grab:

      - A wick that extends beyond the range high or low by at least
        `min_wick_pct`% of the range height
      - That closes back INSIDE the range (wick-only sweep), OR
        closes slightly beyond but price returns inside within
        `max_grab_candles_lookforward` candles.

    Returns:
        {
          "grab_index": int,
          "grab_side":  "high" | "low",
          "grab_extreme": float,   # the wick tip price
          "return_index": int,     # first close back inside the range
        }
    or None if no grab found.
    """
    if consolidation_end + 1 >= len(df):
        return None

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    range_height = range_high - range_low
    if range_height <= 0:
        return None

    min_extension = range_height * min_wick_pct

    scan_end = min(len(df), consolidation_end + 1 + max_grab_candles_lookforward * 3)

    for i in range(consolidation_end + 1, scan_end):
        # --- HIGH side grab ---
        if highs[i] > range_high + min_extension:
            # Check if close is back inside (wick sweep)
            if closes[i] <= range_high:
                return {
                    "grab_index": i,
                    "grab_side": "high",
                    "grab_extreme": float(highs[i]),
                    "return_index": i,
                }
            # Or if price returns inside in the next few candles
            for j in range(i + 1, min(i + max_grab_candles_lookforward + 1, len(df))):
                if closes[j] <= range_high:
                    return {
                        "grab_index": i,
                        "grab_side": "high",
                        "grab_extreme": float(highs[i]),
                        "return_index": j,
                    }

        # --- LOW side grab ---
        if lows[i] < range_low - min_extension:
            if closes[i] >= range_low:
                return {
                    "grab_index": i,
                    "grab_side": "low",
                    "grab_extreme": float(lows[i]),
                    "return_index": i,
                }
            for j in range(i + 1, min(i + max_grab_candles_lookforward + 1, len(df))):
                if closes[j] >= range_low:
                    return {
                        "grab_index": i,
                        "grab_side": "low",
                        "grab_extreme": float(lows[i]),
                        "return_index": j,
                    }

    return None


def analyze_major_liquidity_setup(
    df: pd.DataFrame,
    symbol: str,
    timeframe_label: str = "15m",
    consolidation_lookback: int = 80,
    min_consolidation_candles: int = 8,
    range_threshold_pct: float = 0.5,
    min_touches: int = 2,
) -> SetupState:
    """
    Full Major Liquidity setup analysis for a single timeframe DataFrame.

    1. Identify the most recent consolidation range.
    2. Detect a liquidity grab (wick sweep of the range high or low with
       price returning inside).
    3. Confirm an MSS in the opposite direction.
    4. Compute entry / SL / TP using the same logic as the other setups.

    Returns a SetupState with setup_name = "Major Liquidity ({timeframe_label})".
    """
    setup_name = f"Major Liquidity ({timeframe_label})"

    if len(df) < min_consolidation_candles + 10:
        return SetupState(setup_name, symbol, float("nan"), float("nan"), "Consolidation Range",
                          status="no_data", notes="Not enough data.")

    # Work on a reset-index copy so array indices stay consistent
    df_work = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()

    consol = find_consolidation_range(
        df_work,
        lookback=consolidation_lookback,
        min_candles=min_consolidation_candles,
        range_threshold_pct=range_threshold_pct,
        min_touches=min_touches,
    )
    if consol is None:
        return SetupState(setup_name, symbol, float("nan"), float("nan"), "Consolidation Range",
                          status="waiting",
                          notes="No major consolidation zone identified on this timeframe yet.")

    range_high = consol["high"]
    range_low = consol["low"]
    consol_end = consol["end_index"]

    state = SetupState(
        setup_name, symbol, range_high, range_low,
        f"Consolidation Range ({consol['high_touches']}H/{consol['low_touches']}L touches)",
    )

    grab = detect_liquidity_grab(
        df_work, range_high, range_low, consol_end,
        min_wick_pct=0.05,
        max_grab_candles_lookforward=8,
    )
    if grab is None:
        state.status = "waiting"
        state.notes = (
            f"Consolidation zone identified: {range_low:.4f} — {range_high:.4f} "
            f"({consol['high_touches']} high touches, {consol['low_touches']} low touches). "
            f"Watching for a liquidity grab (wick sweep of either side)."
        )
        return state

    # Treat the grab as a sweep in detect_sweep_and_mss terms:
    # search for an MSS from the grab index onward
    state.sweep_detected = True
    state.sweep_side = grab["grab_side"]
    state.sweep_index = grab["grab_index"]
    state.sweep_price = grab["grab_extreme"]

    swings = find_swings(df_work, lookback=3)
    scan = detect_sweep_and_mss(
        df_work, range_high, range_low, swings,
        search_start=max(0, grab["return_index"] - 2),
    )

    # If detect_sweep_and_mss found a fresh sweep + MSS in the region,
    # use it; otherwise fall back to our grab info + no MSS yet
    if scan["sweep_detected"] and scan["mss_confirmed"]:
        state.sweep_index = scan["sweep_index"]
        state.sweep_price = scan["sweep_price"]
        state.sweep_side = scan["sweep_side"]
        state.mss_confirmed = True
        state.mss_index = scan["mss_index"]
        state.direction = scan["direction"]
    elif not scan["mss_confirmed"]:
        # We know there was a grab but no MSS yet
        state.status = "sweep_only"
        side_label = "HIGH" if grab["grab_side"] == "high" else "LOW"
        state.notes = (
            f"Liquidity grab of range {side_label} detected at {grab['grab_extreme']:.4f} "
            f"(range: {range_low:.4f} — {range_high:.4f}). "
            f"Waiting for MSS confirmation of reversal."
        )
        return state

    # --- MSS confirmed ---
    # HTF bias from the data we have (resample to 4H equivalent)
    try:
        df_4h_bias = df.resample("4h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        bias = compute_htf_bias(df_4h_bias)
    except Exception:
        bias = "neutral"
    state.against_trend = not bias_allows_direction(bias, state.direction)

    fvgs = find_fvgs(df_work)
    breakers = find_breaker_blocks(df_work, swings)
    order_blocks = find_order_blocks(df_work)

    # Entry: inside the consolidation range (same logic as CRT)
    reentry_idx, reentry_price = find_entry_after_range_reentry(
        df_work, state.mss_index, state.direction, range_low, range_high
    )

    if reentry_idx == "pending":
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()} after liquidity grab at {state.sweep_price:.4f}. "
            f"Price broke outside the consolidation range before confirming — "
            f"waiting for price to return inside the range before entering."
        )
        return state
    elif reentry_idx is not None:
        entry_idx = reentry_idx
        entry_price = reentry_price
    else:
        entry_idx = state.mss_index
        entry_price = df_work["Close"].values[entry_idx]

    confluence = check_fvg_breaker_confluence(entry_idx, state.direction, fvgs, breakers, order_blocks=order_blocks)

    # Also check for precise breaker block (same improvement as NY CRT)
    precise_breaker = find_precise_breaker_block(
        df_work, state.sweep_index, state.mss_index, state.direction
    )
    if precise_breaker is not None:
        confluence.append("Precise Breaker Block")

    if not confluence:
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()} after liquidity grab at {state.sweep_price:.4f}. "
            f"No FVG or breaker block confluence yet — waiting for confluence before entering."
        )
        return state

    state.entry_price = float(entry_price)
    state.stop_loss = state.sweep_price
    state.confluence = confluence

    # Reject signals whose stop sits inside the broker spread - they can
    # never be triggered, so there is no trade to take.
    if not min_stop_distance_ok(entry_price, state.stop_loss, state.direction):
        state.entry_price = None
        state.status = "mss_confirmed"
        state.notes = (
            f"MSS confirmed {state.direction.upper()} after liquidity grab at "
            f"{state.sweep_price:.4f} but the stop is too close to the entry "
            f"(inside the broker spread) - not a placeable trade."
        )
        return state

    liquidity = nearest_unswept_liquidity(
        entry_price, state.direction, swings,
        session_high=range_high, session_low=range_low,
        up_to_index=entry_idx,
    )
    if len(liquidity) >= 1:
        state.tp1 = liquidity[0].price
    if len(liquidity) >= 2:
        state.tp2 = liquidity[1].price

    state.status = "mss_confirmed"
    trend_tag = " [AGAINST TREND]" if state.against_trend else ""
    state.notes = (
        f"MSS confirmed {state.direction.upper()}{trend_tag} after major liquidity grab at "
        f"{state.sweep_price:.4f} (range: {range_low:.4f} — {range_high:.4f}). "
        f"Entry ~{state.entry_price:.4f}, SL {state.stop_loss:.4f}, "
        f"TP1 {state.tp1}, TP2 {state.tp2}. "
        f"Confluence: {', '.join(state.confluence) if state.confluence else 'none'}."
    )
    return state


def run_major_liquidity_across_timeframes(
    df_15m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1h: pd.DataFrame,
    symbol: str,
) -> SetupState:
    """
    Runs the Major Liquidity setup on 15m, 30m, and 1h timeframes and
    returns the highest-priority result. Priority order:
      mss_confirmed (with entry) > sweep_only > mss_confirmed (no entry yet) > waiting > no_data
    The winning timeframe's label is included in the setup_name.
    """
    def _score(s: SetupState) -> int:
        if s.status == "mss_confirmed" and s.entry_price is not None:
            return 4
        if s.status == "sweep_only":
            return 3
        if s.status == "mss_confirmed":
            return 2
        if s.status == "waiting":
            return 1
        return 0

    candidates = []
    for df, tf_label in [
        (df_15m, "15m"),
        (df_30m, "30m"),
        (df_1h,  "1h"),
    ]:
        if df is not None and not df.empty:
            try:
                result = analyze_major_liquidity_setup(df, symbol, timeframe_label=tf_label)
                candidates.append(result)
            except Exception as e:
                # Don't let one timeframe failure kill the others
                err_state = SetupState(
                    f"Major Liquidity ({tf_label})", symbol,
                    float("nan"), float("nan"), "Consolidation Range",
                    status="no_data", notes=f"Analysis error: {e}",
                )
                candidates.append(err_state)

    if not candidates:
        return SetupState("Major Liquidity", symbol, float("nan"), float("nan"),
                          "Consolidation Range", status="no_data",
                          notes="No data available for any timeframe.")

    return max(candidates, key=_score)


# --------------------------------------------------------------------------- #
# 5-minute fully-confirmed signal for NY CRT setups
# --------------------------------------------------------------------------- #
#
# A signal is only fired when ALL of the following are met — identical
# requirements to the other two setups, just evaluated on 5m data:
#
#   1. The NY CRT range (5AM 4H candle) must be defined and valid.
#   2. A sweep of the CRT range high or low must have occurred on 5m
#      (wick through, close back inside — same definition as Setup B).
#   3. An MSS in the OPPOSITE direction must confirm the reversal.
#   4. At least one confluence factor (FVG, Breaker Block, Precise Breaker
#      Block, or Order Block) must be present near the entry point.
#   5. The entry must be INSIDE the CRT range (same re-entry rule).
#
# Nothing is sent until every one of those gates is cleared.


@dataclass
class MSSSignal:
    """
    A fully-confirmed 5m signal for the NY CRT setup.
    All fields mirror SetupState so the alert message and dashboard card
    can display the same level of detail as Setup A and Setup B signals.
    """
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    tp1: Optional[float]
    tp2: Optional[float]
    confluence: List[str]
    sweep_price: float          # the wick extreme (SL anchor)
    sweep_side: str             # "high" or "low"
    crt_range_high: float
    crt_range_low: float
    signal_time: Optional[pd.Timestamp] = None
    against_trend: bool = False
    notes: str = ""


def analyze_ny_crt_5m(
    df_5m: pd.DataFrame,
    crt_range_high: float,
    crt_range_low: float,
    symbol: str,
    htf_bias: str = "neutral",
) -> Optional[MSSSignal]:
    """
    Runs the full NY CRT confirmation pipeline on 5-minute data.

    Requires:
      - df_5m filtered to the current NY session (9AM–5PM NY) only,
        with index reset to RangeIndex (caller handles this).
      - crt_range_high / crt_range_low from the 4H CRT candle.
      - htf_bias from the 4H data ("bullish" / "bearish" / "neutral"),
        used to tag against-trend signals (they still fire, just flagged).

    Returns a fully-populated MSSSignal if ALL confirmation requirements
    are met, or None if any gate fails.
    """
    if df_5m is None or df_5m.empty or len(df_5m) < 8:
        return None
    if crt_range_high is None or crt_range_low is None:
        return None
    if crt_range_high != crt_range_high or crt_range_low != crt_range_low:  # NaN guard
        return None

    swings = find_swings(df_5m, lookback=3)
    fvgs = find_fvgs(df_5m)
    breakers = find_breaker_blocks(df_5m, swings)
    order_blocks = find_order_blocks(df_5m)

    # Gate 1 + 2 + 3: sweep of CRT range + MSS — reuse the same state
    # machine used by Setup B, so the sweep/MSS definition is identical.
    scan = detect_sweep_and_mss(df_5m, crt_range_high, crt_range_low, swings)

    if not scan["sweep_detected"] or not scan["mss_confirmed"]:
        return None

    direction: Direction = scan["direction"]
    sweep_side: str = scan["sweep_side"]
    sweep_price: float = scan["sweep_price"]
    mss_index: int = scan["mss_index"]

    # Gate 4: at least one confluence factor
    # Gate 5: entry must be inside the CRT range (re-entry rule)
    reentry_idx, reentry_price = find_entry_after_range_reentry(
        df_5m, mss_index, direction, crt_range_low, crt_range_high
    )

    if reentry_idx == "pending":
        # MSS confirmed but price hasn't come back inside the range yet
        return None
    elif reentry_idx is not None:
        entry_idx = reentry_idx
        entry_price = float(reentry_price)
    else:
        entry_idx = mss_index
        entry_price = float(df_5m["Close"].values[entry_idx])

    confluence = check_fvg_breaker_confluence(
        entry_idx, direction, fvgs, breakers, order_blocks=order_blocks
    )

    # Also check for precise breaker block (same improvement as Setup B)
    precise_breaker = find_precise_breaker_block(
        df_5m, scan["sweep_index"], mss_index, direction
    )
    if precise_breaker is not None:
        confluence.append("Precise Breaker Block")

    if not confluence:
        # All structural requirements met but no confluence — not a signal yet
        return None

    # Reject signals whose stop sits inside the broker spread - they can
    # never be triggered, so there is no trade to take.
    if not min_stop_distance_ok(entry_price, sweep_price, direction):
        return None

    # Trend tag
    against_trend = not bias_allows_direction(htf_bias, direction)

    # TP levels
    liquidity = nearest_unswept_liquidity(
        entry_price, direction, swings,
        session_high=crt_range_high,
        session_low=crt_range_low,
        up_to_index=entry_idx,
    )
    tp1 = liquidity[0].price if len(liquidity) >= 1 else None
    tp2 = liquidity[1].price if len(liquidity) >= 2 else None

    trend_tag = " [AGAINST TREND]" if against_trend else ""
    conf_str = ", ".join(confluence)
    notes = (
        f"5m fully confirmed {direction.upper()}{trend_tag} off NY CRT range "
        f"({crt_range_low:.4f}–{crt_range_high:.4f}). "
        f"Entry ~{entry_price:.4f}, SL {sweep_price:.4f}. "
        f"Confluence: {conf_str}."
    )

    # signal_time: timestamp of the entry candle if available
    times = df_5m.index if isinstance(df_5m.index, pd.DatetimeIndex) else None
    signal_time = times[entry_idx] if times is not None and entry_idx < len(times) else None

    return MSSSignal(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        stop_loss=sweep_price,
        tp1=tp1,
        tp2=tp2,
        confluence=confluence,
        sweep_price=sweep_price,
        sweep_side=sweep_side,
        crt_range_high=crt_range_high,
        crt_range_low=crt_range_low,
        signal_time=signal_time,
        against_trend=against_trend,
        notes=notes,
    )


def detect_5m_mss_bos(
    df_5m: pd.DataFrame,
    symbol: str,
    crt_range_high: Optional[float] = None,
    crt_range_low: Optional[float] = None,
    htf_bias: str = "neutral",
) -> Optional[MSSSignal]:
    """
    Entry point called by daily_dashboard.py.

    Filters df_5m to the current NY session, then delegates to
    analyze_ny_crt_5m for the full confirmation pipeline.
    Returns a fully-confirmed MSSSignal or None — nothing is returned
    for a partial setup (sweep only, MSS without confluence, etc.).
    """
    if df_5m is None or df_5m.empty or len(df_5m) < 10:
        return None

    # Must currently be inside the NY session
    now_ny = pd.Timestamp.now(tz=NY_TZ)
    current_hour = now_ny.hour + now_ny.minute / 60
    if not (9 <= current_hour < 17):
        return None

    # CRT range must be provided and valid — without it there is no
    # defined range to sweep, so the confirmation chain cannot start
    if crt_range_high is None or crt_range_low is None:
        return None
    if crt_range_high != crt_range_high or crt_range_low != crt_range_low:
        return None

    # Filter to today's NY session only
    idx = df_5m.index
    if idx.tz is None:
        idx_ny = idx.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        idx_ny = idx.tz_convert(NY_TZ)

    session_start_ny = now_ny.normalize().replace(hour=9)
    session_mask = (idx_ny >= session_start_ny) & (idx_ny <= now_ny)
    df_session = df_5m[session_mask]

    if len(df_session) < 8:
        return None

    df_work = df_session.reset_index()

    return analyze_ny_crt_5m(
        df_work,
        crt_range_high=crt_range_high,
        crt_range_low=crt_range_low,
        symbol=symbol,
        htf_bias=htf_bias,
    )
