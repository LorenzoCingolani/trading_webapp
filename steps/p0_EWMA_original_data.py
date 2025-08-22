import pandas as pd
import numpy as np
from fib_strategy_funs import (
    main_fib_levels_fun,
    sub_fib_levels_fun,
    calculate_buy_based_fib,
    calculate_sell_based_fib,
)

# ---------- Daily data ----------
daily_dates = [
    "02/07/2025","03/07/2025","04/07/2025","05/07/2025","06/07/2025",
    "07/07/2025","08/07/2025","09/07/2025","10/07/2025","11/07/2025",
    "12/07/2025","13/07/2025","14/07/2025","15/07/2025","16/07/2025",
    "17/07/2025","18/07/2025","19/07/2025","20/07/2025","21/07/2025",
    "22/07/2025","23/07/2025","24/07/2025","25/07/2025","26/07/2025",
    "27/07/2025","28/07/2025","29/07/2025","30/07/2025","31/07/2025",
    "01/08/2025","02/08/2025","03/08/2025","04/08/2025","05/08/2025",
    "06/08/2025","07/08/2025","08/08/2025","09/08/2025","10/08/2025"
]
daily_highs = [
    150,145,148,152,155,158,160,162,165,168,
    170,172,174,176,178,175,173,170,168,166,
    165,163,160,158,155,152,150,148,146,144,
    142,140,138,136,134,132,130,128,126,124
]
daily_lows = [
    145,140,143,147,150,153,155,157,160,162,
    165,167,169,171,173,170,168,165,163,161,
    160,158,155,153,150,148,146,144,142,140,
    138,136,134,132,130,128,126,124,122,120
]
daily_closes = [
    148,142,146,150,153,156,158,160,163,165,
    168,170,172,174,176,173,171,168,166,164,
    163,161,158,156,153,150,148,146,144,142,
    140,138,136,134,132,130,128,126,124,122
]

daily_df = pd.DataFrame({
    "date": pd.to_datetime(daily_dates, format="%d/%m/%Y"),
    "high": daily_highs,
    "low": daily_lows,
    "close": daily_closes,
}).sort_values("date", ignore_index=True)

# ---------- EWMA signals ----------
daily_df["ewma_5"] = daily_df["close"].ewm(span=5, adjust=False).mean()
daily_df["ewma_20"] = daily_df["close"].ewm(span=20, adjust=False).mean()
daily_df["signal"] = np.where(daily_df["ewma_5"] > daily_df["ewma_20"], "buy", "sell")
daily_df["friday_signal"] = np.where(daily_df["date"].dt.dayofweek.eq(4), daily_df["signal"], np.nan)

# ---------- Excel writer ----------
out_xlsx = "daily_analysis_combined.xlsx"
writer = pd.ExcelWriter(out_xlsx, engine="openpyxl")
sheet = "fib_buy_sell"
startrow = 0

def write_block_title(title: str):
    global startrow
    pd.DataFrame({title: [title]}).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False)
    startrow += 2

def write_df(df: pd.DataFrame, label: str, startcol: int = 0):
    global startrow
    pd.DataFrame({label: [label]}).to_excel(writer, sheet_name=sheet, startrow=startrow, startcol=startcol, index=False)
    startrow += 1
    df.to_excel(writer, sheet_name=sheet, startrow=startrow, startcol=startcol, index=False)
    startrow += (len(df) if len(df) else 1) + 1

def collect_prev_week_business_days(idx: int, df: pd.DataFrame, n_days: int = 5):
    """Collect the previous n business days (Mon..Fri) before row idx (exclusive)."""
    prev = []
    j = idx - 1
    while j >= 0 and len(prev) < n_days:
        if df.loc[j, "date"].dayofweek < 5:
            prev.append(j)
        j -= 1
    prev.sort()
    return prev

def collect_next_week_business_days(idx: int, df: pd.DataFrame, n_days: int = 5):
    """Collect the next n business days (Mon..Fri) after row idx (exclusive)."""
    nxt = []
    j = idx + 1
    while j < len(df) and len(nxt) < n_days:
        if df.loc[j, "date"].dayofweek < 5:
            nxt.append(j)
        j += 1
    return nxt

# ---------- Iterate all Fridays ----------
week_counter = 1
for fri_idx, val in daily_df["friday_signal"].items():
    if pd.isna(val):
        continue

    friday_date = daily_df.loc[fri_idx, "date"].date()

    # (A) Previous week (Mon..Fri) -> levels
    prev_idxs = collect_prev_week_business_days(fri_idx, daily_df, n_days=5)
    if len(prev_idxs) < 5:
        write_block_title(f"Week {week_counter} — Friday {friday_date} — Signal={val} — INSUFFICIENT PREV-WEEK DATA")
        # Show what we have
        if prev_idxs:
            write_df(daily_df.loc[prev_idxs, ["date","high","low"]].copy(), "prev_week_partial")
        week_counter += 1
        continue

    prev_week_df = daily_df.loc[prev_idxs, ["date","high","low"]].copy()
    weekly_high = prev_week_df["high"].max()
    weekly_low  = prev_week_df["low"].min()
    weekly_monday_date = prev_week_df.iloc[0]["date"]

    # (B) Next week (Mon..Fri) -> trading window
    next_idxs = collect_next_week_business_days(fri_idx, daily_df, n_days=5)
    if len(next_idxs) < 5:
        write_block_title(f"Week {week_counter} — Friday {friday_date} — Signal={val} — INSUFFICIENT NEXT-WEEK DATA")
        write_df(prev_week_df, "prev_week_df (used for levels)")
        if next_idxs:
            write_df(daily_df.loc[next_idxs, ["date","high","low"]].copy(), "next_week_partial")
        week_counter += 1
        continue

    next_week_df = daily_df.loc[next_idxs, ["date","high","low"]].copy()

    # Build Fib levels from PREVIOUS week
    main_bucket_df = main_fib_levels_fun(weekly_high, weekly_low, weekly_monday_date)
    main_vals = main_bucket_df.iloc[6:13, 0].values

    write_block_title(f"Week {week_counter} — Friday {friday_date} — Signal={val} — Levels from PREV week")
    write_df(pd.DataFrame({
        "prev_week_start": [prev_week_df["date"].iloc[0].date()],
        "prev_week_end":   [prev_week_df["date"].iloc[-1].date()],
        "weekly_high":     [weekly_high],
        "weekly_low":      [weekly_low],
        "levels_monday":   [weekly_monday_date.date()],
        "friday_date":     [friday_date],
        "friday_signal":   [val],
    }), "meta")

    write_df(prev_week_df,  "prev_week_df (levels source)")
    write_df(main_bucket_df, "main_bucket_df")

    if len(main_vals) < 5:
        write_df(pd.DataFrame({"note": ["Insufficient main fib levels (<5)"]}), "guard_note")
        write_df(next_week_df, "next_week_df (trade window)")
        week_counter += 1
        continue

    sub_bucket_df = sub_fib_levels_fun(main_vals)
    write_df(sub_bucket_df, "sub_bucket_df")
    write_df(next_week_df, "next_week_df (trade window)")

    # Run fib logic on NEXT week using PREV week levels
    if val.lower() == "sell":
        fib_result_df, levels_df = calculate_sell_based_fib(main_bucket_df, sub_bucket_df, next_week_df)
        action = "sell"
    else:
        fib_result_df, levels_df = calculate_buy_based_fib(main_bucket_df, sub_bucket_df, next_week_df)
        action = "buy"

    # Enrich result with context (non-destructive)
    fib_result_df = fib_result_df.copy()
    fib_result_df.insert(0, "action_run", action)
    fib_result_df.insert(1, "friday_date", friday_date)
    fib_result_df.insert(2, "friday_signal", val)
    fib_result_df.insert(3, "levels_from_week_start", prev_week_df["date"].iloc[0].date())
    fib_result_df.insert(4, "levels_from_week_end",   prev_week_df["date"].iloc[-1].date())
    fib_result_df.insert(5, "weekly_high", weekly_high)
    fib_result_df.insert(6, "weekly_low", weekly_low)
    fib_result_df.insert(7, "trade_week_start", next_week_df["date"].iloc[0].date())
    fib_result_df.insert(8, "trade_week_end",   next_week_df["date"].iloc[-1].date())

    write_df(fib_result_df, "fib_result_df")
    write_df(levels_df, "levels_df")

    week_counter += 1

# Also write the full daily_df at the end
write_df(daily_df, "daily_df (all) — appended")

writer.close()
print(f"Wrote: {out_xlsx}")
