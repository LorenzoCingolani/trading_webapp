import pandas as pd
import numpy as np
from fib_strategy_funs import (
    main_fib_levels_fun,
    sub_fib_levels_fun,
    calculate_buy_based_fib,
    calculate_sell_based_fib,
)

df = pd.read_csv("DATA/input_instruments/GC1_for_Fib.csv")
# Date,PX_HIGH,PX_LOW,PX_CLOSE_1D
daily_df = pd.DataFrame({
    "date": pd.to_datetime(df["Date"], format="%d/%m/%Y"),
    "high": df["PX_HIGH"],
    "low": df["PX_LOW"],
    "close": df["PX_CLOSE_1D"],
}).sort_values("date", ignore_index=True)

# ---------- EWMA signals (vectorized) ----------
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
    pd.DataFrame({title: []}).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False)
    pd.DataFrame({title: [title]}).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False)
    startrow += 2

def write_df(df: pd.DataFrame, label: str, startcol: int = 0):
    """Write a labeled dataframe and bump startrow."""
    global startrow
    pd.DataFrame({label: [label]}).to_excel(writer, sheet_name=sheet, startrow=startrow, startcol=startcol, index=False)
    startrow += 1
    df.to_excel(writer, sheet_name=sheet, startrow=startrow, startcol=startcol, index=False)
    startrow += (len(df) if len(df) else 1) + 1  # leave 1 empty row

# ---------- Iterate all Fridays ----------
week_counter = 1
for fri_idx, val in daily_df["friday_signal"].items():
    if pd.isna(val):
        continue

    friday_date = daily_df.loc[fri_idx, "date"].date()

    # ====== [ADDED] Build PREVIOUS week window (Mon..Thu before the signal Friday) for LEVELS ======
    prev_start = fri_idx - 5         # 5 trading days before Friday (Mon..Thu)
    prev_end   = fri_idx             # exclude the Friday itself (signal day)
    if prev_start < 0:
        write_block_title(f"Week {week_counter} — Friday {friday_date} — Signal={val} — INSUFFICIENT PREV-WEEK DATA")
        week_counter += 1
        continue

    prev_week_df = daily_df.iloc[prev_start:prev_end][["date","high","low"]].copy()
    weekly_low   = prev_week_df["low"].min()    # lowest low of previous week
    weekly_high  = prev_week_df["high"].max()   # highest high of previous week
    weekly_monday_date = prev_week_df.iloc[0]["date"]  # for stamping the levels week

    # ====== [CHANGED] Build NEXT week window (Mon..Fri AFTER the signal Friday) for EXECUTION ======
    start = fri_idx + 1
    end   = start + 5
    if end > len(daily_df):
        write_block_title(f"Week {week_counter} — Friday {friday_date} — Signal={val} — INSUFFICIENT NEXT-WEEK DATA")
        if start < len(daily_df):
            rem = daily_df.iloc[start:][["date", "high", "low"]].copy()
            write_df(rem, "remaining_days")
        week_counter += 1
        continue

    next_week_df = daily_df.iloc[start:end][["date","high","low"]].copy()

    # ====== [CHANGED] Make levels from PREVIOUS week, apply to NEXT week ======
    main_bucket_df = main_fib_levels_fun(weekly_high, weekly_low, weekly_monday_date)  # levels from prev week
    main_vals = main_bucket_df.iloc[6:13, 0].values

    write_block_title(f"Week {week_counter} — Friday {friday_date} — Signal={val}")
    write_df(pd.DataFrame({
        "levels_from_week_start": [prev_week_df["date"].iloc[0].date()],   # [ADDED]
        "levels_from_week_end":   [prev_week_df["date"].iloc[-1].date()],  # [ADDED]
        "execution_week_start":   [next_week_df["date"].iloc[0].date()],   # [ADDED]
        "execution_week_end":     [next_week_df["date"].iloc[-1].date()],  # [ADDED]
        "weekly_high(prev_week)": [weekly_high],                            # [ADDED]
        "weekly_low(prev_week)":  [weekly_low],                             # [ADDED]
        "friday_date":            [friday_date],
        "friday_signal":          [val],
    }), "meta")

    write_df(prev_week_df, "prev_week_df (levels_source)")      # [ADDED]
    write_df(main_bucket_df, "main_bucket_df")
    if len(main_vals) < 5:
        write_df(pd.DataFrame({"note": ["Insufficient main fib levels (<5)"]}), "guard_note")
        write_df(next_week_df, "next_week_df (execution)")      # [ADDED label]
        week_counter += 1
        continue

    sub_bucket_df = sub_fib_levels_fun(main_vals)
    write_df(sub_bucket_df, "sub_bucket_df")

    # Raw next-week OHLC (for reference)
    write_df(next_week_df, "next_week_df (execution)")          # [CHANGED label]

    # Run fib logic + write full result & levels
    if val.lower() == "sell":
        fib_result_df, levels_df = calculate_sell_based_fib(main_bucket_df, sub_bucket_df, next_week_df)
        action = "sell"
    else:
        fib_result_df, levels_df = calculate_buy_based_fib(main_bucket_df, sub_bucket_df, next_week_df)
        action = "buy"

    # Enrich result with context columns (non-destructive if columns overlap)
    fib_result_df = fib_result_df.copy()
    fib_result_df.insert(0, "action_run", action)
    fib_result_df.insert(1, "friday_date", friday_date)
    fib_result_df.insert(2, "friday_signal", val)
    fib_result_df.insert(3, "week_start", next_week_df["date"].iloc[0].date())
    fib_result_df.insert(4, "week_end", next_week_df["date"].iloc[-1].date())
    fib_result_df.insert(5, "weekly_high_prev", weekly_high)    # [ADDED]
    fib_result_df.insert(6, "weekly_low_prev", weekly_low)      # [ADDED]

    write_df(fib_result_df, "fib_result_df")
    write_df(levels_df, "levels_df")

    week_counter += 1

# Also write the daily_df itself (top of sheet if empty, else at the end)
if startrow == 0:
    write_df(daily_df, "daily_df (all)")
else:
    write_df(daily_df, "daily_df (all) — appended")

writer.close()
print(f"Wrote: {out_xlsx}")
