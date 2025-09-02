import pandas as pd
import numpy as np
from fib_strategy_funs import (
    main_fib_levels_fun,
    sub_fib_levels_fun,
    calculate_buy_based_fib,
    calculate_sell_based_fib,
    write_block_title,
    write_df,
    collect_prev_week_business_days,
    collect_next_week_business_days
)



# ---------- Daily Data ----------
daily_df = pd.DataFrame({
    "date": pd.to_datetime([
        "04/01/1993", "05/01/1993", "06/01/1993", "07/01/1993", "08/01/1993",
        "11/01/1993", "12/01/1993", "13/01/1993", "14/01/1993", "15/01/1993",
        "18/01/1993", "19/01/1993", "20/01/1993", "21/01/1993", "22/01/1993",
        "25/01/1993", "26/01/1993", "27/01/1993", "28/01/1993", "29/01/1993",
        "01/02/1993", "02/02/1993", "03/02/1993", "04/02/1993", "05/02/1993",
        "08/02/1993", "09/02/1993", "10/02/1993", "11/02/1993", "12/02/1993",
        "15/02/1993", "16/02/1993", "17/02/1993", "18/02/1993", "19/02/1993",
        "22/02/1993", "23/02/1993", "24/02/1993", "25/02/1993", "26/02/1993",
        "01/03/1993", "02/03/1993", "03/03/1993", "04/03/1993", "05/03/1993",
        "08/03/1993", "09/03/1993", "10/03/1993", "11/03/1993"
    ], format="%d/%m/%Y"),
    "high": [329.4, 329.5, 330.4, 329.7, 329.8, 328.9, 330.7, 328.5, 328.5, 328.0, 328.2, 329.4, 330.4, 330.3, 329.7, 329.1, 331.5, 331.6, 331.2, 333.0, 331.2, 331.1, 330.2, 329.3, 329.3, 329.4, 329.8, 333.8, 334.6, 331.8, 331.8, 334.4, 332.9, 332.8, 332.5, 332.0, 331.9, 332.2, 331.4, 329.8, 329.8, 330.7, 330.9, 329.7, 331.3, 329.5, 328.2, 327.7, 328.0],   # existing highs
    "low":[326.5, 328.5, 329.2, 328.4, 328.8, 326.9, 328.1, 326.3, 326.9, 326.9, 326.9, 328.4, 328.2, 328.9, 328.5, 328.2, 329.3, 330.3, 330.4, 330.6, 330.1, 330.0, 329.1, 328.5, 328.6, 328.3, 329.0, 330.3, 332.2, 328.8, 328.8, 330.5, 330.8, 329.8, 330.8, 328.7, 330.2, 330.1, 330.3, 327.8, 329.1, 329.5, 329.5, 328.6, 330.2, 327.0, 326.6, 325.8, 326.8]  ,   # existing lows
    "close":[328.4, 329.0, 330.1, 329.0, 329.5, 327.8, 328.8, 327.6, 327.3, 327.0, 328.0, 328.6, 329.8, 329.3, 328.6, 328.8, 331.1, 330.4, 331.1, 330.7, 330.7, 330.6, 329.5, 329.2, 328.9, 329.1, 329.6, 333.7, 332.7, 332.7, 330.0, 333.4, 331.5, 331.5, 331.0, 328.8, 331.6, 330.4, 330.6, 329.0, 329.3, 330.5, 329.7, 329.6, 330.4, 327.7, 327.2, 326.9, 327.6]    # existing closes
}).sort_values("date", ignore_index=True)

# ---------- EWMA Signals ----------
daily_df["ewma_5"] = daily_df["close"].ewm(span=5, adjust=False).mean()
daily_df["ewma_20"] = daily_df["close"].ewm(span=20, adjust=False).mean()
daily_df["signal"] = np.where(daily_df["ewma_5"] > daily_df["ewma_20"], "buy", "sell")
daily_df["friday_signal"] = np.where(daily_df["date"].dt.dayofweek.eq(4), daily_df["signal"], np.nan)

# ---------- Excel Writer ----------
out_xlsx = "daily_analysis_combined.xlsx"
writer = pd.ExcelWriter(out_xlsx, engine="openpyxl")
sheet = "fib_buy_sell"
startrow = 0

# ---------- Iterate All Fridays ----------
week_counter = 1

fib_result_df_full = pd.DataFrame()

for fri_idx, val in daily_df["friday_signal"].items():
    if pd.isna(val):
        continue

    friday_date = daily_df.loc[fri_idx, "date"].date()

    # (A) Previous Week
    prev_idxs = collect_prev_week_business_days(fri_idx, daily_df, n_days=5)
    if len(prev_idxs) < 5:
        startrow = write_block_title(writer, sheet, f"Week {week_counter} — Friday {friday_date} — Signal={val} — INSUFFICIENT PREV-WEEK DATA", startrow)
        if prev_idxs:
            startrow = write_df(writer, sheet, daily_df.loc[prev_idxs, ["date","high","low"]].copy(), "prev_week_partial", startrow)
        week_counter += 1
        continue

    prev_week_df = daily_df.loc[prev_idxs, ["date","high","low"]].copy()
    weekly_high = prev_week_df["high"].max()
    weekly_low  = prev_week_df["low"].min()
    weekly_monday_date = prev_week_df.iloc[0]["date"]

    # (B) Next Week
    next_idxs = collect_next_week_business_days(fri_idx, daily_df, n_days=5)
    if len(next_idxs) < 5:
        startrow = write_block_title(writer, sheet, f"Week {week_counter} — Friday {friday_date} — Signal={val} — INSUFFICIENT NEXT-WEEK DATA", startrow)
        startrow = write_df(writer, sheet, prev_week_df, "prev_week_df (used for levels)", startrow)
        if next_idxs:
            startrow = write_df(writer, sheet, daily_df.loc[next_idxs, ["date","high","low"]].copy(), "next_week_partial", startrow)
        week_counter += 1
        continue

    next_week_df = daily_df.loc[next_idxs, ["date","high","low"]].copy()

    # Build Levels
    main_bucket_df = main_fib_levels_fun(weekly_high, weekly_low, weekly_monday_date)
    main_vals = main_bucket_df.iloc[6:13, 0].values

    startrow = write_block_title(writer, sheet, f"Week {week_counter} — Friday {friday_date} — Signal={val} — Levels from PREV week", startrow)
    startrow = write_df(writer, sheet, pd.DataFrame({
        "prev_week_start": [prev_week_df["date"].iloc[0].date()],
        "prev_week_end":   [prev_week_df["date"].iloc[-1].date()],
        "weekly_high":     [weekly_high],
        "weekly_low":      [weekly_low],
        "levels_monday":   [weekly_monday_date.date()],
        "friday_date":     [friday_date],
        "friday_signal":   [val],
    }), "meta", startrow)

    startrow = write_df(writer, sheet, prev_week_df, "prev_week_df (levels source)", startrow)
    startrow = write_df(writer, sheet, main_bucket_df, "main_bucket_df", startrow)

    if len(main_vals) < 5:
        startrow = write_df(writer, sheet, pd.DataFrame({"note": ["Insufficient main fib levels (<5)"]}), "guard_note", startrow)
        startrow = write_df(writer, sheet, next_week_df, "next_week_df (trade window)", startrow)
        week_counter += 1
        continue

    sub_bucket_df = sub_fib_levels_fun(main_vals)
    startrow = write_df(writer, sheet, sub_bucket_df, "sub_bucket_df", startrow)
    startrow = write_df(writer, sheet, next_week_df, "next_week_df (trade window)", startrow)

    # Trading Logic
    if val.lower() == "sell":
        fib_result_df, levels_df = calculate_sell_based_fib(main_bucket_df, sub_bucket_df, next_week_df)
        action = "sell"
    else:
        fib_result_df, levels_df = calculate_buy_based_fib(main_bucket_df, sub_bucket_df, next_week_df)
        action = "buy"


    fib_result_df = fib_result_df.copy()
    fib_result_df.insert(0, "action_run", action)
    fib_result_df.insert(1, "friday_date", friday_date)
    fib_result_df.insert(2, "friday_signal", val)
    fib_result_df.insert(3, "levels_from_week_start", prev_week_df["date"].iloc[0].date())
    fib_result_df.insert(4, "levels_from_week_end", prev_week_df["date"].iloc[-1].date())
    fib_result_df.insert(5, "weekly_high", weekly_high)
    fib_result_df.insert(6, "weekly_low", weekly_low)
    fib_result_df.insert(7, "trade_week_start", next_week_df["date"].iloc[0].date())
    fib_result_df.insert(8, "trade_week_end", next_week_df["date"].iloc[-1].date())
    
    fib_result_df_full = pd.concat([fib_result_df_full, fib_result_df], ignore_index=True) # to save second sheet


    startrow = write_df(writer, sheet, fib_result_df, "fib_result_df", startrow)
    startrow = write_df(writer, sheet, levels_df, "levels_df", startrow)

    week_counter += 1

# Write full daily_df at end
startrow = write_df(writer, sheet, daily_df, "daily_df (all) — appended", startrow)
# add a column with sharp ratio
all_pnls = pd.to_numeric(fib_result_df_full["pnl_point"],errors="coerce").dropna()
fib_result_df_full["pnl_sharp_ratio_value"] = all_pnls.mean() / all_pnls.std()
fib_result_df_full.to_excel(writer, sheet_name="fib_result_df_full", index=False)

writer.close()


print(f"Wrote: {out_xlsx}")
