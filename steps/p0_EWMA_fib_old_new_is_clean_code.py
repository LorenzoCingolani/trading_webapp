import pandas as pd
from fib_strategy_funs import (
    main_fib_levels_fun,
    sub_fib_levels_fun,
    calculate_buy_based_fib,
    calculate_sell_based_fib,
)


# ---------- Daily data (manually written) ----------
daily_dates = [
    "02/07/2025", "03/07/2025", "04/07/2025", "05/07/2025", "06/07/2025",
    "07/07/2025", "08/07/2025", "09/07/2025", "10/07/2025", "11/07/2025",
    "12/07/2025", "13/07/2025", "14/07/2025", "15/07/2025", "16/07/2025",
    "17/07/2025", "18/07/2025", "19/07/2025", "20/07/2025", "21/07/2025",
    "22/07/2025", "23/07/2025", "24/07/2025", "25/07/2025", "26/07/2025",
    "27/07/2025", "28/07/2025", "29/07/2025", "30/07/2025", "31/07/2025",
    "01/08/2025", "02/08/2025", "03/08/2025", "04/08/2025", "05/08/2025",
    "06/08/2025", "07/08/2025", "08/08/2025", "09/08/2025", "10/08/2025"
]

daily_highs = [
    150, 145, 148, 152, 155, 158, 160, 162, 165, 168,
    170, 172, 174, 176, 178, 175, 173, 170, 168, 166,
    165, 163, 160, 158, 155, 152, 150, 148, 146, 144,
    142, 140, 138, 136, 134, 132, 130, 128, 126, 124
]

daily_lows = [
    145, 140, 143, 147, 150, 153, 155, 157, 160, 162,
    165, 167, 169, 171, 173, 170, 168, 165, 163, 161,
    160, 158, 155, 153, 150, 148, 146, 144, 142, 140,
    138, 136, 134, 132, 130, 128, 126, 124, 122, 120
]

daily_closes = [
    148, 142, 146, 150, 153, 156, 158, 160, 163, 165,
    168, 170, 172, 174, 176, 173, 171, 168, 166, 164,
    163, 161, 158, 156, 153, 150, 148, 146, 144, 142,
    140, 138, 136, 134, 132, 130, 128, 126, 124, 122
]

# Create daily dataframe
daily_df = pd.DataFrame({
    "date": pd.to_datetime(daily_dates, format='%d/%m/%Y'),
    "high": daily_highs,
    "low": daily_lows,
    "close": daily_closes
})

# ---------- EWMA(5) vs EWMA(20) on CLOSE ----------
daily_df["ewma_5"] = daily_df["close"].ewm(span=5, adjust=False).mean()
daily_df["ewma_20"] = daily_df["close"].ewm(span=20, adjust=False).mean()
signal_function = lambda x: "buy" if daily_df["ewma_5"].iloc[x] > daily_df["ewma_20"].iloc[x] else "sell"
# apply on daily_df 
daily_df["signal"] = daily_df.apply(lambda row: signal_function(row.name), axis=1)
# new column that will apply all fridays signal to next week
daily_df["friday_signal"] = daily_df["signal"].where(daily_df["date"].dt.dayofweek == 4)
# now weekly dataframe 
import numpy as np

for idx, val in daily_df["friday_signal"].items():
    if pd.notna(val) and val.lower() == "sell":
        start = idx + 1
        end = start + 7
        print(f"Friday: {daily_df.loc[idx, 'date'].date()} | Signal: {val}")
        daily_high_low_df = daily_df.iloc[start:end][["date", "high", "low"]][:-2]
        weekly_low = daily_high_low_df["low"].min()
        weekly_high = daily_high_low_df["high"].max()
        weekly_monday_date = daily_high_low_df.iloc[0]["date"]
        main_bucket_df = main_fib_levels_fun(weekly_high, weekly_low, weekly_monday_date)
        main_fib_levels_values = main_bucket_df.iloc[6:13, 0].values
        sub_bucket_df = sub_fib_levels_fun(main_fib_levels_values)
        sell_fib_result_df, sell_levels_df = calculate_sell_based_fib(main_bucket_df, sub_bucket_df, daily_high_low_df)
        print(sell_fib_result_df)
        print(sell_levels_df)

        

        print("-" * 40)

    elif pd.notna(val) and val.lower() == "buy":
        start = idx + 1
        end = start + 7
        print(f"Friday: {daily_df.loc[idx, 'date'].date()} | Signal: {val}")
        daily_high_low_df = daily_df.iloc[start:end][["date", "high", "low"]][:-2]
        print(daily_high_low_df)
        print("-" * 40)
        weekly_low = daily_high_low_df["low"].min()
        weekly_high = daily_high_low_df["high"].max()
        weekly_monday_date = daily_high_low_df.iloc[0]["date"]
        main_bucket_df = main_fib_levels_fun(weekly_high, weekly_low, weekly_monday_date)
        main_fib_levels_values = main_bucket_df.iloc[6:13, 0].values
        sub_bucket_df = sub_fib_levels_fun(main_fib_levels_values)
        buy_fib_result_df, buy_levels_df = calculate_buy_based_fib(main_bucket_df, sub_bucket_df, daily_high_low_df)
        print(buy_fib_result_df)
        print(buy_levels_df)
