import numpy as np
import pandas as pd
from datetime import datetime

def generate_final_log_df_fixed(week_high, week_low, week_date,
                                 daily_lows, daily_highs, daily_dates):
    fibonacci_standard = np.array([0.236, 0.382, 0.500, 0.618, 0.786])
    range_val = week_high - week_low
    levels = week_high - (fibonacci_standard * range_val)
    levels = np.append([week_high], np.append(levels, [week_low]))

    level_names = [f"{round(lvl, 2)}" for lvl in levels]
    level_units = [0] * len(levels)

    # Example: You buy 1 at 23.6%, 2 at 38.2%, 2 at 61.8%, 1 at 76.4%
    buy_indices = [1, 2, 4, 5]
    unit_buys = [1, 2, 2, 1]
    for i, idx in enumerate(buy_indices):
        level_units[idx] = unit_buys[i]

    unit_tracker = level_units.copy()
    log_df = pd.DataFrame(index=[f"buy {u}" if u > 0 else "" for u in level_units],
                          columns=daily_dates)

    # To track if buy was already triggered
    buy_triggered_flags = [False] * len(levels)

    for day_idx, (day, high, low) in enumerate(zip(daily_dates, daily_highs, daily_lows)):
        for i, level in enumerate(levels):
            prev_msgs = log_df.iloc[i, :day_idx].tolist()
            prev_msgs_cleaned = [m for m in prev_msgs if isinstance(m, str)]

            msg = "n/a"

            # Skip levels with no buy
            if level_units[i] == 0:
                log_df.iat[i, day_idx] = msg
                continue

            # If already TRADECLOSED, keep it closed
            if "TRADECLOSED" in prev_msgs_cleaned:
                msg = "TRADECLOSED"

            # If buy not triggered yet
            elif not buy_triggered_flags[i] and low < level:
                msg = "buyTriggered"
                buy_triggered_flags[i] = True

            # If buy was triggered and still units open
            elif buy_triggered_flags[i] and unit_tracker[i] > 0:
                stop_level = level - 2  # example stop loss 2 points below level
                if low <= stop_level:
                    msg = f"stopLossTriggered {unit_tracker[i]} unit"
                    unit_tracker[i] = 0
                else:
                    msg = "n/a"

            # If all units closed now
            if unit_tracker[i] == 0 and "TRADECLOSED" not in prev_msgs_cleaned and \
               any("stopLossTriggered" in m for m in prev_msgs_cleaned):
                msg = "TRADECLOSED"

            log_df.iat[i, day_idx] = msg

    log_df.insert(0, "Level", level_names)
    log_df.insert(1, "Units", level_units)

    return log_df

# Inputs
week_high = 150
week_low = 60
week_date = "01/07/2024"
daily_lows = [148, 125, 110, 97, 92, 85, 70]
daily_highs = [151, 130, 115, 105, 100, 95, 75]
daily_dates = ["02/07/2024", "03/07/2024", "04/07/2024", "05/07/2024", "06/07/2024", "07/07/2024", "08/07/2024"]

# Generate final table
final_log_df_fixed = generate_final_log_df_fixed(
    week_high, week_low, week_date,
    daily_lows, daily_highs, daily_dates
)

print(final_log_df_fixed)
