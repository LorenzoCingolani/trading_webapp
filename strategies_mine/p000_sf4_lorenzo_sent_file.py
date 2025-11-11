import numpy as np
import pandas as pd
from datetime import datetime

def fibonacci_decision(week_high, week_low, week_date, daily_lows, daily_highs, daily_dates):
    # Fibonacci levels
    fibonacci_standard = np.array([0.236, 0.382, 0.500, 0.618, 0.786])
    range_val = week_high - week_low
    levels = week_high - (fibonacci_standard * range_val)
    levels = np.insert(levels, 0, week_high)
    levels = np.append(levels, week_low)

    entry_levels = [round(l, 2) for l in levels]
    buy_config = {1: 1, 2: 2, 4: 2, 5: 1}  # Example: buy 1 at 23.6%, 2 at 38.2%, etc.
    total_units = [buy_config.get(i, 0) for i in range(len(entry_levels))]

    stop_loss_levels = [round(entry_levels[i] * 0.98, 2) if total_units[i] > 0 else None for i in range(len(entry_levels))]
    take_profit_levels = [entry_levels[i] if total_units[i] > 0 else None for i in range(len(entry_levels))]

    log_df = pd.DataFrame(index=[f"buy {total_units[i]}" if total_units[i] > 0 else "" for i in range(len(entry_levels))],
                          columns=daily_dates)
    unit_states = [[None for _ in range(total_units[i])] for i in range(len(entry_levels))]
    buy_triggered = [False] * len(entry_levels)
    unit_closed_day = [[None for _ in range(total_units[i])] for i in range(len(entry_levels))]

    for day_idx, (day, low, high) in enumerate(zip(daily_dates, daily_lows, daily_highs)):
        for i in range(len(entry_levels)):
            msg = "n/a"

            if total_units[i] == 0:
                log_df.iat[i, day_idx] = msg
                continue

            # Buy trigger
            if not buy_triggered[i] and low <= entry_levels[i]:
                buy_triggered[i] = True
                for u in range(total_units[i]):
                    unit_states[i][u] = 'open'
                msg = "buyTriggered"

            # Check SL/TP if trade is open
            elif buy_triggered[i]:
                for u in range(total_units[i]):
                    if unit_states[i][u] == 'open':
                        if low <= stop_loss_levels[i]:
                            unit_states[i][u] = 'SL'
                            unit_closed_day[i][u] = day
                            msg = f"stopLossTriggered {1 if total_units[i]==1 else f'{u+1} unit'}"
                            break
                        elif high >= take_profit_levels[i]:
                            unit_states[i][u] = 'TP'
                            unit_closed_day[i][u] = day
                            msg = f"takeProfitTriggered {1 if total_units[i]==1 else f'{u+1} unit'}"
                            break

                # If all units closed and today is not last unit close day → TRADECLOSED
                if all(s in ['TP', 'SL'] for s in unit_states[i]) and all(unit_closed_day[i]):
                    last_closed = max(unit_closed_day[i])
                    if last_closed != day:
                        msg = "TRADECLOSED"

            log_df.iat[i, day_idx] = msg

    # Add level and units
    log_df.insert(0, "level", entry_levels)
    log_df.insert(1, "unit", total_units)
    return log_df


# Inputs
week_high = 150
week_low = 60
week_date = "01/07/2024"
daily_highs = [151, 130, 115, 105, 100, 95, 75]
daily_lows = [148, 125, 110, 97, 92, 85, 70]

# daily_lows  = [145, 125, 110, 90, 88, 70, 60]
# daily_highs = [151, 130, 116, 150, 101, 95, 80]


daily_dates = ["02/07/2024", "03/07/2024", "04/07/2024", "05/07/2024", "06/07/2024", "07/07/2024", "08/07/2024"]

# Generate final table
final_log_df_fixed = fibonacci_decision(
    week_high, week_low, week_date,
    daily_lows, daily_highs, daily_dates
)
final_log_df_fixed.to_csv('final_log_df_fixed.csv')


