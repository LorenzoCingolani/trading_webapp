import numpy as np
import pandas as pd
from datetime import datetime

def analyze_single_week_day(week_high, week_low, week_date, week_close,
                             daily_lows, daily_highs, daily_dates):
    fibonacci_high = np.array([2.000, 1.764, 1.618, 1.500, 1.382, 1.236])
    fibonacci_standard = np.array([0.236, 0.382, 0.500, 0.618, 0.786])
    fibonacci_low = np.array([1.236, 1.382, 1.500, 1.618, 1.764, 2.000])

    range_val = week_high - week_low

    price_levels = {
        "high": week_low + (fibonacci_high * range_val),
        "standard": week_high - (fibonacci_standard * range_val),
        "low": week_high - (fibonacci_low * range_val)
    }

    price_levels['standard'] = np.append(price_levels['standard'], week_low)
    price_levels['standard'] = np.insert(price_levels['standard'], 0, week_high)
    main_bucket_df = pd.DataFrame(price_levels['standard'], columns=[f'FSL, h={week_high},l={week_low}'])
    sub_bucket_df = pd.DataFrame()

    for h, l in zip(price_levels['standard'][:-1], price_levels['standard'][1:]):
        sub_range = h - l
        sub_levels = {
            f"h={h},l={l}": (h - (fibonacci_standard * sub_range))
        }
        sub_levels[f'h={h},l={l}'] = np.append(sub_levels[f'h={h},l={l}'], l)
        sub_levels[f'h={h},l={l}'] = np.insert(sub_levels[f'h={h},l={l}'], 0, h)
        sub_bucket_df = pd.concat([sub_bucket_df, pd.DataFrame(sub_levels)], axis=1)

    u1 = float(main_bucket_df.iloc[1, 0])
    u2 = float(main_bucket_df.iloc[2, 0])
    u3 = float(main_bucket_df.iloc[4, 0])
    u4 = float(main_bucket_df.iloc[5, 0])
    units_to_buy = pd.DataFrame(np.array([0, u1, u2, 0, u3, u4, 0]), columns=['units_to_buy'])
    main_bucket_df = pd.concat([main_bucket_df, units_to_buy], axis=1)

    week_date_obj = datetime.strptime(week_date, '%d/%m/%Y')

    # Initialize empty dicts to hold columns
    stop_loss_dict = {}
    take_profit_dict = {}

    for low_daily, high_daily, date_d in zip(daily_lows, daily_highs, daily_dates):
        date_d_obj = datetime.strptime(date_d, '%d/%m/%Y')
        if date_d_obj < week_date_obj:
            continue

        triggering = low_daily < main_bucket_df['units_to_buy']
        triggering = triggering.astype(int)
        main_bucket_df[f'triggering_{date_d}'] = triggering

        stop_loss_col = []
        take_profit_col = []

        for i in range(len(main_bucket_df)):
            buy_price = main_bucket_df.iloc[i, 0]
            units = main_bucket_df.iloc[i, 1]
            stop_val = 0
            take_val = 0

            if units > 0:
                try:
                    stop_loss = sub_bucket_df.iloc[1, i]
                except IndexError:
                    stop_loss = None

                if stop_loss is not None:
                    if high_daily >= buy_price:
                        take_val = 1
                        print(f"{date_d}: 📈 Take profit hit for buy @ {buy_price:.2f}")
                    if low_daily <= stop_loss:
                        stop_val = -1
                        print(f"{date_d}: 🛑 Stop loss hit for buy @ {buy_price:.2f} (stop @ {stop_loss:.2f})")

            stop_loss_col.append(stop_val)
            take_profit_col.append(take_val)

        main_bucket_df[f'stop_loss_{date_d}'] = stop_loss_col
        main_bucket_df[f'take_profit_{date_d}'] = take_profit_col
        stop_loss_dict[date_d] = stop_loss_col
        take_profit_dict[date_d] = take_profit_col

    # Create DataFrames for stop loss and take profit like sub_bucket_df
    stop_loss_df = pd.DataFrame(stop_loss_dict)
    take_profit_df = pd.DataFrame(take_profit_dict)
    # save to csv
    main_bucket_df.to_csv('main_bucket.csv', index=False)
    sub_bucket_df.to_csv('sub_bucket.csv', index=False)
    stop_loss_df.to_csv('stop_loss.csv', index=False)
    take_profit_df.to_csv('take_profit.csv', index=False)
    return main_bucket_df, sub_bucket_df, stop_loss_df, take_profit_df



# Inputs
week_high = 150
week_low = 60
week_date = "01/07/2024"
week_close = 120

daily_lows = [148, 125, 110, 97, 92, 85, 70]
daily_highs = [151, 130, 115, 105, 100, 95, 75]
daily_dates = ["02/07/2024", "03/07/2024", "04/07/2024", "05/07/2024", "06/07/2024", "07/07/2024", "08/07/2024"]

# Call function
results = analyze_single_week_day(
    week_high, week_low, week_date, week_close,
    daily_lows, daily_highs, daily_dates
)
print(results)

