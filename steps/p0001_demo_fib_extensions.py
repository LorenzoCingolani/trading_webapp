from re import sub
import numpy as np
import pandas as pd
from datetime import datetime

def fibonacci_retracement_levels_with_sublevels_and_extensions(high, low):
    """
    This function calculates Fibonacci retracement levels, sub-levels, and extensions based on high and low values.
    It returns a dictionary with high, standard, low Fibonacci levels, extensions, and their sub-levels.
    """
    # Fibonacci retracement levels - High values
    fibonacci_high = np.array([2.000, 1.764, 1.618, 1.500, 1.382, 1.236])

    # Fibonacci retracement levels - Standard values
    fibonacci_standard = np.array([0.236, 0.382, 0.500, 0.618, 0.786])

    # Fibonacci retracement levels - Low/Extension values
    fibonacci_low = np.array([1.236, 1.382, 1.500, 1.618, 1.764, 2.000])
    
    # Fibonacci extension levels (for projected price beyond the high)
    fibonacci_extensions = np.array([1.618, 2.618, 4.236])

    range_val = high - low
    print('high' ,high)
    print('low' ,low)
    
    # Calculate main Fibonacci levels
    price_levels = {
        "high": low + (fibonacci_high * range_val),
        "standard": high - (fibonacci_standard * range_val),
        "low": high - (fibonacci_low * range_val),
        "extensions": high + (fibonacci_extensions * range_val)  # Calculating extensions
    }
    
    # Add low to standard and high to the start of standard levels
    price_levels['standard'] = np.append(price_levels['standard'], low)
    price_levels['standard'] = np.insert(price_levels['standard'], 0, high)

    # Create DataFrame for standard levels
    main_bucket_df = pd.DataFrame(price_levels['standard'], columns=[f'FSL, h={high},l={low}'])
    
    # Initialize an empty DataFrame for sub-levels
    sub_bucket_df = pd.DataFrame()

    # Sub-levels calculation using rolling window
    for high, low in zip(price_levels['standard'][:-1], price_levels['standard'][1:]):
        sub_range = high - low
        sub_levels = {
            f"h={high},l={low}": (high - (fibonacci_standard * sub_range))
        }
        sub_levels[f'h={high},l={low}'] = np.append(sub_levels[f'h={high},l={low}'], low)
        sub_levels[f'h={high},l={low}'] = np.insert(sub_levels[f'h={high},l={low}'], 0, high)
        sub_bucket_df = pd.concat([sub_bucket_df, pd.DataFrame(sub_levels)], axis=1)

    # Extract certain Fibonacci levels for further analysis
    u1 = float(main_bucket_df.iloc[1, 0])
    u2 = float(main_bucket_df.iloc[2, 0])
    u3 = float(main_bucket_df.iloc[4, 0])
    u4 = float(main_bucket_df.iloc[5, 0])
    
    # Create a DataFrame for units to buy based on Fibonacci levels
    units_to_buy = pd.DataFrame(np.array([0, u1, u2, 0, u3, u4, 0]), columns=['units_to_buy'])
    main_bucket_df = pd.concat([main_bucket_df, units_to_buy], axis=1)
    
    # Add Fibonacci extensions to the result
    extensions_df = pd.DataFrame(price_levels['extensions'], columns=[f'Fibonacci Extensions, h={high},l={low}'])
    
    return main_bucket_df, sub_bucket_df, extensions_df

def calculate_profit_and_loss(entry_price, exit_price, stop_loss_price):
    """
    This function calculates the profit and loss based on entry, exit, and stop-loss prices.
    """
    profit_loss = exit_price - entry_price
    if profit_loss > 0:
        return profit_loss  # Profit
    else:
        return profit_loss  # Loss (negative value)

if __name__ == "__main__":
    # Example high and low values
    data_w = pd.read_csv(r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\steps\GL1_weekly.csv')
    data_d = pd.read_csv(r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\steps\GL1_daily.csv')
    threshold_Date = datetime.strptime(data_d['Date'].iloc[0], '%d/%m/%Y')  # Assuming you want to start from the first date in daily data

    for index, row in data_w.iterrows():
        date_w = row['Date']
        low = row['PX_LOW']
        high = row['PX_HIGH']
        price = row['PX_CLOSE_1D']
        print(f'low: {low}, high: {high}, price: {price}')

        # Calculate Fibonacci levels, sub-levels, and extensions
        main_bks, sub_bks, extensions_df = fibonacci_retracement_levels_with_sublevels_and_extensions(high, low)
        print(f'weekly_date {date_w}')
        print(main_bks)
        print(sub_bks)
        print(extensions_df)
        
        for low_daily, high_daily, date_d in zip(data_d['PX_LOW'], data_d['PX_HIGH'], data_d['Date']):
            date_d_obj = datetime.strptime(date_d, '%d/%m/%Y')
            if date_d_obj <= threshold_Date:
                continue
            
            print(f'Weekly Date: {date_w}')
            print(f'Daily Date: {date_d}')
            print(f'low_daily: {low_daily}, high_daily: {high_daily}')
            
            # Triggering: Compare low_daily with Fibonacci levels to buy
            triggering = low_daily < main_bks['units_to_buy']
            print(f'Triggering: {triggering}')

            # Profit and Loss Calculation (assuming you have entry price and exit price)
            entry_price = main_bks.iloc[1, 0]  # Example entry at the first Fibonacci level
            exit_price = high_daily  # Exit at the daily high (for example)
            stop_loss_price = main_bks.iloc[0, 0]  # Stop loss at the first Fibonacci level (for example)

            # Calculate P&L for this trade
            pnl = calculate_profit_and_loss(entry_price, exit_price, stop_loss_price)
            print(f'Profit and Loss: {pnl}')

            # Update threshold date
            date_d_obj = datetime.strptime(date_d, '%d/%m/%Y')
            date_w_obj = datetime.strptime(date_w, '%d/%m/%Y')

            if date_d_obj >= date_w_obj:
                threshold_Date = date_d_obj
                print(f'Updating threshold date to: {threshold_Date}')
                break
            
            input('stop to take next day data, hit enter to continue')

        input('Press Enter to continue...')
