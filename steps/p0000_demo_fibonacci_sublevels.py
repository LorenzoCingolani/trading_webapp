from re import sub
import numpy as np
import pandas as pd

def fibonacci_retracement_levels_with_sublevels(high, low):
    """
    This function calculates Fibonacci retracement levels and their sub-levels based on high and low values.
    It returns a dictionary with high, standard, low Fibonacci levels, and their sub-levels.
    """
    # Fibonacci retracement levels - High values
    fibonacci_high = np.array([2.000, 1.764, 1.618, 1.500, 1.382, 1.236])

    # Fibonacci retracement levels - Standard values
    fibonacci_standard = np.array([0.236, 0.382, 0.500, 0.618, 0.786])

    # Fibonacci retracement levels - Low/Extension values
    fibonacci_low = np.array([1.236, 1.382, 1.500, 1.618, 1.764, 2.000])
    
    range_val = high - low
    print('high' ,high)
    print('low' ,low)
    # Calculate main Fibonacci levels
    price_levels = {
        "high": low + (fibonacci_high * range_val),
        "standard": high - (fibonacci_standard * range_val),
        "low": high - (fibonacci_low * range_val)
    }
    # in array add low  valriable at the end
    price_levels['standard'] = np.append(price_levels['standard'], low)
    # now add high  at the start
    price_levels['standard'] = np.insert(price_levels['standard'], 0, high)
    main_bucket_df = pd.DataFrame(price_levels['standard'], columns=[f'FSL, h={high},l={low}'])
    sub_bucket_df = pd.DataFrame()
    # on price_standard use rolling window take first high and second as low and recalculate
    for high,low in zip(price_levels['standard'][:-1], price_levels['standard'][1:]):
        
        sub_range = high - low
        sub_levels = {
            f"h={high},l={low}":(high - (fibonacci_standard * sub_range))
        }
        sub_levels[f'h={high},l={low}'] = np.append(sub_levels[f'h={high},l={low}'], low)
        sub_levels[f'h={high},l={low}'] = np.insert(sub_levels[f'h={high},l={low}'], 0, high)
    
        sub_bucket_df = pd.concat([sub_bucket_df, pd.DataFrame(sub_levels)], axis=1)
    
       
    
    u1 = float(main_bucket_df.iloc[1, 0])
    u2 = float(main_bucket_df.iloc[2, 0])
    u3 = float(main_bucket_df.iloc[4, 0])
    u4 = float(main_bucket_df.iloc[5, 0])
    

    units_to_buy = pd.DataFrame(np.array([0, u1, u2, 0, u3, u4, 0]),columns=['units_to_buy'])
    main_bucket_df = pd.concat([main_bucket_df, units_to_buy], axis=1)
    
    return main_bucket_df, sub_bucket_df

if __name__ == "__main__":
    # Example high and low values
    data_w = pd.read_csv(r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\steps\GL1_weekly.csv')
    data_d = pd.read_csv(r'C:\Users\eeuma\Desktop\students_clients_data\Lorenzo\trading_webapp\steps\GL1_daily.csv')
    for index, row in data_w.iterrows():
        date_w = row['Date']
        low = row['PX_LOW']
        high = row['PX_HIGH']
        price = row['PX_CLOSE_1D']
        print(f'low: {low}, high: {high}, price: {price}')


        # Calculate Fibonacci levels and sub-levels
        main_bks, sub_bks = fibonacci_retracement_levels_with_sublevels(high, low)
        print(main_bks)
        print(sub_bks)
        for low_daily,high_daily,date_d in zip(data_d['PX_LOW'], data_d['PX_HIGH'],data_d['Date']):
            print(f'Date: {date_d}')
            print(f'low_daily: {low_daily}, high_daily: {high_daily}')
            triggering = low_daily < main_bks['units_to_buy'] 
            if any(triggering):
                print('Executed')
            else:
                print('Not executed')
            input('stop to take next day data, hit enter to continue')

        input('Press Enter to continue...')