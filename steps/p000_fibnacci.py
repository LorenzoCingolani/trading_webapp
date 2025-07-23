import numpy as np
import pandas as pd

def fibonacci_retracement_levels(high, low):
    """
    This function calculates Fibonacci retracement levels based on high and low values.
    It returns a dictionary with high, standard, and low Fibonacci levels.
    """
    # Fibonacci retracement levels - High values
    fibonacci_high = np.array([
        2.000,
        1.764,
        1.618,
        1.500,
        1.382,
        1.236
    ])

    # Fibonacci retracement levels - Standard values
    fibonacci_standard = np.array([
        0.236,
        0.382,
        0.500,
        0.618,
        0.786
    ])

    # Fibonacci retracement levels - Low/Extension values
    fibonacci_low = np.array([
        1.236,
        1.382,
        1.500,
        1.618,
        1.764,
        2.000
    ])
    
    range_val = high - low

    # Fibonacci retracement levels - Combined
    price_levels = {
        "high": low + (fibonacci_high * range_val),
        "standard": high - (fibonacci_standard * range_val),
        "low": high - (fibonacci_low * range_val)
    }

    return price_levels

def apply_fibonacci_to_dataframe(df, high_col='PX_HIGH', low_col='PX_LOW'):
    """
    Apply Fibonacci retracement levels to each row of a DataFrame.
    
    Parameters:
    df: pandas DataFrame with high and low columns
    high_col: name of the high price column
    low_col: name of the low price column
    
    Returns:
    DataFrame with additional Fibonacci level columns
    """
    # Create a copy of the original DataFrame
    result_df = df.copy()
    
    # Calculate Fibonacci levels for each row
    for index, row in df.iterrows():
        high = row[high_col]
        low = row[low_col]
        
        # Get Fibonacci levels for this row
        fib_levels = fibonacci_retracement_levels(high, low)
        
        # Add high levels columns
        for i, level in enumerate(['2.000', '1.764', '1.618', '1.500', '1.382', '1.236']):
            result_df.loc[index, f'fib_high_{level}'] = fib_levels['high'][i]
        
        # Add standard levels columns
        for i, level in enumerate(['0.236', '0.382', '0.500', '0.618', '0.786']):
            result_df.loc[index, f'fib_std_{level}'] = fib_levels['standard'][i]
        
        # Add low levels columns
        for i, level in enumerate(['1.236', '1.382', '1.500', '1.618', '1.764', '2.000']):
            result_df.loc[index, f'fib_low_{level}'] = fib_levels['low'][i]
    
    return result_df 

if __name__ == "__main__":
    # Test with DataFrame
    data = pd.read_csv(r'C:\Users\loci_\Desktop\trading_webapp\DATA\all_input_files\AD1.csv')
    data = data[['Date', 'PX_HIGH', 'PX_LOW']]

    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data.set_index('Date', inplace=True)
    weekly_data = pd.DataFrame()
    weekly_data['PX_HIGH'] = data.resample('W-FRI')['PX_HIGH'].max()
    weekly_data['PX_LOW'] = data.resample('W-FRI')['PX_LOW'].min()

    
    # # Apply Fibonacci levels to DataFrame
    result_df = apply_fibonacci_to_dataframe(weekly_data)
    print("DataFrame with Fibonacci levels:")
    result_df.to_csv(r'C:\Users\loci_\Desktop\trading_webapp\DATA\output_instruments\AD1_fibonacci_levels2.csv')
    