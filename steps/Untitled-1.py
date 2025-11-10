


import pandas as pd
import glob

def compute_transition_probabilities(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)

    # Weekly Resample
    weekly = pd.DataFrame()
    weekly['Open'] = df['PX_OPEN'].resample('W-FRI').first()
    weekly['Close'] = df['PX_CLOSE_1D'].resample('W-FRI').last()

    # Determine bullish weeks
    weekly['Bullish'] = weekly['Close'] > weekly['Open']
    weekly['Next_Bullish'] = weekly['Bullish'].shift(-1)

    total_bullish = weekly['Bullish'].sum()
    total_bearish = (~weekly['Bullish']).sum()

    bullish_to_bullish = weekly[weekly['Bullish']]['Next_Bullish'].sum() / total_bullish
    bearish_to_bullish = weekly[~weekly['Bullish']]['Next_Bullish'].sum() / total_bearish

    return bullish_to_bullish, bearish_to_bullish

# --- Main Execution ---

# Assuming all your CSV files are in the 'data/' folder
file_list = glob.globfile_list = glob.glob(r'C:\Users\loci_\Desktop\trading_webapp\*.csv')  # Change folder path accordingly

results = []

for file in file_list:
    security_name = file.split('/')[-1].split('.')[0]
    bullish_bullish, bearish_bullish = compute_transition_probabilities(file)
    results.append({
        'Security': security_name,
        'Bullish->Bullish': bullish_bullish,
        'Bearish->Bullish': bearish_bullish
    })

# Compile into DataFrame
result_df = pd.DataFrame(results)
print(result_df)

# Optional: Save to CSV
result_df.to_csv('Weekly_Transition_Probabilities.csv', index=False)
