import numpy as np
import pandas as pd



def main_fib_levels_fun(weekly_high, weekly_low, weekly_date):
    """
    This function calculates Fibonacci levels based on weekly high and low values.
    It returns a DataFrame with the calculated levels.
    """
    # Convert weekly date to datetime object
    week_date_obj = pd.to_datetime(weekly_date, format='%d/%m/%Y')

    # Fibonacci retracement levels
    fibonacci_high_extension = np.array([2.000, 1.764, 1.618, 1.500, 1.382, 1.236])
    fibonacci_standard = np.array([0.236, 0.382, 0.500, 0.618, 0.786])
    fibonacci_low_extension = np.array([1.236, 1.382, 1.500, 1.618, 1.764, 2.000])

    # Weekly range
    weekly_range = weekly_high - weekly_low

    # Calculated levels
    high_levels = weekly_low + (fibonacci_high_extension * weekly_range)
    standard_levels = np.hstack(([weekly_high], weekly_high - (fibonacci_standard * weekly_range), [weekly_low]))
    low_levels = weekly_high - (fibonacci_low_extension * weekly_range)

    # Combine all levels into one column
    all_levels = np.concatenate((high_levels, standard_levels, low_levels))

    # Create DataFrame
    data = pd.DataFrame(all_levels, columns=[f'main_level'])

    return data


def sub_fib_levels_fun(range_data):
    '''this function make the sub buckets of fib standard means it further sub divide data'''
    sub_bucket_df = pd.DataFrame()
    col = 1
    for i in range(len(range_data) - 1):
        high = range_data[i]
        low = range_data[i + 1]
        sub_range = high - low
        sub_levels = high - (np.array([0.236, 0.382, 0.500, 0.618, 0.786]) * sub_range)
        sub_levels = np.hstack(([high], sub_levels, [low]))
        sub_bucket_df = pd.concat([sub_bucket_df, pd.DataFrame(sub_levels, columns=[f'sub_bucket_{col}'])], axis=1)
        col += 1

    return sub_bucket_df


def calculate_buy_based_fib(main_bucket_df, sub_bucket_df, daily_high_low_internal_df):
    import pandas as pd

    # ---- entries ----
    lv = main_bucket_df.iloc[6:13, 0].values
    u1_1, u2_2_1, u2_2_2, u3_2_1, u3_2_2, u1_4 = lv[1], lv[2], lv[2], lv[4], lv[4], lv[5]

    # ---- TP ----
    tp1, tp2_1, tp2_2, tp3_1, tp3_2, tp4 = (
        main_bucket_df.iloc[4,0], main_bucket_df.iloc[6,0], main_bucket_df.iloc[5,0],
        main_bucket_df.iloc[7,0], main_bucket_df.iloc[8,0], main_bucket_df.iloc[9,0]
    )

    # ---- SL ----
    sl1, sl2_1, sl2_2, sl3_1, sl3_2, sl4 = (
        sub_bucket_df.iloc[1,3], sub_bucket_df.iloc[1,3], sub_bucket_df.iloc[1,4],
        sub_bucket_df.iloc[3,5], sub_bucket_df.iloc[3,5], sub_bucket_df.iloc[6,5]
    )

    # ---- 6-row result ----
    buy_fib_result_df = pd.DataFrame([
        ['buy',1,None,u1_1, *([None]*7)],
        ['buy',2,1,   u2_2_1,*([None]*7)],
        ['buy',2,2,   u2_2_2,*([None]*7)],
        ['buy',3,1,   u3_2_1,*([None]*7)],
        ['buy',3,2,   u3_2_2,*([None]*7)],
        ['buy',4,None,u1_4, *([None]*7)],
    ], columns=['action','unit','sub_unit','unit_value','day1','day2','day3','day4','day5','day6','day7'])

    names = ['1','2_1','2_2','3_1','3_2','4']
    TP = [tp1,tp2_1,tp2_2,tp3_1,tp3_2,tp4]
    SL = [sl1,sl2_1,sl2_2,sl3_1,sl3_2,sl4]

    bought = [False]*6
    status = ['active']*6  # 'active' or 'closed'

    for day_i, (d, hi, lo) in enumerate(zip(daily_high_low_internal_df['date'],
                                            daily_high_low_internal_df['high'],
                                            daily_high_low_internal_df['low']), start=1):
        col = day_i + 3  # day1 col index offset

        for i in range(6):
            # NEW: propagate closure to future days so it's visible
            if status[i] == 'closed':
                if buy_fib_result_df.iat[i, col] is None:
                    buy_fib_result_df.iat[i, col] = 'trade_closed'
                continue

            # TP -> SL -> Buy (TP/SL only after buy)
            if bought[i] and hi >= TP[i]:
                buy_fib_result_df.iat[i, col] = f"{TP[i]} sell_profit_{names[i]} | trade_closed"
                status[i] = 'closed'
                continue

            if bought[i] and lo <= SL[i]:
                buy_fib_result_df.iat[i, col] = f"{SL[i]} stop_loss_{names[i]} | trade_closed"
                status[i] = 'closed'
                continue

            if not bought[i] and lo < buy_fib_result_df.iat[i, 3]:
                buy_fib_result_df.iat[i, col] = "buy_triggered"
                bought[i] = True

    # OPTIONAL: return also a compact levels catalog for comparison
    levels_df = pd.DataFrame({
        'name'       : ['u1_1','u2_2_1','u2_2_2','u3_2_1','u3_2_2','u1_4'],
        'entry'      : [u1_1, u2_2_1,  u2_2_2,  u3_2_1,  u3_2_2,  u1_4],
        'take_profit': [tp1,  tp2_1,   tp2_2,   tp3_1,   tp3_2,   tp4],
        'stop_loss'  : [sl1,  sl2_1,   sl2_2,   sl3_1,   sl3_2,   sl4],
    })

    return buy_fib_result_df, levels_df
