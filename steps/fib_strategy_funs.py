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

    # ---- TP levels ----
    tp1, tp2_1, tp2_2, tp3_1, tp3_2, tp4 = (
        main_bucket_df.iloc[4,0], main_bucket_df.iloc[6,0], main_bucket_df.iloc[5,0],
        main_bucket_df.iloc[7,0], main_bucket_df.iloc[8,0], main_bucket_df.iloc[9,0]
    )

    # ---- SL levels ----
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

    # per-row state
    bought = [False]*6
    row_status = ['active']*6        # 'active', 'sub_closed', 'group_closed' (for 1 & 4 goes straight to group_closed)

    # group tracking for units with sub-units
    group_open_subs = {2:2, 3:2}     # how many sub-rows still open
    group_closed = {2:False, 3:False}

    for day_i, (d, hi, lo) in enumerate(zip(daily_high_low_internal_df['date'],
                                            daily_high_low_internal_df['high'],
                                            daily_high_low_internal_df['low']), start=1):
        col = day_i + 3  # day1 col offset

        # 1) Propagate only for rows already 'group_closed'
        for i in range(6):
            if row_status[i] == 'group_closed' and buy_fib_result_df.iat[i, col] is None:
                buy_fib_result_df.iat[i, col] = 'trade_closed'

        # 2) Evaluate actions
        for i in range(6):
            if row_status[i] != 'active':
                continue

            # TP -> SL -> Buy (TP/SL only after buy)
            if bought[i] and hi >= TP[i]:
                buy_fib_result_df.iat[i, col] = f"{TP[i]} sell_profit_{names[i]}"
                # mark sub-row done
                if i in (0,5):
                    row_status[i] = 'group_closed'  # units 1 & 4 close fully
                else:
                    row_status[i] = 'sub_closed'
                continue

            if bought[i] and lo <= SL[i]:
                buy_fib_result_df.iat[i, col] = f"{SL[i]} stop_loss_{names[i]}"
                if i in (0,5):
                    row_status[i] = 'group_closed'
                else:
                    row_status[i] = 'sub_closed'
                continue

            if not bought[i] and lo < buy_fib_result_df.iat[i, 3]:
                buy_fib_result_df.iat[i, col] = "buy_triggered"
                bought[i] = True

        # 3) After processing the day, check group completion for unit 2 and 3
        #    Rows: (1,2) -> unit 2; (3,4) -> unit 3
        #    When both sub-rows are 'sub_closed', flip both to 'group_closed' and stamp current day as trade_closed if empty.
        # unit 2
        if not group_closed[2]:
            sub_done = (row_status[1] in ('sub_closed','group_closed')) + (row_status[2] in ('sub_closed','group_closed'))
            if sub_done == 2:
                group_closed[2] = True
                # flip both sub-rows to group_closed and stamp current day
                for idx in (1,2):
                    row_status[idx] = 'group_closed'
                    if buy_fib_result_df.iat[idx, col] is None:
                        buy_fib_result_df.iat[idx, col] = 'trade_closed'
        # unit 3
        if not group_closed[3]:
            sub_done = (row_status[3] in ('sub_closed','group_closed')) + (row_status[4] in ('sub_closed','group_closed'))
            if sub_done == 2:
                group_closed[3] = True
                for idx in (3,4):
                    row_status[idx] = 'group_closed'
                    if buy_fib_result_df.iat[idx, col] is None:
                        buy_fib_result_df.iat[idx, col] = 'trade_closed'

    # levels catalog (unchanged)
    levels_df = pd.DataFrame({
        'name'       : ['u1_1','u2_2_1','u2_2_2','u3_2_1','u3_2_2','u1_4'],
        'entry'      : [u1_1,  u2_2_1,  u2_2_2,  u3_2_1,  u3_2_2,  u1_4],
        'take_profit': [tp1,   tp2_1,   tp2_2,   tp3_1,   tp3_2,   tp4],
        'stop_loss'  : [sl1,   sl2_1,   sl2_2,   sl3_1,   sl3_2,   sl4],
    })

    return buy_fib_result_df, levels_df
