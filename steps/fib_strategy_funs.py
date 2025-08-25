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

    lv = main_bucket_df.iloc[6:13, 0].values # take only standard fib levels
    u1_1, u2_2_1, u2_2_2, u3_2_1, u3_2_2, u1_4 = lv[1], lv[2], lv[2], lv[4], lv[4], lv[5] # buy levels
    tp1, tp2_1, tp2_2, tp3_1, tp3_2, tp4 = (
        main_bucket_df.iloc[4,0], main_bucket_df.iloc[6,0], main_bucket_df.iloc[5,0],
        main_bucket_df.iloc[7,0], main_bucket_df.iloc[8,0], main_bucket_df.iloc[9,0]
    ) # take profit levels
    sl1, sl2_1, sl2_2, sl3_1, sl3_2, sl4 = (
        sub_bucket_df.iloc[1,3], sub_bucket_df.iloc[1,3], sub_bucket_df.iloc[1,4],
        sub_bucket_df.iloc[3,5], sub_bucket_df.iloc[3,5], sub_bucket_df.iloc[6,5]
    ) # stop loss levels

    buy_fib_result_df = pd.DataFrame([
        ['buy',1,None,u1_1, *([None]*7)],
        ['buy',2,1,   u2_2_1,*([None]*7)],
        ['buy',2,2,   u2_2_2,*([None]*7)],
        ['buy',3,1,   u3_2_1,*([None]*7)],
        ['buy',3,2,   u3_2_2,*([None]*7)],
        ['buy',4,None,u1_4, *([None]*7)],
    ], columns=['action','unit','sub_unit','unit_value',
                'day1','day2','day3','day4','day5','day6','day7']) # 7 days empty cols

    names = ['1','2_1','2_2','3_1','3_2','4']
    TP = [tp1,tp2_1,tp2_2,tp3_1,tp3_2,tp4]
    SL = [sl1,sl2_1,sl2_2,sl3_1,sl3_2,sl4]
    bought = [False]*6 # to track if buy triggered
    row_status = ['active']*6 # can be 'active', 'sub_closed', 'group_closed'
    group_closed = {2:False, 3:False} # to track if both sub units closed for unit 2 and 3
    pnl_points = ["trade_not_closed"] * 6 # to store PnL points or status

    def mark_pair_conflict(df, col, a, b):
        """Mark conflicts in buy/sell signals for a pair of rows."""
        ca = df.iat[a, col]
        cb = df.iat[b, col]
        if isinstance(ca, str) and isinstance(cb, str):
            pair_conflict = (("sell_profit" in ca and "stop_loss" in cb) or
                             ("stop_loss" in ca and "sell_profit" in cb))
            if pair_conflict:
                if "( error1)" not in ca:
                    df.iat[a, col] = ca + " ( error1)"
                if "( error1)" not in cb:
                    df.iat[b, col] = cb + " ( error1)"

    for day_i, (d, hi, lo) in enumerate(zip(daily_high_low_internal_df['date'],
                                            daily_high_low_internal_df['high'],
                                            daily_high_low_internal_df['low']), start=1):
        col = day_i + 3

        for i in range(6):
            if row_status[i] == 'group_closed' and pd.isna(buy_fib_result_df.iat[i, col]):
                buy_fib_result_df.iat[i, col] = 'trade_closed'

        for i in range(6):
            if row_status[i] != 'active':
                continue

            tp_hit = bought[i] and (hi >= TP[i]) # take profit hit
            sl_hit = bought[i] and (lo <= SL[i]) # stop loss hit

            if tp_hit or sl_hit:
                if tp_hit and sl_hit:
                    buy_fib_result_df.iat[i, col] = f"{TP[i]} sell_profit_{names[i]} ( error1)"
                    pnl_points[i] = round(TP[i] - buy_fib_result_df.iat[i,3], 2)
                elif tp_hit:
                    buy_fib_result_df.iat[i, col] = f"{TP[i]} sell_profit_{names[i]}"
                    pnl_points[i] = round(TP[i] - buy_fib_result_df.iat[i,3], 2)
                else:
                    buy_fib_result_df.iat[i, col] = f"{SL[i]} stop_loss_{names[i]}"
                    pnl_points[i] = round(SL[i] - buy_fib_result_df.iat[i,3], 2)

                row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                continue

            if not bought[i] and lo < buy_fib_result_df.iat[i,3]:
                buy_fib_result_df.iat[i, col] = "buy_triggered"
                bought[i] = True

                # NEW: same-day TP/SL check immediately after trigger (no look-ahead)
                tp_now = (hi >= TP[i])
                sl_now = (lo <= SL[i])
                if tp_now or sl_now:
                    if tp_now and sl_now:
                        buy_fib_result_df.iat[i, col] = f"{TP[i]} sell_profit_{names[i]} ( error1)"
                        pnl_points[i] = round(TP[i] - buy_fib_result_df.iat[i,3], 2)
                    elif tp_now:
                        buy_fib_result_df.iat[i, col] = f"{TP[i]} sell_profit_{names[i]}"
                        pnl_points[i] = round(TP[i] - buy_fib_result_df.iat[i,3], 2)
                    else:
                        buy_fib_result_df.iat[i, col] = f"{SL[i]} stop_loss_{names[i]}"
                        pnl_points[i] = round(SL[i] - buy_fib_result_df.iat[i,3], 2)
                    row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                    continue  # NEW: finish row for the day if same-day exit happens

        mark_pair_conflict(buy_fib_result_df, col, 1, 2)
        mark_pair_conflict(buy_fib_result_df, col, 3, 4)

        if not group_closed[2]:
            sub_done = (row_status[1] in ('sub_closed','group_closed')) + \
                       (row_status[2] in ('sub_closed','group_closed'))
            if sub_done == 2:
                group_closed[2] = True
                for idx in (1,2):
                    row_status[idx] = 'group_closed'
                    if pd.isna(buy_fib_result_df.iat[idx, col]):
                        buy_fib_result_df.iat[idx, col] = 'trade_closed'

        if not group_closed[3]:
            sub_done = (row_status[3] in ('sub_closed','group_closed')) + \
                       (row_status[4] in ('sub_closed','group_closed'))
            if sub_done == 2:
                group_closed[3] = True
                for idx in (3,4):
                    row_status[idx] = 'group_closed'
                    if pd.isna(buy_fib_result_df.iat[idx, col]):
                        buy_fib_result_df.iat[idx, col] = 'trade_closed'

    buy_fib_result_df["pnl_point"] = pnl_points

    levels_df = pd.DataFrame({
        'name'       : ['u1_1','u2_2_1','u2_2_2','u3_2_1','u3_2_2','u1_4'],
        'entry'      : [u1_1,  u2_2_1,  u2_2_2,  u3_2_1,  u3_2_2,  u1_4],
        'take_profit': [tp1,   tp2_1,   tp2_2,   tp3_1,   tp3_2,   tp4],
        'stop_loss'  : [sl1,   sl2_1,   sl2_2,   sl3_1,   sl3_2,   sl4],
    })

    return buy_fib_result_df, levels_df



def calculate_sell_based_fib(main_bucket_df, sub_bucket_df, daily_high_low_internal_df):
    import pandas as pd

    # --- ladder & entries ---
    lv = main_bucket_df.iloc[6:13, 0].values
    u1_1, u2_2, u3_2, u1_4 = lv[1], lv[2], lv[4], lv[5]

    # --- SHORT mapping: TP BELOW entry, SL ABOVE entry ---
    entries = [u1_1, u2_2, u2_2, u3_2, u3_2, u1_4]
    TP = [lv[2], lv[3], lv[4], lv[5], lv[6], lv[6]]    # profit when LOW <= TP
    SL = [lv[0], lv[1], lv[0], lv[3], lv[2], lv[4]]    # loss   when HIGH >= SL

    sell_fib_result_df = pd.DataFrame([
        ['sell',1,None,entries[0], *([None]*7)],
        ['sell',2,1,  entries[1], *([None]*7)],
        ['sell',2,2,  entries[2], *([None]*7)],
        ['sell',3,1,  entries[3], *([None]*7)],
        ['sell',3,2,  entries[4], *([None]*7)],
        ['sell',4,None,entries[5], *([None]*7)],
    ], columns=['action','unit','sub_unit','unit_value','day1','day2','day3','day4','day5','day6','day7'])

    names = ['1','2_1','2_2','3_1','3_2','4']
    sold = [False]*6
    row_status = ['active']*6
    group_closed = {2:False, 3:False}
    pnl_points = ["trade_not_closed"]*6  # single P&L column (pos=profit, neg=loss)

    for day_i, (d, hi, lo) in enumerate(zip(daily_high_low_internal_df['date'],
                                            daily_high_low_internal_df['high'],
                                            daily_high_low_internal_df['low']), start=1):
        col = day_i + 3  # 'day1' offset

        # propagate closure forward
        for i in range(6):
            if row_status[i] == 'group_closed' and pd.isna(sell_fib_result_df.iat[i, col]):
                sell_fib_result_df.iat[i, col] = 'trade_closed'

        # evaluate rows
        for i in range(6):
            if row_status[i] != 'active':
                continue

            # SHORT: TP if lo<=TP, SL if hi>=SL (after sell)
            tp_hit = sold[i] and (lo <= TP[i])
            sl_hit = sold[i] and (hi >= SL[i])

            if tp_hit or sl_hit:
                if tp_hit and sl_hit:
                    sell_fib_result_df.iat[i, col] = f"{TP[i]} buyback_profit_{names[i]} ( error1)"
                    pnl_points[i] = entries[i] - TP[i]  # profit
                elif tp_hit:
                    sell_fib_result_df.iat[i, col] = f"{TP[i]} buyback_profit_{names[i]}"
                    pnl_points[i] = entries[i] - TP[i]  # profit
                else:
                    sell_fib_result_df.iat[i, col] = f"{SL[i]} stop_loss_{names[i]}"
                    pnl_points[i] = entries[i] - SL[i]  # loss (negative)

                row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                continue

            if not sold[i] and hi > sell_fib_result_df.iat[i, 3]:
                sell_fib_result_df.iat[i, col] = "sell_triggered"
                sold[i] = True

                # NEW: same-day TP/SL check immediately after trigger (short side)
                tp_now = (lo <= TP[i])    # short profit if low pierces TP
                sl_now = (hi >= SL[i])    # short loss if high pierces SL
                if tp_now or sl_now:
                    if tp_now and sl_now:
                        sell_fib_result_df.iat[i, col] = f"{TP[i]} buyback_profit_{names[i]} ( error1)"
                        pnl_points[i] = entries[i] - TP[i]
                    elif tp_now:
                        sell_fib_result_df.iat[i, col] = f"{TP[i]} buyback_profit_{names[i]}"
                        pnl_points[i] = entries[i] - TP[i]
                    else:
                        sell_fib_result_df.iat[i, col] = f"{SL[i]} stop_loss_{names[i]}"
                        pnl_points[i] = entries[i] - SL[i]
                    row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                    continue  # NEW: finish row for the day if same-day exit happens

        # cross-subunit same-day conflict marking (ONLY within same unit)
        def mark_pair_conflict(a, b):
            ca = sell_fib_result_df.iat[a, col]
            cb = sell_fib_result_df.iat[b, col]
            if isinstance(ca, str) and isinstance(cb, str):
                pair_conflict = (("buyback_profit" in ca and "stop_loss" in cb) or
                                 ("stop_loss" in ca and "buyback_profit" in cb))
                if pair_conflict:
                    if "( error1)" not in ca:
                        sell_fib_result_df.iat[a, col] = ca + " ( error1)"
                    if "( error1)" not in cb:
                        sell_fib_result_df.iat[b, col] = cb + " ( error1)"

        mark_pair_conflict(1, 2)  # unit 2 sub-units
        mark_pair_conflict(3, 4)  # unit 3 sub-units

        # group completion
        if not group_closed[2]:
            sub_done = (row_status[1] in ('sub_closed','group_closed')) + (row_status[2] in ('sub_closed','group_closed'))
            if sub_done == 2:
                group_closed[2] = True
                for idx in (1,2):
                    row_status[idx] = 'group_closed'
                    if pd.isna(sell_fib_result_df.iat[idx, col]):
                        sell_fib_result_df.iat[idx, col] = 'trade_closed'
        if not group_closed[3]:
            sub_done = (row_status[3] in ('sub_closed','group_closed')) + (row_status[4] in ('sub_closed','group_closed'))
            if sub_done == 2:
                group_closed[3] = True
                for idx in (3,4):
                    row_status[idx] = 'group_closed'
                    if pd.isna(sell_fib_result_df.iat[idx, col]):
                        sell_fib_result_df.iat[idx, col] = 'trade_closed'

    # levels (short) — KEEP EXACT NAMES
    levels_df = pd.DataFrame({
        'name'       : ['u1_1','u2_2_1','u2_2_2','u3_2_1','u3_2_2','u1_4'],
        'entry'      : entries,
        'take_profit': TP,   # short: LOW <= TP
        'stop_loss'  : SL,   # short: HIGH >= SL
    })

    # append single P&L column to results
    sell_fib_result_df['pnl_point'] = pnl_points

    return sell_fib_result_df, levels_df
