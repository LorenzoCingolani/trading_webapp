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

    lv = main_bucket_df.iloc[6:13, 0].values
    u1_1, u2_2_1, u2_2_2, u3_2_1, u3_2_2, u1_4 = lv[1], lv[2], lv[2], lv[4], lv[4], lv[5]
    tp1, tp2_1, tp2_2, tp3_1, tp3_2, tp4 = (
        main_bucket_df.iloc[4,0], main_bucket_df.iloc[6,0], main_bucket_df.iloc[5,0],
        main_bucket_df.iloc[7,0], main_bucket_df.iloc[8,0], main_bucket_df.iloc[9,0]
    )
    sl1, sl2_1, sl2_2, sl3_1, sl3_2, sl4 = (
        sub_bucket_df.iloc[1,3], sub_bucket_df.iloc[1,3], sub_bucket_df.iloc[1,4],
        sub_bucket_df.iloc[3,5], sub_bucket_df.iloc[3,5], sub_bucket_df.iloc[6,5]
    )

    buy_fib_result_df = pd.DataFrame([
        ['buy',1,None,u1_1, *([None]*7)],
        ['buy',2,1,   u2_2_1,*([None]*7)],
        ['buy',2,2,   u2_2_2,*([None]*7)],
        ['buy',3,1,   u3_2_1,*([None]*7)],
        ['buy',3,2,   u3_2_2,*([None]*7)],
        ['buy',4,None,u1_4, *([None]*7)],
    ], columns=['action','unit','sub_unit','unit_value',
                'day1','day2','day3','day4','day5','day6','day7'])

    names = ['1','2_1','2_2','3_1','3_2','4']
    TP = [tp1,tp2_1,tp2_2,tp3_1,tp3_2,tp4]
    SL = [sl1,sl2_1,sl2_2,sl3_1,sl3_2,sl4]
    bought = [False]*6
    row_status = ['active']*6
    group_closed = {2:False, 3:False}
    pnl_points = ["trade_not_closed"] * 6

    def mark_pair_conflict(df, col, a, b):
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

            tp_hit = bought[i] and (hi >= TP[i])
            sl_hit = bought[i] and (lo <= SL[i])

            # --- NEW: If trigger + TP/SL on the same day, show combined result ---
            if not bought[i] and lo < buy_fib_result_df.iat[i,3]:
                bought[i] = True
                if hi >= TP[i]:  # Trigger + TP same day
                    buy_fib_result_df.iat[i, col] = f"buy_triggered_{TP[i]}_sell_profit_{names[i]}"
                    pnl_points[i] = round(TP[i] - buy_fib_result_df.iat[i,3], 2)
                    row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                    continue
                elif lo <= SL[i]:  # Trigger + SL same day
                    buy_fib_result_df.iat[i, col] = f"buy_triggered_{SL[i]}_stop_loss_{names[i]}"
                    pnl_points[i] = round(SL[i] - buy_fib_result_df.iat[i,3], 2)
                    row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                    continue
                else:
                    buy_fib_result_df.iat[i, col] = "buy_triggered"
                continue

            # Separate TP/SL after trigger day
            if tp_hit or sl_hit:
                if tp_hit:
                    buy_fib_result_df.iat[i, col] = f"{TP[i]} sell_profit_{names[i]}"
                    pnl_points[i] = round(TP[i] - buy_fib_result_df.iat[i,3], 2)
                else:
                    buy_fib_result_df.iat[i, col] = f"{SL[i]} stop_loss_{names[i]}"
                    pnl_points[i] = round(SL[i] - buy_fib_result_df.iat[i,3], 2)
                row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                continue

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
    ], columns=['action','unit','sub_unit','unit_value',
                'day1','day2','day3','day4','day5','day6','day7'])

    names = ['1','2_1','2_2','3_1','3_2','4']
    sold = [False]*6
    row_status = ['active']*6
    group_closed = {2:False, 3:False}
    pnl_points = ["trade_not_closed"] * 6

    def mark_pair_conflict(df, col, a, b):
        ca = df.iat[a, col]
        cb = df.iat[b, col]
        if isinstance(ca, str) and isinstance(cb, str):
            pair_conflict = (("buyback_profit" in ca and "stop_loss" in cb) or
                             ("stop_loss" in ca and "buyback_profit" in cb))
            if pair_conflict:
                if "( error1)" not in ca:
                    df.iat[a, col] = ca + " ( error1)"
                if "( error1)" not in cb:
                    df.iat[b, col] = cb + " ( error1)"

    for day_i, (d, hi, lo) in enumerate(zip(daily_high_low_internal_df['date'],
                                            daily_high_low_internal_df['high'],
                                            daily_high_low_internal_df['low']), start=1):
        col = day_i + 3

        # propagate closed trades forward
        for i in range(6):
            if row_status[i] == 'group_closed' and pd.isna(sell_fib_result_df.iat[i, col]):
                sell_fib_result_df.iat[i, col] = 'trade_closed'

        for i in range(6):
            if row_status[i] != 'active':
                continue

            tp_hit = sold[i] and (lo <= TP[i])
            sl_hit = sold[i] and (hi >= SL[i])

            # --- NEW: If trigger + TP/SL on the same day, combine into single result ---
            if not sold[i] and hi > sell_fib_result_df.iat[i,3]:
                sold[i] = True
                if lo <= TP[i]:  # Trigger + TP same day
                    sell_fib_result_df.iat[i, col] = f"sell_triggered_{TP[i]}_buyback_profit_{names[i]}"
                    pnl_points[i] = round(entries[i] - TP[i], 2)
                    row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                    continue
                elif hi >= SL[i]:  # Trigger + SL same day
                    sell_fib_result_df.iat[i, col] = f"sell_triggered_{SL[i]}_stop_loss_{names[i]}"
                    pnl_points[i] = round(entries[i] - SL[i], 2)
                    row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                    continue
                else:
                    sell_fib_result_df.iat[i, col] = "sell_triggered"
                continue

            # Separate TP/SL after trigger day
            if tp_hit or sl_hit:
                if tp_hit:
                    sell_fib_result_df.iat[i, col] = f"{TP[i]} buyback_profit_{names[i]}"
                    pnl_points[i] = round(entries[i] - TP[i], 2)
                else:
                    sell_fib_result_df.iat[i, col] = f"{SL[i]} stop_loss_{names[i]}"
                    pnl_points[i] = round(entries[i] - SL[i], 2)
                row_status[i] = 'group_closed' if i in (0,5) else 'sub_closed'
                continue

        mark_pair_conflict(sell_fib_result_df, col, 1, 2)
        mark_pair_conflict(sell_fib_result_df, col, 3, 4)

        if not group_closed[2]:
            sub_done = (row_status[1] in ('sub_closed','group_closed')) + \
                       (row_status[2] in ('sub_closed','group_closed'))
            if sub_done == 2:
                group_closed[2] = True
                for idx in (1,2):
                    row_status[idx] = 'group_closed'
                    if pd.isna(sell_fib_result_df.iat[idx, col]):
                        sell_fib_result_df.iat[idx, col] = 'trade_closed'

        if not group_closed[3]:
            sub_done = (row_status[3] in ('sub_closed','group_closed')) + \
                       (row_status[4] in ('sub_closed','group_closed'))
            if sub_done == 2:
                group_closed[3] = True
                for idx in (3,4):
                    row_status[idx] = 'group_closed'
                    if pd.isna(sell_fib_result_df.iat[idx, col]):
                        sell_fib_result_df.iat[idx, col] = 'trade_closed'

    # levels (short)
    levels_df = pd.DataFrame({
        'name'       : ['u1_1','u2_2_1','u2_2_2','u3_2_1','u3_2_2','u1_4'],
        'entry'      : entries,
        'take_profit': TP,
        'stop_loss'  : SL,
    })

    sell_fib_result_df["pnl_point"] = pnl_points

    return sell_fib_result_df, levels_df



import pandas as pd

def write_block_title(writer, sheet, title, startrow):
    """Write a section title into the Excel sheet."""
    pd.DataFrame({title: [title]}).to_excel(writer, sheet_name=sheet, startrow=startrow, index=False)
    return startrow + 2

def write_df(writer, sheet, df, label, startrow, startcol=0):
    """Write a labeled dataframe into Excel and return updated startrow."""
    pd.DataFrame({label: [label]}).to_excel(writer, sheet_name=sheet, startrow=startrow, startcol=startcol, index=False)
    startrow += 1
    df.to_excel(writer, sheet_name=sheet, startrow=startrow, startcol=startcol, index=False)
    return startrow + (len(df) if len(df) else 1) + 1

def collect_prev_week_business_days(idx, df, n_days=5):
    """Collect indices of the previous n business days."""
    prev = []
    j = idx - 1
    while j >= 0 and len(prev) < n_days:
        if df.loc[j, "date"].dayofweek < 5:
            prev.append(j)
        j -= 1
    prev.sort()
    return prev

def collect_next_week_business_days(idx, df, n_days=5):
    """Collect indices of the next n business days."""
    nxt = []
    j = idx + 1
    while j < len(df) and len(nxt) < n_days:
        if df.loc[j, "date"].dayofweek < 5:
            nxt.append(j)
        j += 1
    return nxt







