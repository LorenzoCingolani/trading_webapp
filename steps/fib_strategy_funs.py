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
    """this function takes main bucket, sub bucket, daily highs, daily low and daily dates and calculates the weekly fib triggers and execution
    test if buy is triggered or not then take appropriate action """

    main_fib_unit_levels = main_bucket_df.iloc[6:13,0].values # get unit values 1 2 2 1

    buy_fib_result_df = pd.DataFrame(np.ones([4, 10])*np.nan, columns=['action', 'unit', 'unit_value', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7'])
    
    buy_fib_result_df['action'] = 'buy'
    buy_fib_result_df['unit'] = [1,2,2,1]
    u1_1 = main_fib_unit_levels[1]
    u2_2 = main_fib_unit_levels[2]
    u2_3 = main_fib_unit_levels[4]
    u1_4 = main_fib_unit_levels[5]

    # weekly 138 percent 
    weekly_138_percent = main_bucket_df.iloc[4,0]
    stop_loss_value = sub_bucket_df.iloc[1,4]
    print('weekly_138_percent', weekly_138_percent)
    print('stop_loss_value', stop_loss_value)

    buy_fib_result_df['unit_value'] = [u1_1, u2_2, u2_3, u1_4]


    day = 1
    unit1_1_counter = 0
    unit2_2_counter = 0
    unit2_3_counter = 0
    unit1_4_counter = 0

    take_profit1_1_counter = 0
    take_profit2_2_counter = 0
    take_profit2_3_counter = 0
    take_profit1_4_counter = 0

    stop_loss1_1_counter = 0
    stop_loss2_2_counter = 0
    stop_loss2_3_counter = 0
    stop_loss1_4_counter = 0

    for daily_date, daily_high, daily_low in zip(daily_high_low_internal_df['date'], daily_high_low_internal_df['high'], daily_high_low_internal_df['low']):
        buy_fib_result_df.iloc[:,day+2] = buy_fib_result_df.iloc[:,day+2].astype(object)
        # trade closed
        if stop_loss1_1_counter >=1:
            buy_fib_result_df.iloc[0, day + 2] = 'trade_closed'
            continue
        if stop_loss2_2_counter >=2:
            buy_fib_result_df.iloc[1, day + 2] = 'trade_closed'
            continue
        if stop_loss2_3_counter >=2:
            buy_fib_result_df.iloc[2, day + 2] = 'trade_closed'
            continue
        if stop_loss1_4_counter >=1:
            buy_fib_result_df.iloc[3, day + 2] = 'trade_closed'
            continue



        # buy triggering
        triggering_u1 = daily_low < buy_fib_result_df['unit_value'].loc[0]
        triggering_u2 = daily_low < buy_fib_result_df['unit_value'].loc[1]
        triggering_u3 = daily_low < buy_fib_result_df['unit_value'].loc[2]
        triggering_u4 = daily_low < buy_fib_result_df['unit_value'].loc[3]
        # for i in range(4):
        #     if triggering[i]:
        #         buy_fib_result_df.iloc[i,day+2] = 'buy_triggered'
        if triggering_u1:
            if unit1_1_counter<1: # trigger once
               buy_fib_result_df.iloc[0,day+2] = 'buy_triggered'
               unit1_1_counter += 1 

        if triggering_u2:
            if unit2_2_counter<2:
                buy_fib_result_df.iloc[1,day+2] = 'buy_triggered'
                unit2_2_counter += 1

        if triggering_u3:
            if unit2_3_counter<2:
                buy_fib_result_df.iloc[2,day+2] = 'buy_triggered'
                unit2_3_counter += 1

        if triggering_u4:
            if unit1_4_counter<1:
                buy_fib_result_df.iloc[3,day+2] = 'buy_triggered'
                unit1_4_counter += 1

        # take profit and stop loss
        executed_take_profit1 = daily_high >= weekly_138_percent
        executed_stop_loss1 = daily_low <= stop_loss_value
        if executed_take_profit1:
            if take_profit1_1_counter < 1:
                buy_fib_result_df.iloc[0, day + 2] = 'take_profit_hit'
                take_profit1_1_counter += 1
        elif executed_stop_loss1:
            if stop_loss1_1_counter < 1:
                buy_fib_result_df.iloc[0, day + 2] = 'stop_loss_hit'
                stop_loss1_1_counter += 1

        executed_take_profit2 = daily_high >= weekly_138_percent
        executed_stop_loss2 = daily_low <= stop_loss_value
        if executed_take_profit2:
            if take_profit2_2_counter < 2:
                buy_fib_result_df.iloc[1, day + 2] = 'take_profit_hit'
                take_profit2_2_counter += 1
        elif executed_stop_loss2:
            if stop_loss2_2_counter < 2:
                buy_fib_result_df.iloc[1, day + 2] = 'stop_loss_hit'
                stop_loss2_2_counter += 1

        executed_take_profit3 = daily_high >= weekly_138_percent
        executed_stop_loss3 = daily_low <= stop_loss_value
        if executed_take_profit3:
            if take_profit2_3_counter < 2:
                buy_fib_result_df.iloc[2, day + 2] = 'take_profit_hit'
                take_profit2_3_counter += 1
        elif executed_stop_loss3:
            if stop_loss2_3_counter < 2:
                buy_fib_result_df.iloc[2, day + 2] = 'stop_loss_hit'
                stop_loss2_3_counter += 1

        executed_take_profit4 = daily_high >= weekly_138_percent
        executed_stop_loss4 = daily_low <= stop_loss_value
        if executed_take_profit4:
            if take_profit1_4_counter < 1:
                buy_fib_result_df.iloc[3, day + 2] = 'take_profit_hit'
                take_profit1_4_counter += 1
        elif executed_stop_loss4:
            if stop_loss1_4_counter < 1:
                buy_fib_result_df.iloc[3, day + 2] = 'stop_loss_hit'
                stop_loss1_4_counter += 1


        day += 1
    return buy_fib_result_df





### not confirmed for buy
def calculate_sell_based_fib(main_bucket_df, sub_bucket_df, daily_high_low_internal_df):
    """This function calculates SELL fib triggers and executions based on fib levels."""

    main_fib_unit_levels = main_bucket_df.iloc[6:13, 0].values

    sell_fib_result_df = pd.DataFrame(np.ones([4, 10]) * np.nan, columns=['action', 'unit', 'unit_value', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7'])

    sell_fib_result_df['action'] = 'sell'  # [CHANGED] from 'buy' to 'sell'
    sell_fib_result_df['unit'] = [1, 2, 2, 1]
    u1_1 = main_fib_unit_levels[1]
    u2_2 = main_fib_unit_levels[2]
    u2_3 = main_fib_unit_levels[4]
    u1_4 = main_fib_unit_levels[5]

    weekly_138_percent = main_bucket_df.iloc[4, 0]
    stop_loss_value = sub_bucket_df.iloc[1, 4]

    print('weekly_138_percent', weekly_138_percent)
    print('stop_loss_value', stop_loss_value)

    sell_fib_result_df['unit_value'] = [u1_1, u2_2, u2_3, u1_4]

    day = 1
    unit1_1_counter = 0
    unit2_2_counter = 0
    unit2_3_counter = 0
    unit1_4_counter = 0

    take_profit1_1_counter = 0
    take_profit2_2_counter = 0
    take_profit2_3_counter = 0
    take_profit1_4_counter = 0

    stop_loss1_1_counter = 0
    stop_loss2_2_counter = 0
    stop_loss2_3_counter = 0
    stop_loss1_4_counter = 0

    for daily_date, daily_high, daily_low in zip(daily_high_low_internal_df['date'], daily_high_low_internal_df['high'], daily_high_low_internal_df['low']):
        sell_fib_result_df.iloc[:, day + 2] = sell_fib_result_df.iloc[:, day + 2].astype(object)

        # [SAME AS BUY]: Skip if already stopped
        if stop_loss1_1_counter >= 1:
            sell_fib_result_df.iloc[0, day + 2] = 'trade_closed'
            continue
        if stop_loss2_2_counter >= 2:
            sell_fib_result_df.iloc[1, day + 2] = 'trade_closed'
            continue
        if stop_loss2_3_counter >= 2:
            sell_fib_result_df.iloc[2, day + 2] = 'trade_closed'
            continue
        if stop_loss1_4_counter >= 1:
            sell_fib_result_df.iloc[3, day + 2] = 'trade_closed'
            continue

        # [CHANGED] Triggering based on HIGH instead of LOW for sell
        triggering_u1 = daily_high > sell_fib_result_df['unit_value'].loc[0]
        triggering_u2 = daily_high > sell_fib_result_df['unit_value'].loc[1]
        triggering_u3 = daily_high > sell_fib_result_df['unit_value'].loc[2]
        triggering_u4 = daily_high > sell_fib_result_df['unit_value'].loc[3]

        # [CHANGED] Labels switched to 'sell_triggered'
        if triggering_u1:
            if unit1_1_counter < 1:
                sell_fib_result_df.iloc[0, day + 2] = 'sell_triggered'
                unit1_1_counter += 1

        if triggering_u2:
            if unit2_2_counter < 2:
                sell_fib_result_df.iloc[1, day + 2] = 'sell_triggered'
                unit2_2_counter += 1

        if triggering_u3:
            if unit2_3_counter < 2:
                sell_fib_result_df.iloc[2, day + 2] = 'sell_triggered'
                unit2_3_counter += 1

        if triggering_u4:
            if unit1_4_counter < 1:
                sell_fib_result_df.iloc[3, day + 2] = 'sell_triggered'
                unit1_4_counter += 1

        # [CHANGED] For sell: stop_loss is checked FIRST (priority) and uses HIGH >= stop_loss_value
        #           take_profit is LOW <= weekly_138_percent
        executed_stop_loss1 = daily_high >= stop_loss_value
        executed_take_profit1 = daily_low <= weekly_138_percent
        if executed_stop_loss1:
            if stop_loss1_1_counter < 1:
                sell_fib_result_df.iloc[0, day + 2] = 'stop_loss_hit'
                stop_loss1_1_counter += 1
        elif executed_take_profit1:
            if take_profit1_1_counter < 1:
                sell_fib_result_df.iloc[0, day + 2] = 'take_profit_hit'
                take_profit1_1_counter += 1

        executed_stop_loss2 = daily_high >= stop_loss_value
        executed_take_profit2 = daily_low <= weekly_138_percent
        if executed_stop_loss2:
            if stop_loss2_2_counter < 2:
                sell_fib_result_df.iloc[1, day + 2] = 'stop_loss_hit'
                stop_loss2_2_counter += 1
        elif executed_take_profit2:
            if take_profit2_2_counter < 2:
                sell_fib_result_df.iloc[1, day + 2] = 'take_profit_hit'
                take_profit2_2_counter += 1

        executed_stop_loss3 = daily_high >= stop_loss_value
        executed_take_profit3 = daily_low <= weekly_138_percent
        if executed_stop_loss3:
            if stop_loss2_3_counter < 2:
                sell_fib_result_df.iloc[2, day + 2] = 'stop_loss_hit'
                stop_loss2_3_counter += 1
        elif executed_take_profit3:
            if take_profit2_3_counter < 2:
                sell_fib_result_df.iloc[2, day + 2] = 'take_profit_hit'
                take_profit2_3_counter += 1

        executed_stop_loss4 = daily_high >= stop_loss_value
        executed_take_profit4 = daily_low <= weekly_138_percent
        if executed_stop_loss4:
            if stop_loss1_4_counter < 1:
                sell_fib_result_df.iloc[3, day + 2] = 'stop_loss_hit'
                stop_loss1_4_counter += 1
        elif executed_take_profit4:
            if take_profit1_4_counter < 1:
                sell_fib_result_df.iloc[3, day + 2] = 'take_profit_hit'
                take_profit1_4_counter += 1

        day += 1

    return sell_fib_result_df
