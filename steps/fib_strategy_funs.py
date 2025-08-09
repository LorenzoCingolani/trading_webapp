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
    sell_profit_level_1 = main_bucket_df.iloc[4,0] # 138
    sell_profit_level_2_1 = main_bucket_df.iloc[6,0] # 0 percent
    sell_profit_level_2_2 = main_bucket_df.iloc[5,0] # 123 percent
    sell_profit_level_3_1 = main_bucket_df.iloc[7,0] # 23 percent
    sell_profit_level_3_2 = main_bucket_df.iloc[8,0] # 38 percent
    sell_profit_level_4 = main_bucket_df.iloc[9,0]   # 50 percent



    stop_loss_level_1 = sub_bucket_df.iloc[1,3] # one level below 50%
    stop_loss_level_2_1 = sub_bucket_df.iloc[1,3] # one level below 50%
    stop_loss_level_2_2 = sub_bucket_df.iloc[1,4] # one level below 61.8%
    stop_loss_level_3_1 = sub_bucket_df.iloc[3,5] # 3 levels below 76.4%
    stop_loss_level_3_2 = sub_bucket_df.iloc[3,5] # 3 levels below 76.4%
    stop_loss_level_4 = sub_bucket_df.iloc[6,5] # last level (low)



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
    stop_loss1_4_counter = 0 # to be continue

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
        executed_take_profit1 = daily_high >= sell_profit_level_1
        executed_stop_loss1 = daily_low <= stop_loss_level_1
        if executed_take_profit1:
            if take_profit1_1_counter < 1:
                buy_fib_result_df.iloc[0, day + 2] = sell_profit_level_1
                take_profit1_1_counter += 1
        elif executed_stop_loss1:
            if stop_loss1_1_counter < 1:
                buy_fib_result_df.iloc[0, day + 2] = stop_loss_level_1
                stop_loss1_1_counter += 1

        executed_take_profit2 = daily_high >= sell_profit_level_2_1
        executed_stop_loss2 = daily_low <= stop_loss_level_2_1
        if executed_take_profit2:
            if take_profit2_2_counter < 2:
                buy_fib_result_df.iloc[1, day + 2] = sell_profit_level_2_1
                take_profit2_2_counter += 1
        elif executed_stop_loss2:
            if stop_loss2_2_counter < 2:
                buy_fib_result_df.iloc[1, day + 2] = stop_loss_level_2_1
                stop_loss2_2_counter += 1

        executed_take_profit3 = daily_high >= sell_profit_level_3_1
        executed_stop_loss3 = daily_low <= stop_loss_level_3_1
        if executed_take_profit3:
            if take_profit2_3_counter < 2:
                buy_fib_result_df.iloc[2, day + 2] = stop_loss_level_3_1
                take_profit2_3_counter += 1
        elif executed_stop_loss3:
            if stop_loss2_3_counter < 2:
                buy_fib_result_df.iloc[2, day + 2] = stop_loss_level_3_1
                stop_loss2_3_counter += 1

        executed_take_profit4 = daily_high >= sell_profit_level_4
        executed_stop_loss4 = daily_low <= stop_loss_level_4
        if executed_take_profit4:
            if take_profit1_4_counter < 1:
                buy_fib_result_df.iloc[3, day + 2] = sell_profit_level_4
                take_profit1_4_counter += 1
        elif executed_stop_loss4:
            if stop_loss1_4_counter < 1:
                buy_fib_result_df.iloc[3, day + 2] = stop_loss_level_4
                stop_loss1_4_counter += 1


        day += 1
    return buy_fib_result_df




