import pandas as pd
from fib_strategy_funs import main_fib_levels_fun, sub_fib_levels_fun, calculate_buy_based_fib
# Weekly data fixed
weekly_high = 150
weekly_low = 60
weekly_date = "01/07/2024"
main_bucket_df = main_fib_levels_fun(weekly_high, weekly_low, weekly_date)
print(main_bucket_df)


main_fib_levels_values = main_bucket_df.iloc[6:13,0].values
sub_bucket_df = sub_fib_levels_fun(main_fib_levels_values)
print(sub_bucket_df)



# Daily data
daily_highs = [151, 130, 115, 150, 100, 200, 75]
daily_lows = [148, 125, 110, 97, 92, 85, 70]
daily_dates = ["02/07/2024", "03/07/2024", "04/07/2024", "05/07/2024", "06/07/2024", "07/07/2024", "08/07/2024"]
daily_high_low_df = pd.DataFrame({'date': daily_dates, 'high': daily_highs, 'low': daily_lows})



# calculate buy based fib
buy_fib_result_df, levels_df = calculate_buy_based_fib(main_bucket_df, sub_bucket_df, daily_high_low_df)
print(buy_fib_result_df)
print(levels_df)


    

    # now alternatively put all dataframe on same sheet row wise with column names too
with pd.ExcelWriter('daily_analysis_combined.xlsx', engine='openpyxl') as writer:
    startrow = 0    

    # Write main_bucket_df
    main_bucket_df.to_excel(writer, sheet_name='fib_buy_sell', startrow=startrow, index=False)
    sub_bucket_df.to_excel(writer, sheet_name='fib_buy_sell', startrow=4, startcol=2, index=False)
    startrow += len(main_bucket_df) +3 


    # Write buy_fib_result_df
    buy_fib_result_df.to_excel(writer, sheet_name='fib_buy_sell', startrow=startrow, index=False)
    startrow += len(buy_fib_result_df) + 2  # Leave one empty row


    # Write daily_high_low_df
    daily_high_low_df.T.to_excel(writer, sheet_name='fib_buy_sell', startrow=startrow, startcol=4, index=False)
    startrow += daily_high_low_df.shape[1] + 2

    # Write levels_df
    levels_df.to_excel(writer, sheet_name='fib_buy_sell', startrow=startrow, index=False)
    startrow += len(levels_df) + 2  # Leave one empty row

