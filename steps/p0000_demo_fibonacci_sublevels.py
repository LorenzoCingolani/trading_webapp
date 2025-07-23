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
    print(price_levels['standard'])

    # on price_standard use rolling window take first high and second as low and recalculate
    for high,low in zip(price_levels['standard'][:-1], price_levels['standard'][1:]):
        sub_range = high - low
        sub_levels = {
            "sub_standard":(high - (fibonacci_standard * sub_range))
        }
        sub_levels['sub_standard'] = np.append(sub_levels['sub_standard'], low)
        sub_levels['sub_standard'] = np.insert(sub_levels['sub_standard'], 0, high)
        print(sub_levels['sub_standard'])
        print()
    



    return price_levels

if __name__ == "__main__":
    # Example high and low values
    high = 150
    low = 60

    # Calculate Fibonacci levels and sub-levels
    fib_levels = fibonacci_retracement_levels_with_sublevels(high, low)

   