import numpy as np
import pandas as pd

def fibonacci_retracement_levels_with_sublevels(high, low, num_sublevels=3):
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

    # Calculate sub-levels for each range
    sub_levels = {
        "high_sub": [
            np.linspace(low, level, num=num_sublevels + 2)[1:-1]  # Exclude low and level
            for level in price_levels["high"]
        ],
        "standard_sub": [
            np.linspace(high, level, num=num_sublevels + 2)[1:-1]  # Exclude high and level
            for level in price_levels["standard"]
        ],
        "low_sub": [
            np.linspace(high, level, num=num_sublevels + 2)[1:-1]  # Exclude high and level
            for level in price_levels["low"]
        ]
    }

    return {**price_levels, **sub_levels}

if __name__ == "__main__":
    # Example high and low values
    high = 150
    low = 60

    # Calculate Fibonacci levels and sub-levels
    fib_levels = fibonacci_retracement_levels_with_sublevels(high, low, num_sublevels=3)

    # Display the results
    print("Fibonacci Levels:")
    for key, values in fib_levels.items():
        print(f"{key}: {values}")
