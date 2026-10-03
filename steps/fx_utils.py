import numpy as np
import pandas as pd

# Market quoting convention for a currency pair against USD, decides which way an
# instrument's own-currency value gets converted to USD. Two conventions exist and they are
# NOT interchangeable:
#   DIRECT_QUOTE ("XXXUSD"): USD is the counter currency, e.g. EURUSD=1.13, AUDUSD=0.65 -
#       the raw quote already means "USD per 1 unit of local currency", so converting to USD
#       is a multiply.
#   INDIRECT_QUOTE ("USDXXX"): USD is the base currency, e.g. USDJPY=150, USDCHF=0.90 - the
#       raw quote means "units of local currency per 1 USD", the inverse of what's needed, so
#       converting to USD is a divide.
# EUR/GBP/AUD/NZD are DIRECT_QUOTE; JPY/CHF/CAD/MXN/ZAR/NOK/SEK are INDIRECT_QUOTE. Per-instrument
# convention is read from the 'QUOTE_CONVENTION' column (defaults to DIRECT_QUOTE, matching
# every instrument currently in use, so existing data/behaviour is unchanged).
DIRECT_QUOTE = "XXXUSD"
INDIRECT_QUOTE = "USDXXX"


def to_usd(local_value, exchange_rate, quote_convention=DIRECT_QUOTE):
    """Convert a value denominated in an instrument's local currency into USD.

    exchange_rate is the raw market quote, unmodified (no manual pre-inversion expected) -
    this function picks multiply vs divide based on quote_convention, which may be a single
    string or a pandas Series/array aligned with local_value/exchange_rate for per-row
    (per-instrument) conventions.
    """
    is_indirect = np.asarray(quote_convention) == INDIRECT_QUOTE
    direct_result = local_value * exchange_rate
    indirect_result = local_value / exchange_rate
    result = np.where(is_indirect, indirect_result, direct_result)

    if isinstance(local_value, pd.Series):
        return pd.Series(result, index=local_value.index)
    if isinstance(exchange_rate, pd.Series):
        return pd.Series(result, index=exchange_rate.index)
    return result.item() if np.ndim(result) == 0 else result
