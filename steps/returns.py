"""Daily % returns that stay correct when PX_CLOSE_1D is back-adjusted.

Back-adjusted prices (Panama method) have correct daily CHANGES but shifted LEVELS, which can even
go negative far back in history. So a % return must divide the adjusted change by the price of the
contract actually held (Carver, Advanced Futures Trading Strategies):

    return_t = (adjusted_t - adjusted_t-1) / current_price_t-1

current_price is the raw price of the held contract: 'near', except on roll dates where it's
'far' (from the close of the roll date you hold the new contract). tools/merge_near_far.py writes it.

Files without a current_price column keep the old calculation (plain pct_change), so existing
input files give exactly the same results as before.
"""
import pandas as pd


def current_price_of(df: pd.DataFrame):
    """The held-contract price column, or None for files that don't have one."""
    if 'current_price' in df.columns:
        return pd.to_numeric(df['current_price'], errors='coerce')
    return None


def pct_returns(px, current_price=None):
    """Daily % return: px change / previous day's held-contract price.

    Without current_price this is the same as px.pct_change(). Works on Series or DataFrames."""
    px = px.apply(pd.to_numeric, errors='coerce') if isinstance(px, pd.DataFrame) \
        else pd.to_numeric(px, errors='coerce')
    if current_price is None:
        return px.pct_change(fill_method=None)
    return px.diff() / current_price.shift(1)


def pct_returns_of(df: pd.DataFrame, price_col: str = 'PX_CLOSE_1D', fallback_col: str = None) -> pd.Series:
    """pct_returns() from df's own columns.

    With current_price: back-adjusted price_col change / current_price.
    Without it: plain pct_change of fallback_col (default price_col) - the old behaviour."""
    held = current_price_of(df)
    if held is None:
        return pct_returns(df[fallback_col or price_col])
    return pct_returns(df[price_col], held)
