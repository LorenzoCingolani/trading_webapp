
# Q. is it using tick size and tick value now ? just to make sure???

I’m checking the current control flow directly so I can answer whether tick size/value still affect the active EWMA path or only the other strategies.

No, the active EWMA path is not using TICK_SIZE or TICK_VALUE anymore. The EWMA branch in steps/p1_analysis.py now calls strategies_mine/ewma_no_tick.py, and that module only uses price data plus EWMA scalers.

Those tick fields are still loaded into the control dictionary in p_pages/main_analysis_page.py, but they are still there mainly because other strategy code, like carry, still reads POINT_VALUE and EXCHANGE_RATE. If you want, I can also strip the unused tick fields from the EWMA-specific flow entirely.

# Q. is it treating like cash intrument?

Ans.
Yes, for the EWMA leg it is being treated like a price-only, cash-style series rather than a futures/tick-value instrument.

In strategies_mine/ewma_no_tick.py the forecast is built from `PX_CLOSE_1D` using price changes and percentage returns, with no use of `TICK_SIZE`, `TICK_VALUE`, or `POINT_VALUE`. So the EWMA calculation itself is cash-instrument-like. The main page still loads those tick fields in p_pages/main_analysis_page.py, but the no-tick EWMA path does not consume them.

If you want, I can also remove those tick fields from the EWMA-specific control flow so it is explicit in the app UI and code.


# 🎯 EWMA No-Tick Swap — Complete

| What | Status | Details |
|------|--------|---------|
| **Engine** | ✅ | [p1_analysis.py](steps/p1_analysis.py) → `ewma_no_tick.compute_all_ewma()` |
| **Calc** | ✅ | Price-only, cash-style (no tick/point values) |
| **Constants** | ✅ | Module-scoped: `TRADING_DAYS`, `CAP`, `ALPHA`, `FORECAST_SCALERS` |
| **UI Label** | ✅ | [main_analysis_page.py](p_pages/main_analysis_page.py) shows active engine |

**3 files touched** • **No breaking changes** • **Ready to run**