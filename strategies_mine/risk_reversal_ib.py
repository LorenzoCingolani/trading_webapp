#!/usr/bin/env python3
"""
risk_reversal_ib.py
Compute 25-delta Risk Reversal (RR) for any underlying using Interactive Brokers (IBKR) via ib_insync.

What it does
------------
1) Connects to TWS/Gateway (paper/live) via ib_insync.
2) Fetches option chain metadata to get available strikes for a chosen expiry.
3) Uses a quick Black–Scholes delta estimate to pick the 25Δ put and 25Δ call strikes (minimizes IB snapshots).
4) Requests snapshot market data for those two options to get mid prices and implied vols (from modelGreeks).
5) Computes:
   - Price-based RR  = Price_put(25Δ) - Price_call(25Δ)
   - Vol-based RR    = IV_call(25Δ) - IV_put(25Δ)   (market convention on vol traders' desks)
6) Prints a concise report and writes a CSV with the results.

Usage
-----
python risk_reversal_ib.py --symbol PLTR --expiry 2025-12-19 --host 127.0.0.1 --port 7497 --client-id 1

Notes
-----
- You need market data permissions in IB for options.
- For equity indices (e.g., SPX), contract definitions differ; you may need to adjust exchange/tradingClass.
- This script uses snapshot requests to stay under pacing. Run sparingly or add pacing control for many underlyings.

Author: ChatGPT
"""

import math
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

from ib_insync import IB, Stock, Index, Option, util

# ---------------- Black–Scholes helpers ----------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def bs_d1(S, K, T, r, q, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float('nan')
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def bs_delta_call(S, K, T, r, q, sigma):
    d1 = bs_d1(S, K, T, r, q, sigma)
    return math.exp(-q * T) * _norm_cdf(d1)

def bs_delta_put(S, K, T, r, q, sigma):
    return bs_delta_call(S, K, T, r, q, sigma) - math.exp(-q * T)

# ---------------- Data classes ----------------

@dataclass
class RRResult:
    symbol: str
    expiry: str
    und_price: float
    call_strike: float
    put_strike: float
    call_mid: float
    put_mid: float
    call_iv: float
    put_iv: float
    rr_price: float
    rr_vol: float

# ---------------- Core logic ----------------

def choose_25d_strikes(strikes, S, T, r, q, sigma_guess=0.6):
    """
    Choose strikes whose BS delta is closest to +0.25 (call) and -0.25 (put).
    Only consider OTM side: call strike >= S, put strike <= S.
    """
    best_call = None
    best_put = None
    min_call_diff = float('inf')
    min_put_diff = float('inf')

    for K in strikes:
        if K <= 0:
            continue
        # Call candidate (OTM/upside)
        if K >= S:
            d_call = bs_delta_call(S, K, T, r, q, sigma_guess)
            diff = abs(d_call - 0.25)
            if diff < min_call_diff:
                min_call_diff = diff
                best_call = K
        # Put candidate (OTM/downside)
        if K <= S:
            d_put = bs_delta_put(S, K, T, r, q, sigma_guess)
            diff = abs(d_put + 0.25)  # put delta is negative
            if diff < min_put_diff:
                min_put_diff = diff
                best_put = K

    return best_call, best_put

def mid_from_ticker(t):
    bid = getattr(t, 'bid', float('nan'))
    ask = getattr(t, 'ask', float('nan'))
    if bid is None or ask is None:
        return float('nan')
    if bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    last = getattr(t, 'last', float('nan'))
    if last and last > 0:
        return float(last)
    return float('nan')

def fetch_rr(ib: IB, symbol: str, expiry: str, is_index: bool=False, currency: str='USD',
             exchange: str='SMART', rf_rate: float=0.04, div_yield: float=0.0, timeout: float=5.0) -> RRResult:
    # 1) Underlying contract & price
    if is_index:
        und = Index(symbol, exchange, currency)
    else:
        und = Stock(symbol, exchange, currency)
    ib.qualifyContracts(und)
    und_ticker = ib.reqMktData(und, '', snapshot=True, regulatorySnapshot=False)
    ib.sleep(timeout)
    S = und_ticker.marketPrice()
    if not S or S <= 0:
        raise RuntimeError(f"Could not get market price for underlying {symbol}.")

    # 2) Option chain metadata
    # secDefOptParams gives available strikes, expirations, etc.
    params = ib.reqSecDefOptParams(und.symbol, '', und.secType, und.conId)
    if not params:
        raise RuntimeError("No option chain metadata returned. Check symbol and permissions.")
    # Match the correct trading class/exchange that contains our target expiry
    chosen = None
    for p in params:
        if expiry in p.expirations:
            chosen = p
            break
    if chosen is None:
        raise RuntimeError(f"Expiry {expiry} not found. Available expiries: e.g. {list(params[0].expirations)[:10]}")

    strikes = sorted([k for k in chosen.strikes if k is not None and k > 0])
    if not strikes:
        raise RuntimeError("No strikes found in chain metadata.")

    # 3) Time to expiry (year fraction; ACT/365)
    # Expiry format 'YYYY-MM-DD'
    dt_exp = datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    T = max((dt_exp - now).days, 0) / 365.0
    if T <= 0:
        raise RuntimeError("Expiry is not in the future.")

    # 4) Pick 25Δ strikes using a sigma guess to minimize snapshots
    call_K, put_K = choose_25d_strikes(strikes, S, T, rf_rate, div_yield, sigma_guess=0.6)
    if call_K is None or put_K is None:
        raise RuntimeError("Could not determine 25Δ strikes.")

    # 5) Build contracts
    call = Option(symbol, expiry, call_K, 'C', exchange, currency)
    put  = Option(symbol, expiry, put_K,  'P', exchange, currency)
    ib.qualifyContracts(call, put)

    # 6) Snapshot market data for the two options
    call_t = ib.reqMktData(call, '', snapshot=True, regulatorySnapshot=False)
    put_t  = ib.reqMktData(put,  '', snapshot=True, regulatorySnapshot=False)
    ib.sleep(timeout)

    call_mid = mid_from_ticker(call_t)
    put_mid  = mid_from_ticker(put_t)

    # 7) Pull modelGreeks for implied vols (if available)
    call_iv = float('nan')
    put_iv  = float('nan')
    if call_t.modelGreeks and call_t.modelGreeks.impliedVol:
        call_iv = call_t.modelGreeks.impliedVol
    if put_t.modelGreeks and put_t.modelGreeks.impliedVol:
        put_iv = put_t.modelGreeks.impliedVol

    # 8) Risk Reversal metrics
    rr_price = (put_mid - call_mid) if (not math.isnan(put_mid) and not math.isnan(call_mid)) else float('nan')
    rr_vol = (call_iv - put_iv) if (not math.isnan(call_iv) and not math.isnan(put_iv)) else float('nan')

    return RRResult(
        symbol=symbol,
        expiry=expiry,
        und_price=S,
        call_strike=call_K,
        put_strike=put_K,
        call_mid=call_mid,
        put_mid=put_mid,
        call_iv=call_iv,
        put_iv=put_iv,
        rr_price=rr_price,
        rr_vol=rr_vol
    )

def main():
    parser = argparse.ArgumentParser(description="Compute 25Δ Risk Reversal (RR) via IBKR")
    parser.add_argument("--symbol", required=True, help="Underlying symbol (e.g., PLTR, AAPL, SPY)")
    parser.add_argument("--expiry", required=True, help="Option expiry YYYY-MM-DD (e.g., 2025-12-19)")
    parser.add_argument("--host", default="127.0.0.1", help="IB host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7497, help="IB port (TWS=7497, Gateway=4002 by default)")
    parser.add_argument("--client-id", type=int, default=1, help="IB client ID (default 1)")
    parser.add_argument("--index", action="store_true", help="Set if underlying is an index (uses Index contract)")
    parser.add_argument("--rf", type=float, default=0.04, help="Risk-free rate (annual, e.g., 0.04 for 4%)")
    parser.add_argument("--div", type=float, default=0.0, help="Dividend yield (annual, e.g., 0.01 for 1%)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for snapshots (default 5)")
    args = parser.parse_args()

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)

    try:
        res = fetch_rr(
            ib=ib,
            symbol=args.symbol.upper(),
            expiry=args.expiry,
            is_index=args.index,
            rf_rate=args.rf,
            div_yield=args.div,
            timeout=args.timeout
        )
    finally:
        ib.disconnect()

    # Print report
    print("\n=== 25Δ Risk Reversal Report ===")
    print(f"Symbol:           {res.symbol}")
    print(f"Expiry:           {res.expiry}")
    print(f"Underlying Px:    {res.und_price:.4f}")
    print(f"25Δ Call Strike:  {res.call_strike}")
    print(f"25Δ Put Strike:   {res.put_strike}")
    print(f"25Δ Call Mid:     {res.call_mid:.4f}" if not math.isnan(res.call_mid) else "25Δ Call Mid:     N/A")
    print(f"25Δ Put Mid:      {res.put_mid:.4f}"  if not math.isnan(res.put_mid)  else "25Δ Put Mid:      N/A")
    print(f"25Δ Call IV:      {res.call_iv:.4%}"  if not math.isnan(res.call_iv)  else "25Δ Call IV:      N/A")
    print(f"25Δ Put IV:       {res.put_iv:.4%}"   if not math.isnan(res.put_iv)   else "25Δ Put IV:       N/A")
    print(f"RR (Price):       {res.rr_price:.4f}" if not math.isnan(res.rr_price) else "RR (Price):       N/A")
    print(f"RR (Vol, C-P):    {res.rr_vol:.2%}"   if not math.isnan(res.rr_vol)   else "RR (Vol, C-P):    N/A")
    print("Convention: Vol-based RR = IV_call(25Δ) - IV_put(25Δ); negative => puts richer\n")

    # Save CSV
    import csv
    csv_path = f"rr_{res.symbol}_{res.expiry}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","expiry","und_price","call_strike","put_strike",
                    "call_mid","put_mid","call_iv","put_iv","rr_price","rr_vol_CminusP"])
        w.writerow([res.symbol, res.expiry, f"{res.und_price:.6f}", res.call_strike, res.put_strike,
                    f"{res.call_mid:.6f}" if not math.isnan(res.call_mid) else "",
                    f"{res.put_mid:.6f}"  if not math.isnan(res.put_mid)  else "",
                    f"{res.call_iv:.6f}"  if not math.isnan(res.call_iv)  else "",
                    f"{res.put_iv:.6f}"   if not math.isnan(res.put_iv)   else "",
                    f"{res.rr_price:.6f}" if not math.isnan(res.rr_price) else "",
                    f"{res.rr_vol:.6f}"   if not math.isnan(res.rr_vol)   else ""])
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    util.startLoop()  # allows running inside notebooks, safe in scripts too
    main()
