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
# Expiry may be 'YYYY-MM-DD' or 'YYYYMMDD'

Notes
-----
- You need market data permissions in IB for options. If you do not, the script requests delayed data (where allowed).
- For equity indices (e.g., SPX), add --index (contract definition differs).
- Snapshot requests are used; be mindful of pacing limits.

Author: ChatGPT
"""

import math
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from ib_insync import IB, Stock, Index, Option, util

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- Black–Scholes helpers ----------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_d1(S, K, T, r, q, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float('nan')
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def bs_delta_call(S, K, T, r, q, sigma):
    d1 = bs_d1(S, K, T, r, q, sigma)
    # equity-style with dividend yield q
    return math.exp(-q * T) * 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

def bs_delta_put(S, K, T, r, q, sigma):
    return bs_delta_call(S, K, T, r, q, sigma) - math.exp(-q * T)

# ---------------- Data classes ----------------

@dataclass
class RRResult:
    symbol: str
    expiry_iso: str
    expiry_ib: str
    und_price: float
    call_strike: float
    put_strike: float
    call_mid: float
    put_mid: float
    call_iv: float
    put_iv: float
    rr_price: float
    rr_vol: float

# ---------------- Utilities ----------------

def parse_expiry(expiry_str: str):
    """
    Accept 'YYYY-MM-DD' or 'YYYYMMDD'.
    Returns (iso 'YYYY-MM-DD', ib 'YYYYMMDD', dt aware UTC)
    """
    s = expiry_str.strip()
    if '-' in s:
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        ib_fmt = dt.strftime('%Y%m%d')
        iso = dt.strftime('%Y-%m-%d')
    else:
        dt = datetime.strptime(s, '%Y%m%d').replace(tzinfo=timezone.utc)
        ib_fmt = s
        iso = dt.strftime('%Y-%m-%d')
    return iso, ib_fmt, dt

def mid_from_ticker(t):
    bid = getattr(t, 'bid', float('nan'))
    ask = getattr(t, 'ask', float('nan'))
    if bid and ask and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    last = getattr(t, 'last', float('nan'))
    if last and last > 0:
        return float(last)
    return float('nan')

# ---------------- Core logic ----------------

def choose_25d_strikes(strikes, S, T, r, q, sigma_guess=0.6):
    """
    Choose strikes whose BS delta is closest to +0.25 (call) and -0.25 (put).
    Only consider OTM side: call strike >= S, put strike <= S.
    """
    best_call, best_put = None, None
    min_call_diff, min_put_diff = float('inf'), float('inf')

    for K in strikes:
        if K <= 0:
            continue
        if K >= S:  # OTM call
            d_call = bs_delta_call(S, K, T, r, q, sigma_guess)
            diff = abs(d_call - 0.25)
            if diff < min_call_diff:
                min_call_diff = diff
                best_call = K
        if K <= S:  # OTM put
            d_put = bs_delta_put(S, K, T, r, q, sigma_guess)
            diff = abs(d_put + 0.25)  # put delta negative
            if diff < min_put_diff:
                min_put_diff = diff
                best_put = K
    return best_call, best_put

def fetch_rr(
    ib: IB,
    symbol: str,
    expiry: str,
    is_index: bool = False,
    currency: str = 'USD',
    exchange: str = 'SMART',
    rf_rate: float = 0.04,
    div_yield: float = 0.0,
    timeout: float = 5.0
) -> RRResult:
    # Normalize expiry and compute T
    expiry_iso, expiry_ib, dt_exp = parse_expiry(expiry)
    now = datetime.now(timezone.utc)
    T = max((dt_exp - now).days, 0) / 365.0
    if T <= 0:
        raise RuntimeError(f"Expiry {expiry_iso} is not in the future.")

    # 1) Underlying contract & price
    und = Index(symbol, exchange, currency) if is_index else Stock(symbol, exchange, currency)
    # Validate contract before requesting option chain metadata
    try:
        ib.qualifyContracts(und)
        logging.debug(f"Qualified contract: {und}")
    except Exception as e:
        logging.error(f"Failed to qualify contract: {e}")
        raise RuntimeError("Contract qualification failed. Check symbol and exchange.")

    # Debugging: Log conId and secType before reqSecDefOptParams
    logging.debug(f"und.conId: {und.conId} (type: {type(und.conId)})")
    logging.debug(f"und.secType: {und.secType} (type: {type(und.secType)})")

    # Fetch dynamic conId and secType from the qualified contract
    dynamic_conId = und.conId
    dynamic_secType = und.secType

    # Debugging: Log dynamic values
    logging.debug(f"Using dynamic values for reqSecDefOptParams: conId={dynamic_conId}, secType={dynamic_secType}")

    # Fetch option chain metadata using dynamic values
    chains = ib.reqSecDefOptParams(und.symbol, '', dynamic_conId, dynamic_secType)

    # Fetch option chain metadata
    logging.debug(f"Raw option chain metadata: {chains}")

    if not chains:
        logging.error("No option chain metadata returned. Check symbol and permissions.")
        raise RuntimeError("No option chain metadata returned. Check symbol and permissions.")

    # Process option chain metadata
    for chain in chains:
        logging.debug(f"Processing chain: {chain}")
        # p.expirations contains strings like '20251219'
        if expiry_ib in chain.expirations:
            chosen = chain
            break
    else:
        raise RuntimeError(
            f"Expiry {expiry} not found in chain. Example expiries: {sorted(list(chains[0].expirations))[:10]}"
        )

    strikes = sorted([k for k in chosen.strikes if k and k > 0])
    logging.debug(f"Strikes: {strikes}")
    if not strikes:
        raise RuntimeError("No strikes found in chain metadata.")

    # 3) Pick 25Δ strikes using a sigma guess to minimize snapshots
    call_K, put_K = choose_25d_strikes(strikes, spot_price, T, rf_rate, div_yield, sigma_guess=0.6)
    logging.debug(f"Chosen 25Δ call strike: {call_K}, put strike: {put_K}")
    if call_K is None or put_K is None:
        logging.error("Could not determine 25Δ strikes (likely missing spot or extreme chain).")
        raise RuntimeError("Could not determine 25Δ strikes (likely missing spot or extreme chain).")

    logging.debug(f"Determined strikes: {strikes}")

    # 4) Build contracts (use IB-formatted expiry)
    # Use the exchange and multiplier from the chosen option chain metadata
    call = Option(symbol, expiry_ib, call_K, 'C', chosen.exchange, chosen.multiplier)
    put  = Option(symbol, expiry_ib, put_K,  'P', chosen.exchange, chosen.multiplier)
    try:
        ib.qualifyContracts(call, put)
    except Exception as e:
        print(f"DEBUG: Failed to qualify contracts on exchange {chosen.exchange}. Error: {e}")
        # Retry with a fallback exchange if available
        fallback_exchange = 'SMART'
        call.exchange = fallback_exchange
        put.exchange = fallback_exchange
        try:
            ib.qualifyContracts(call, put)
        except Exception as e:
            raise RuntimeError(f"Failed to qualify contracts even with fallback exchange {fallback_exchange}. Error: {e}")

    # 5) Snapshot market data for the two options
    call_t = ib.reqMktData(call, '', snapshot=True, regulatorySnapshot=False)
    put_t  = ib.reqMktData(put,  '', snapshot=True, regulatorySnapshot=False)
    ib.sleep(timeout)

    call_mid = mid_from_ticker(call_t)
    put_mid  = mid_from_ticker(put_t)

    # 6) Implied vols from modelGreeks (if available)
    call_iv = float('nan')
    put_iv  = float('nan')
    if getattr(call_t, 'modelGreeks', None) and call_t.modelGreeks.impliedVol:
        call_iv = float(call_t.modelGreeks.impliedVol)
    if getattr(put_t, 'modelGreeks', None) and put_t.modelGreeks.impliedVol:
        put_iv = float(put_t.modelGreeks.impliedVol)

    # 7) Risk Reversal metrics
    rr_price = (put_mid - call_mid) if (not math.isnan(put_mid) and not math.isnan(call_mid)) else float('nan')
    rr_vol = (call_iv - put_iv) if (not math.isnan(call_iv) and not math.isnan(put_iv)) else float('nan')

    return RRResult(
        symbol=symbol,
        expiry_iso=expiry_iso,
        expiry_ib=expiry_ib,
        und_price=float(S),
        call_strike=float(call_K),
        put_strike=float(put_K),
        call_mid=float(call_mid) if not math.isnan(call_mid) else float('nan'),
        put_mid=float(put_mid) if not math.isnan(put_mid) else float('nan'),
        call_iv=float(call_iv) if not math.isnan(call_iv) else float('nan'),
        put_iv=float(put_iv) if not math.isnan(put_iv) else float('nan'),
        rr_price=float(rr_price) if not math.isnan(rr_price) else float('nan'),
        rr_vol=float(rr_vol) if not math.isnan(rr_vol) else float('nan')
    )

def main():
    parser = argparse.ArgumentParser(description="Compute 25Δ Risk Reversal (RR) via IBKR")
    parser.add_argument("--symbol", required=True, help="Underlying symbol (e.g., PLTR, AAPL, SPY)")
    parser.add_argument("--expiry", required=True, help="Option expiry 'YYYY-MM-DD' or 'YYYYMMDD' (e.g., 2025-12-19)")
    parser.add_argument("--host", default="127.0.0.1", help="IB host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7497, help="IB port (TWS Paper=7497, TWS Live=7496, Gateway Paper=4002, Live=4001)")
    parser.add_argument("--client-id", type=int, default=1, help="IB client ID (default 1)")
    parser.add_argument("--index", action="store_true", help="Set if underlying is an index (uses Index contract)")
    parser.add_argument("--rf", type=float, default=0.04, help="Risk-free rate (annual, e.g., 0.04 for 4%) for 25Δ strike pre-selection")
    parser.add_argument("--div", type=float, default=0.0, help="Dividend yield (annual, e.g., 0.01 for 1%) for 25Δ strike pre-selection")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for snapshots (default 5)")
    args = parser.parse_args()

    ib = IB()
    # Connect and ask for delayed market data if real-time is unavailable
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=20)
    ib.reqMarketDataType(3)  # 1=real-time, 2=frozen, 3=delayed, 4=delayed-frozen

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
    print(f"Expiry (ISO):     {res.expiry_iso}   (IB: {res.expiry_ib})")
    print(f"Underlying Px:    {res.und_price:.4f}")
    print(f"25Δ Call Strike:  {res.call_strike}")
    print(f"25Δ Put  Strike:  {res.put_strike}")
    print(f"25Δ Call Mid:     {res.call_mid:.4f}" if not math.isnan(res.call_mid) else "25Δ Call Mid:     N/A")
    print(f"25Δ Put  Mid:     {res.put_mid:.4f}"  if not math.isnan(res.put_mid)  else "25Δ Put  Mid:     N/A")
    print(f"25Δ Call IV:      {res.call_iv:.4%}"  if not math.isnan(res.call_iv)  else "25Δ Call IV:      N/A")
    print(f"25Δ Put  IV:      {res.put_iv:.4%}"   if not math.isnan(res.put_iv)   else "25Δ Put  IV:      N/A")
    print(f"RR (Price):       {res.rr_price:.4f}" if not math.isnan(res.rr_price) else "RR (Price):       N/A")
    print(f"RR (Vol, C-P):    {res.rr_vol:.2%}"   if not math.isnan(res.rr_vol)   else "RR (Vol, C-P):    N/A")
    print("Convention: Vol-based RR = IV_call(25Δ) - IV_put(25Δ); negative => puts richer\n")

    # Save CSV
    import csv
    csv_path = f"rr_{res.symbol}_{res.expiry_ib}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","expiry_iso","expiry_ib","und_price","call_strike","put_strike",
                    "call_mid","put_mid","call_iv","put_iv","rr_price","rr_vol_CminusP"])
        w.writerow([res.symbol, res.expiry_iso, res.expiry_ib, f"{res.und_price:.6f}", res.call_strike, res.put_strike,
                    f"{res.call_mid:.6f}" if not math.isnan(res.call_mid) else "",
                    f"{res.put_mid:.6f}"  if not math.isnan(res.put_mid)  else "",
                    f"{res.call_iv:.6f}"  if not math.isnan(res.call_iv) else "",
                    f"{res.put_iv:.6f}"   if not math.isnan(res.put_iv)  else "",
                    f"{res.rr_price:.6f}" if not math.isnan(res.rr_price) else "",
                    f"{res.rr_vol:.6f}"   if not math.isnan(res.rr_vol)   else ""])
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    util.startLoop()  # allows running inside notebooks, safe in scripts too
    main()
