from risk_reversal_ib import fetch_rr

def main():
    # Define the values directly in the script
    symbol = "AAPL"
    expiry = "20251219"
    host = "127.0.0.1"
    port = 7497
    client_id = 1

    # Call the fetch_rr function with the defined values
    fetch_rr(symbol=symbol, expiry=expiry, host=host, port=port, client_id=client_id)

if __name__ == "__main__":
    main()