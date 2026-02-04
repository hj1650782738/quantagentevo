
import argparse
import qlib
from qlib.data import D
import pandas as pd
import os

def export_data(market, provider_uri, region, benchmark, start_time, end_time, output_file):
    print(f"Initializing Qlib with provider: {provider_uri}, region: {region}")
    qlib.init(provider_uri=provider_uri, region=region)

    print(f"Fetching instruments for market: {market}")
    try:
        if market == 'all':
            # Handle 'all' market if necessary, or just pass 'all' if D.instruments supports it
            inst_obj = D.instruments('all')
        else:
            inst_obj = D.instruments(market)
            
        instruments = D.list_instruments(instruments=inst_obj, start_time=start_time, end_time=end_time, as_list=True)
        print(f"Found {len(instruments)} instruments in {market}")
    except Exception as e:
        print(f"Error listing instruments: {e}")
        # Fallback: try to just use the market string if D.features supports it, or maybe it's not a valid group.
        # But based on config it should be.
        return

    # Add benchmark to the list
    all_instruments = instruments + [benchmark]
    
    print(f"Fetching data from {start_time} to {end_time}...")
    # Fetch close price
    # fields = ['$close']
    # Using $close from qlib which is usually adjusted close if the data is adjusted, or raw. 
    # In Qlib, $close is the close price.
    
    try:
        # We fetch in chunks if necessary, but for 500 stocks 4 years it should fit in memory.
        df = D.features(all_instruments, ['$close'], start_time=start_time, end_time=end_time, freq='day')
        
        if df.empty:
            print("No data found!")
            return

        # df has MultiIndex (instrument, datetime) and columns ['$close']
        # Reset index to make it easier to handle
        df = df.reset_index()
        
        # Rename columns
        df.columns = ['instrument', 'date', 'close']
        
        # Pivot to wide format: Index=date, Columns=instrument
        df_wide = df.pivot(index='date', columns='instrument', values='close')
        
        # Save to CSV
        print(f"Saving to {output_file}...")
        df_wide.to_csv(output_file)
        print("Done.")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    
    export_data(args.market, args.provider, args.region, args.benchmark, args.start, args.end, args.output)
