
import pandas as pd
import akshare as ak
from pathlib import Path
import time
import sys

def get_sp500_benchmark(start_date, end_date):
    print(f"Downloading SPX data from akshare (.INX) for {start_date} to {end_date}...")
    try:
        spx = ak.index_us_stock_sina(symbol=".INX")
        if spx is None or spx.empty:
            print("Warning: SPX data is empty.")
            return None
        
        spx['date'] = pd.to_datetime(spx['date'])
        spx.set_index('date', inplace=True)
        spx.sort_index(inplace=True)
        
        close = spx['close']
        ret = close.pct_change().dropna()
        ret.index = pd.to_datetime(ret.index).normalize()
        ret = ret.loc[start_date:end_date]
        return ret
    except Exception as e:
        print(f"Error downloading SPX data: {e}")
        return None

def merge_csi500():
    print("\n=== Merging CSI500 Results ===")
    base_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_csi500")
    target_csv = base_dir / "csi500_daily_excess_combined.csv"
    new_result_csv = base_dir / "Alphaagent_csi500_daily_performance.csv"
    
    if not new_result_csv.exists():
        return False

    if not target_csv.exists():
        print(f"Error: Target CSV not found: {target_csv}")
        return True # Stop trying

    print("Loading files...")
    target_df = pd.read_csv(target_csv, index_col=0, parse_dates=True)
    new_df = pd.read_csv(new_result_csv, index_col=0, parse_dates=True)
    
    # Calculate cumulative excess return for new result
    # For CSI500, excess_return is already calculated by Qlib (Strategy - Benchmark)
    if 'excess_return' in new_df.columns:
        daily_excess = new_df['excess_return']
        cum_excess = (1 + daily_excess).cumprod() - 1
        
        # Align with target_df index
        aligned_cum_excess = cum_excess.reindex(target_df.index)
        
        # Add column
        target_df['Alphaagent'] = aligned_cum_excess
        
        # Save
        target_df.to_csv(target_csv, index_label='date')
        print(f"Successfully added 'Alphaagent' to {target_csv}")
        print(target_df.tail())
        return True
    else:
        print("Error: 'excess_return' column not found in new result CSV")
        return True # Stop trying

def merge_sp500():
    print("\n=== Merging SP500 Results ===")
    base_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500")
    target_csv = base_dir / "sp500_top150_excess_comparison.csv"
    new_result_csv = base_dir / "Alphaagent_sp500_daily_performance.csv"
    
    if not new_result_csv.exists():
        return False

    if not target_csv.exists():
        print(f"Error: Target CSV not found: {target_csv}")
        return True # Stop trying

    print("Loading files...")
    target_df = pd.read_csv(target_csv, index_col=0, parse_dates=True)
    new_df = pd.read_csv(new_result_csv, index_col=0, parse_dates=True)
    
    # Date range from target_df
    start_date = target_df.index.min().strftime('%Y-%m-%d')
    end_date = target_df.index.max().strftime('%Y-%m-%d')
    
    # Get Benchmark
    spx_ret = get_sp500_benchmark(start_date, end_date)
    if spx_ret is None:
        print("Failed to get benchmark data.")
        return True # Stop trying
        
    # Align benchmark
    benchmark_series = spx_ret.reindex(target_df.index).fillna(0)
    
    # Calculate cumulative excess return for new result
    # For SP500, new_df['excess_return'] is actually ABSOLUTE return
    if 'excess_return' in new_df.columns:
        daily_absolute = new_df['excess_return'].reindex(target_df.index).fillna(0)
        
        # Calculate Excess = Strategy - Benchmark
        daily_excess = daily_absolute - benchmark_series
        
        cum_excess = (1 + daily_excess).cumprod() - 1
        
        # Add column
        target_df['Alphaagent'] = cum_excess
        
        # Save
        target_df.to_csv(target_csv, index_label='date')
        print(f"Successfully added 'Alphaagent' to {target_csv}")
        print(target_df.tail())
        return True
    else:
        print("Error: 'excess_return' column not found in new result CSV")
        return True # Stop trying

def main():
    print("Waiting for backtest results...")
    
    csi_done = False
    sp_done = False
    
    # Wait loop
    max_retries = 600 # 600 * 10s = 6000s = 100 mins
    for i in range(max_retries):
        if not csi_done:
            if merge_csi500():
                csi_done = True
        
        if not sp_done:
            if merge_sp500():
                sp_done = True
                
        if csi_done and sp_done:
            print("\nAll tasks completed successfully!")
            break
            
        if i % 6 == 0:
            print(f"Waiting... ({i*10}s elapsed)")
            
        time.sleep(10)

if __name__ == "__main__":
    main()
