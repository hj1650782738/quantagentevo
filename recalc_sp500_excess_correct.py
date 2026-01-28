
import pandas as pd
import akshare as ak
from pathlib import Path

def get_sp500_benchmark(start_date, end_date):
    print(f"Downloading SPX data from akshare (.INX) for {start_date} to {end_date}...")
    try:
        # .INX is the symbol for S&P 500 in akshare (sourced from sina)
        spx = ak.index_us_stock_sina(symbol=".INX")
        if spx is None or spx.empty:
            print("Warning: SPX data is empty.")
            return None
        
        # Columns: date, open, high, low, close, volume, amount
        spx['date'] = pd.to_datetime(spx['date'])
        spx.set_index('date', inplace=True)
        spx.sort_index(inplace=True)
        
        close = spx['close']
        
        # Calculate daily return
        ret = close.pct_change().dropna()
        ret.index = pd.to_datetime(ret.index).normalize()
        
        # Filter for relevant period
        ret = ret.loc[start_date:end_date]
        
        return ret
    except Exception as e:
        print(f"Error downloading SPX data: {e}")
        return None

def main():
    base_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500")
    
    # Files containing ABSOLUTE returns (since Qlib missed the benchmark)
    files = {
        'Quantalpha': base_dir / "Quantalpha_top150_daily_performance.csv",
        'alpha158': base_dir / "alpha158_top150_daily_performance.csv",
        'alpha360': base_dir / "alpha360_top150_daily_performance.csv"
    }
    
    start_date = "2022-01-01"
    end_date = "2025-12-23"
    
    # 1. Get Benchmark
    spx_ret = get_sp500_benchmark(start_date, end_date)
    if spx_ret is None:
        print("Failed to get benchmark data. Aborting.")
        return
        
    print(f"Benchmark data loaded: {len(spx_ret)} days")
    
    combined_data = {}
    
    # Create a DataFrame for alignment
    combined_df = pd.DataFrame(index=spx_ret.index)
    combined_df['spx_return'] = spx_ret
    
    print("\nProcessing SP500 results (Calculating Excess Return = Strategy - Benchmark)...")
    
    for name, path in files.items():
        if not path.exists():
            print(f"Warning: {path} not found. Skipping {name}.")
            continue
            
        print(f"Loading {name} from {path}...")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        
        if 'excess_return' in df.columns:
            # This 'excess_return' is actually ABSOLUTE return because Qlib didn't have SPX data
            strategy_ret = df['excess_return']
        else:
            print(f"Warning: 'excess_return' column not found in {name}.")
            continue
            
        # Align dates
        aligned_strat = strategy_ret.reindex(combined_df.index).fillna(0)
        
        # Calculate TRUE Excess Return
        # Excess = Strategy - Benchmark
        daily_excess = aligned_strat - combined_df['spx_return']
        
        # Fill NaN (e.g. holidays) with 0
        daily_excess = daily_excess.fillna(0)
        
        # Calculate Cumulative Excess Return
        cum_excess = (1 + daily_excess).cumprod() - 1
        
        combined_data[name] = cum_excess
        print(f"  > Final Cumulative Excess for {name}: {cum_excess.iloc[-1]:.4f}")

    # Combine into one DataFrame
    final_df = pd.DataFrame(combined_data)
    
    # Save
    output_path = base_dir / "sp500_top150_excess_comparison.csv"
    final_df.to_csv(output_path, index_label='date')
    print(f"\nSuccessfully saved RECALCULATED results to: {output_path}")
    print("\nPreview:")
    print(final_df.head())
    print("...")
    print(final_df.tail())

if __name__ == "__main__":
    main()
