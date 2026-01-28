
import pandas as pd
import akshare as ak
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from AlphaAgent.backtest_v2.backtest_runner import BacktestRunner

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
    start_date = "2022-01-01"
    end_date = "2025-12-21"
    
    # 1. Get Benchmark
    spx_ret = get_sp500_benchmark(start_date, end_date)
    if spx_ret is None:
        print("Failed to get benchmark data. Aborting.")
        return

    # 2. Setup Backtest Runner
    config_path = Path(__file__).parent / 'backtest_v2/config_sp500.yaml'
    runner = BacktestRunner(str(config_path))
    
    # Define strategies
    # User requested:
    # 1. backtest_v2/top150_factors_sp500.json -> Label: Quantalpha
    # 2. alpha158 -> Label: alpha158
    # 3. alpha360 -> Label: alpha360
    
    json_path = Path(__file__).parent / 'backtest_v2/top150_factors_sp500.json'
    
    strategies = [
        {
            'name': 'Quantalpha',
            'col_name': 'Quantalpha',
            'source': 'custom',
            'json': [str(json_path)]
        },
        {
            'name': 'alpha158',
            'col_name': 'alpha158',
            'source': 'alpha158',
            'json': None
        },
        {
            'name': 'alpha360',
            'col_name': 'alpha360',
            'source': 'alpha360',
            'json': None
        }
    ]
    
    results = {}
    
    # 3. Run Backtests
    for strat in strategies:
        print(f"\nRunning backtest for {strat['name']}...")
        try:
            # Check if json file exists for custom strategy
            if strat['source'] == 'custom':
                 if not Path(strat['json'][0]).exists():
                     print(f"Error: Custom factor file not found at {strat['json'][0]}")
                     continue

            # We use output_name to control the CSV filename
            runner.run(
                factor_source=strat['source'],
                factor_json=strat['json'],
                experiment_name=f"exp_sp500_{strat['name']}_top150", # Distinct experiment name
                output_name=f"{strat['name']}_top150" # Distinct output name
            )
            
            # Read the generated CSV
            output_dir = Path(runner.config['experiment'].get('output_dir', './backtest_v2_results_sp500'))
            csv_path = output_dir / f"{strat['name']}_top150_daily_performance.csv"
            
            if not csv_path.exists():
                # Try absolute path if relative failed
                csv_path = Path('/home/tjxy/quantagent/AlphaAgent') / output_dir / f"{strat['name']}_top150_daily_performance.csv"
            
            if csv_path.exists():
                print(f"Loading results from {csv_path}")
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                # Since benchmark is null in config, 'excess_return' is actually 'portfolio_return'
                ret_series = df['excess_return']
                ret_series.index = pd.to_datetime(ret_series.index).normalize()
                results[strat['col_name']] = ret_series
            else:
                print(f"Error: Output file not found at {csv_path}")
                
        except Exception as e:
            print(f"Error running {strat['name']}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Calculate Cumulative Excess Returns and Merge
    if not results:
        print("No results to merge.")
        return

    print("\nCalculating cumulative excess returns...")
    
    # Use SPX index as base
    common_index = spx_ret.index
    
    combined_df = pd.DataFrame(index=common_index)
    combined_df['spx_return'] = spx_ret
    
    final_data = {}
    
    for col_name, ret_series in results.items():
        # Align dates
        aligned_ret = ret_series.reindex(combined_df.index)
        
        # Calculate daily excess return
        # For SP500, we must verify if 'excess_return' is absolute or excess.
        # Based on analysis, Qlib's SP500 results with missing benchmark data in binary 
        # often default to absolute returns.
        # We explicitly subtract the downloaded SPX benchmark to be safe.
        if 'spx_return' in combined_df.columns:
             daily_excess = aligned_ret - combined_df['spx_return'].fillna(0)
        else:
             daily_excess = aligned_ret
        
        # Fill NaN with 0 (no return / no excess return)
        daily_excess = daily_excess.fillna(0)
        
        # Calculate cumulative excess return (Geometric)
        cum_excess = (1 + daily_excess).cumprod() - 1
        
        final_data[col_name] = cum_excess

    final_df = pd.DataFrame(final_data, index=combined_df.index)
    
    # Filter by date range
    final_df = final_df.loc[start_date:end_date]
    
    # Save
    output_path = Path('/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500/sp500_top150_excess_comparison.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index_label='date')
    print(f"\nSuccessfully saved combined results to: {output_path}")
    print(final_df.head())
    print(final_df.tail())

if __name__ == "__main__":
    main()
