
import pandas as pd
import yfinance as yf
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from AlphaAgent.backtest_v2.backtest_runner import BacktestRunner

def get_sp500_benchmark():
    print("Downloading SPX data from akshare (.INX)...")
    try:
        import akshare as ak
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
        ret = ret.loc['2022-01-01':'2025-12-26']
        
        return ret
    except Exception as e:
        print(f"Error downloading SPX data: {e}")
        return None

def main():
    # 1. Get Benchmark
    spx_ret = get_sp500_benchmark()
    if spx_ret is None:
        print("Failed to get benchmark data. Aborting.")
        return

    # 2. Setup Backtest Runner
    config_path = Path(__file__).parent / 'backtest_v2/config_sp500.yaml'
    runner = BacktestRunner(str(config_path))
    
    # Define strategies
    strategies = [
        {
            'name': 'custom_123',
            'col_name': 'cumulative_excess_return',
            'source': 'custom',
            'json': [str(Path(__file__).parent / 'all_factors_library_QA_liwei_sp500_123_best_deepseek_aliyun.json')]
        },
        {
            'name': 'alpha158',
            'col_name': 'alpha158',
            'source': 'alpha158',
            'json': None
        },
        {
            'name': 'alpha360', # Assuming alpha60 -> alpha360
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
            # We use output_name to control the CSV filename
            runner.run(
                factor_source=strat['source'],
                factor_json=strat['json'],
                experiment_name=f"exp_sp500_{strat['name']}",
                output_name=strat['name']
            )
            
            # Read the generated CSV
            # Config output_dir is ./backtest_v2_results_sp500 (relative to CWD usually, but config says ./)
            # The runner uses config['experiment']['output_dir']
            output_dir = Path(runner.config['experiment'].get('output_dir', './backtest_v2_results_sp500'))
            
            # Note: BacktestRunner might resolve path relative to where it's run.
            # We will run this script from project root usually.
            
            csv_path = output_dir / f"{strat['name']}_daily_performance.csv"
            
            if not csv_path.exists():
                # Try absolute path if relative failed
                csv_path = Path('/home/tjxy/quantagent/AlphaAgent') / output_dir / f"{strat['name']}_daily_performance.csv"
            
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
    
    # Extend index to cover all results if needed (intersection or union?)
    # Usually we want the backtest period.
    # Backtest period in config: 2022-01-01 to 2025-12-25
    
    combined_df = pd.DataFrame(index=common_index)
    combined_df['spx_return'] = spx_ret
    
    final_data = {}
    
    for col_name, ret_series in results.items():
        # Align dates
        aligned_ret = ret_series.reindex(combined_df.index)
        
        # Calculate daily excess return
        daily_excess = aligned_ret - combined_df['spx_return']
        
        # Fill NaN with 0 (no return / no excess return)
        daily_excess = daily_excess.fillna(0)
        
        # Calculate cumulative excess return (Geometric)
        cum_excess = (1 + daily_excess).cumprod() - 1
        
        final_data[col_name] = cum_excess

    final_df = pd.DataFrame(final_data, index=combined_df.index)
    
    # Filter by date range (optional, but good to clean up)
    final_df = final_df.loc['2022-01-01':'2025-12-26']
    
    # Save
    output_path = Path('/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500/sp500_daily_excess_combined.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index_label='date')
    print(f"\nSuccessfully saved combined results to: {output_path}")
    print(final_df.head())
    print(final_df.tail())

if __name__ == "__main__":
    main()
