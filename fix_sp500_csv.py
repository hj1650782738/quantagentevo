
import pandas as pd
from pathlib import Path
import os

def main():
    base_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500")
    
    # Files to process
    files = {
        'Quantalpha': base_dir / "Quantalpha_top150_daily_performance.csv",
        'alpha158': base_dir / "alpha158_top150_daily_performance.csv",
        'alpha360': base_dir / "alpha360_top150_daily_performance.csv"
    }
    
    start_date = "2022-01-01"
    end_date = "2025-12-23"
    
    combined_data = {}
    
    print("Processing SP500 results (Fixing double-subtraction issue)...")
    
    for name, path in files.items():
        if not path.exists():
            print(f"Warning: {path} not found. Skipping {name}.")
            continue
            
        print(f"Loading {name} from {path}...")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        
        # In backtest_runner.py, 'excess_return' is calculated as:
        # portfolio_return - benchmark_return - cost
        # So this column ALREADY represents the excess return.
        if 'excess_return' in df.columns:
            daily_excess = df['excess_return']
        else:
            print(f"Warning: 'excess_return' column not found in {name}. Columns: {df.columns}")
            continue
            
        # Filter date range
        daily_excess = daily_excess.loc[start_date:end_date]
        
        # Fill NaN
        daily_excess = daily_excess.fillna(0)
        
        # Calculate Cumulative Excess Return
        # We do NOT subtract benchmark again!
        cum_excess = (1 + daily_excess).cumprod() - 1
        
        combined_data[name] = cum_excess
        print(f"  > Final Cumulative Excess for {name}: {cum_excess.iloc[-1]:.4f}")

    # Combine into one DataFrame
    final_df = pd.DataFrame(combined_data)
    
    # Save
    output_path = base_dir / "sp500_top150_excess_comparison.csv"
    final_df.to_csv(output_path, index_label='date')
    print(f"\nSuccessfully saved fixed results to: {output_path}")
    print("\nPreview:")
    print(final_df.head())
    print("...")
    print(final_df.tail())

if __name__ == "__main__":
    main()
