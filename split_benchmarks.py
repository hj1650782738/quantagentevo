
import pandas as pd
import os

def split_benchmarks():
    # Source files
    csi500_path = '/home/tjxy/quantagent/AlphaAgent/csi500_close.csv'
    sp500_path = '/home/tjxy/quantagent/AlphaAgent/sp500_close.csv'
    
    # Output files
    output_csi = '/home/tjxy/quantagent/AlphaAgent/csi500_benchmark_only.csv'
    output_sp = '/home/tjxy/quantagent/AlphaAgent/sp500_benchmark_only.csv'

    # Process CSI500
    print(f"Processing CSI500 from {csi500_path}...")
    try:
        df_csi = pd.read_csv(csi500_path)
        if 'SH000905' in df_csi.columns:
            # Select date and benchmark column
            # Ensure date is the first column if it's not named 'date' in csv but was index
            # Based on previous tool output, the first column is date.
            # But let's be safe and assume 'date' is a column if read_csv(index_col=False) default
            # In previous export_prices.py we did df_wide.to_csv(output_file) which saves index. 
            # So the first column in CSV is 'date'.
            
            # Keep only date and benchmark
            df_csi_bench = df_csi[['date', 'SH000905']].copy()
            df_csi_bench['date'] = pd.to_datetime(df_csi_bench['date'])
            df_csi_bench = df_csi_bench.sort_values('date')
            
            # Remove rows with NaN in benchmark (if any)
            df_csi_bench = df_csi_bench.dropna(subset=['SH000905'])
            
            print(f"Saving CSI500 benchmark to {output_csi}...")
            df_csi_bench.to_csv(output_csi, index=False)
            print(f"CSI500 rows: {len(df_csi_bench)}")
        else:
            print("Error: SH000905 not found in CSI500 file")
    except Exception as e:
        print(f"Error processing CSI500: {e}")

    # Process S&P500
    print(f"\nProcessing S&P500 from {sp500_path}...")
    try:
        df_sp = pd.read_csv(sp500_path)
        if 'SPX' in df_sp.columns:
            df_sp_bench = df_sp[['date', 'SPX']].copy()
            df_sp_bench['date'] = pd.to_datetime(df_sp_bench['date'])
            df_sp_bench = df_sp_bench.sort_values('date')
            
            # Remove rows with NaN
            df_sp_bench = df_sp_bench.dropna(subset=['SPX'])
            
            print(f"Saving S&P500 benchmark to {output_sp}...")
            df_sp_bench.to_csv(output_sp, index=False)
            print(f"S&P500 rows: {len(df_sp_bench)}")
        else:
            print("Error: SPX not found in S&P500 file")
    except Exception as e:
        print(f"Error processing S&P500: {e}")

if __name__ == "__main__":
    split_benchmarks()
