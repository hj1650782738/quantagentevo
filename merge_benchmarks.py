
import pandas as pd
import os

def merge_benchmarks():
    # File paths
    csi500_path = '/home/tjxy/quantagent/AlphaAgent/csi500_close.csv'
    sp500_path = '/home/tjxy/quantagent/AlphaAgent/sp500_close.csv'
    output_path = '/home/tjxy/quantagent/AlphaAgent/benchmark_indices_close.csv'

    print(f"Reading {csi500_path}...")
    df_csi = pd.read_csv(csi500_path)
    if 'SH000905' not in df_csi.columns:
        print("Error: SH000905 not found in CSI500 data")
        return
    
    # Extract date and benchmark
    # Assuming the first column is date, but let's check columns
    # Based on previous pivot, 'date' should be a column or index. 
    # pd.read_csv without index_col will make date a column if it was saved with index=True/False?
    # Previous script: df_wide.to_csv(output_file). The index was 'date'. So read_csv will have 'date' as first column.
    
    df_csi = df_csi[['date', 'SH000905']].copy()
    df_csi['date'] = pd.to_datetime(df_csi['date'])
    df_csi.rename(columns={'SH000905': 'CSI500_Benchmark_SH000905'}, inplace=True)

    print(f"Reading {sp500_path}...")
    df_sp = pd.read_csv(sp500_path)
    if 'SPX' not in df_sp.columns:
        print("Error: SPX not found in S&P500 data")
        return

    df_sp = df_sp[['date', 'SPX']].copy()
    df_sp['date'] = pd.to_datetime(df_sp['date'])
    df_sp.rename(columns={'SPX': 'SP500_Benchmark_SPX'}, inplace=True)

    print("Merging data...")
    # Outer join to keep all dates
    df_merged = pd.merge(df_csi, df_sp, on='date', how='outer')
    
    # Sort by date
    df_merged = df_merged.sort_values('date')
    
    # Save to CSV
    print(f"Saving to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    
    print("Preview:")
    print(df_merged.head())
    print(f"\nSaved {len(df_merged)} rows.")

if __name__ == "__main__":
    merge_benchmarks()
