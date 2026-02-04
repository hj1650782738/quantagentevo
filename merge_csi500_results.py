
import pandas as pd
from pathlib import Path

# Paths
base_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_csi500")
original_csv_path = base_dir / "csi500_daily_excess.csv"
alpha158_csv_path = base_dir / "csi500_alpha158_daily_performance.csv"
alpha360_csv_path = base_dir / "csi500_alpha360_daily_performance.csv"

# Function to calculate cumulative return
def calc_cumulative(df):
    if 'excess_return' in df.columns:
        return (1 + df['excess_return']).cumprod() - 1
    return None

print("Loading data...")

# Load original CSV (which already has cumulative returns)
if original_csv_path.exists():
    df_main = pd.read_csv(original_csv_path)
    # Rename cumulative column to match request implicitly (it's the main strategy)
    # The user asked to add alpha158 and alpha360 columns to this file
    # Let's assume the existing second column is the main strategy
    print(f"Loaded main CSV: {len(df_main)} rows")
else:
    print(f"Error: Main CSV not found at {original_csv_path}")
    exit(1)

# Load Alpha158
if alpha158_csv_path.exists():
    df_158 = pd.read_csv(alpha158_csv_path)
    df_158['alpha158'] = calc_cumulative(df_158)
    # Keep only date and cumulative return
    df_158 = df_158[['date', 'alpha158']] if 'date' in df_158.columns else df_158.reset_index()[['date', 'alpha158']] if 'date' in df_158.reset_index().columns else None
    
    # Ensure date format consistency if needed, but usually string match is enough for merge
    print(f"Loaded Alpha158: {len(df_158)} rows")
else:
    print(f"Error: Alpha158 CSV not found")
    df_158 = pd.DataFrame(columns=['date', 'alpha158'])

# Load Alpha360
if alpha360_csv_path.exists():
    df_360 = pd.read_csv(alpha360_csv_path)
    df_360['alpha360'] = calc_cumulative(df_360)
    # Keep only date and cumulative return
    df_360 = df_360[['date', 'alpha360']] if 'date' in df_360.columns else df_360.reset_index()[['date', 'alpha360']] if 'date' in df_360.reset_index().columns else None
    
    print(f"Loaded Alpha360: {len(df_360)} rows")
else:
    print(f"Error: Alpha360 CSV not found")
    df_360 = pd.DataFrame(columns=['date', 'alpha360'])

# Merge
# df_main should have 'date' column. Let's verify.
if 'date' not in df_main.columns:
    # If the first column is date but unnamed or named differently
    # Based on previous turn, it is 'date'
    pass

print("Merging data...")
merged_df = df_main.merge(df_158, on='date', how='left')
merged_df = merged_df.merge(df_360, on='date', how='left')

# Save
output_path = base_dir / "csi500_daily_excess_combined.csv"
merged_df.to_csv(output_path, index=False)
print(f"Successfully merged and saved to: {output_path}")
print("\nPreview:")
print(merged_df.head())
print("...")
print(merged_df.tail())
