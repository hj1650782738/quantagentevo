
from qlib.data import D
import qlib
import pandas as pd

# Initialize Qlib for CN data
provider_uri = "/home/tjxy/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region="cn")

print("Checking SH000905 (CSI500) data in Qlib...")

benchmark_code = "SH000905"

try:
    print(f"Fetching data for {benchmark_code}...")
    # Fetch data
    df = D.features([benchmark_code], ["$close", "$open", "$high", "$low"], start_time="2022-01-01", end_time="2022-01-10")
    print(f"\nData for {benchmark_code}:")
    print(df)
    
    if df.empty:
        print("WARNING: Data is empty!")
    else:
        # Check if close price is 0 or NaN
        if (df['$close'] == 0).any():
             print("WARNING: Some close prices are 0!")
        if df['$close'].isna().any():
             print("WARNING: Some close prices are NaN!")
             
        # Calculate returns
        rets = df['$close'].pct_change().dropna()
        print("\nDaily Returns preview:")
        print(rets.head())

except Exception as e:
    print(f"Failed to fetch {benchmark_code} data: {e}")
