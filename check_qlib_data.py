
from qlib.data import D
import qlib

# Initialize Qlib
provider_uri = "/home/tjxy/.qlib/qlib_data/us_data"
qlib.init(provider_uri=provider_uri, region="us")

# Check instruments
instruments = D.list_instruments(instruments=D.instruments("all"), start_time="2022-01-01", end_time="2022-01-05", as_list=True)
print(f"Total instruments: {len(instruments)}")

# Check for SPX or ^GSPC
potential_benchmarks = ["SPX", "^GSPC", ".INX", "SPY"]
for bench in potential_benchmarks:
    if bench in instruments:
        print(f"Found benchmark in instruments: {bench}")
    else:
        print(f"Benchmark not in instruments: {bench}")

# Try to get features for SPX
try:
    df = D.features(["SPX"], ["$close"], start_time="2022-01-01", end_time="2022-01-05")
    print("Successfully fetched SPX data:")
    print(df)
except Exception as e:
    print(f"Failed to fetch SPX data: {e}")

try:
    df = D.features(["^GSPC"], ["$close"], start_time="2022-01-01", end_time="2022-01-05")
    print("Successfully fetched ^GSPC data:")
    print(df)
except Exception as e:
    print(f"Failed to fetch ^GSPC data: {e}")
