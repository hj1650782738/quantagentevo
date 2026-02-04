
import akshare as ak
import pandas as pd

def test_sp500_benchmark():
    print("Testing SP500 benchmark retrieval from akshare (.INX)...")
    try:
        # .INX is the symbol for S&P 500 in akshare (sourced from sina)
        spx = ak.index_us_stock_sina(symbol=".INX")
        
        if spx is None or spx.empty:
            print("Error: SPX data is empty or None.")
            return
            
        print(f"Successfully retrieved {len(spx)} rows.")
        
        # Columns: date, open, high, low, close, volume, amount
        print("Columns:", spx.columns.tolist())
        
        spx['date'] = pd.to_datetime(spx['date'])
        spx.set_index('date', inplace=True)
        spx.sort_index(inplace=True)
        
        # Check date range
        print(f"Date range: {spx.index.min()} to {spx.index.max()}")
        
        # Filter for relevant period
        target_start = '2022-01-01'
        target_end = '2025-12-26'
        spx_filtered = spx.loc[target_start:target_end]
        
        print(f"\nData in target range ({target_start} to {target_end}):")
        if spx_filtered.empty:
             print("Warning: No data in target range!")
        else:
             print(f"Row count: {len(spx_filtered)}")
             print("First 5 rows:")
             print(spx_filtered[['close']].head())
             print("\nLast 5 rows:")
             print(spx_filtered[['close']].tail())
             
             # Check for missing values in close price
             missing = spx_filtered['close'].isna().sum()
             print(f"\nMissing close prices: {missing}")
             
             # Calculate daily return to ensure it works
             ret = spx_filtered['close'].pct_change().dropna()
             print(f"\nCalculated daily returns. Rows: {len(ret)}")
             print("First 5 returns:")
             print(ret.head())

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sp500_benchmark()
