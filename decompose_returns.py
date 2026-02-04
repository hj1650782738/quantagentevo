
import pandas as pd
import akshare as ak
import numpy as np
from pathlib import Path

def analyze_returns():
    print("Analyzing Returns Decomposition...")
    
    # 1. Load Strategy Excess Returns from CSV
    csv_path = '/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500/sp500_top150_excess_comparison.csv'
    if not Path(csv_path).exists():
        print("CSV file not found.")
        return
        
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # df contains Cumulative Excess Returns: (1+excess).cumprod() - 1
    
    # Recover daily excess returns
    # daily_excess = (1 + cum_excess) / (1 + cum_excess.shift(1)) - 1
    daily_excess = (1 + df).pct_change().fillna(0)
    daily_excess.iloc[0] = df.iloc[0] # First day
    
    # 2. Load Benchmark (SP500)
    print("Downloading SP500 data...")
    try:
        spx = ak.index_us_stock_sina(symbol=".INX")
        spx['date'] = pd.to_datetime(spx['date'])
        spx.set_index('date', inplace=True)
        spx.sort_index(inplace=True)
        spx_close = spx['close']
        
        # Calculate daily benchmark returns
        bench_ret = spx_close.pct_change().dropna()
        bench_ret.index = pd.to_datetime(bench_ret.index).normalize()
        
        # Align dates
        common_index = df.index.intersection(bench_ret.index)
        daily_excess = daily_excess.loc[common_index]
        bench_ret = bench_ret.loc[common_index]
        
    except Exception as e:
        print(f"Error getting benchmark: {e}")
        return

    # 3. Calculate Absolute Returns
    # Absolute = Excess + Benchmark (Approx for simple subtraction, but correctly: (1+r_strat) = (1+r_excess)*(1+r_bench) if geometric? 
    # Usually in Qlib backtest: excess = portfolio - bench. So portfolio = excess + bench.
    
    strategies = ['alpha158', 'alpha360', 'Quantalpha']
    
    print("\n" + "="*100)
    print(f"{'Metric':<20} | {'Benchmark (SP500)':<18} | {'alpha158':<15} | {'alpha360':<15} | {'Quantalpha':<15}")
    print("-" * 100)
    
    # Annualized Return Calculation (Mean * 252)
    ann_factor = 252
    
    # Benchmark Stats
    bench_ann_ret = bench_ret.mean() * ann_factor
    bench_total_ret = (1 + bench_ret).prod() - 1
    
    print(f"{'Ann. Return (Abs)':<20} | {bench_ann_ret:<18.4%} | ", end="")
    
    results = {}
    
    for strat in strategies:
        if strat not in daily_excess.columns:
            print(f"{'N/A':<15} | ", end="")
            continue
            
        strat_excess = daily_excess[strat]
        strat_abs = strat_excess + bench_ret
        
        # Stats
        abs_ann = strat_abs.mean() * ann_factor
        excess_ann = strat_excess.mean() * ann_factor
        
        results[strat] = {
            'abs_ann': abs_ann,
            'excess_ann': excess_ann,
            'total_abs': (1 + strat_abs).prod() - 1,
            'total_excess': (1 + strat_excess).prod() - 1
        }
        
        print(f"{abs_ann:<15.4%} | ", end="")
        
    print("\n" + "-" * 100)
    
    print(f"{'Ann. Return (Exc)':<20} | {'0.00%':<18} | ", end="")
    for strat in strategies:
        if strat in results:
            print(f"{results[strat]['excess_ann']:<15.4%} | ", end="")
    print("\n" + "-" * 100)
    
    print(f"{'Total Return (Abs)':<20} | {bench_total_ret:<18.4%} | ", end="")
    for strat in strategies:
        if strat in results:
            print(f"{results[strat]['total_abs']:<15.4%} | ", end="")
    print("\n" + "="*100)

if __name__ == "__main__":
    analyze_returns()
