
import pandas as pd
import akshare as ak
import numpy as np
from pathlib import Path

def calculate_metrics(daily_returns):
    """Calculate annualized return, max drawdown, and Sharpe ratio."""
    if daily_returns.empty:
        return {}
        
    # Annualized Return
    total_ret = (1 + daily_returns).prod() - 1
    days = len(daily_returns)
    ann_ret = (1 + total_ret) ** (252 / days) - 1
    
    # Max Drawdown
    cum_ret = (1 + daily_returns).cumprod()
    max_cum_ret = cum_ret.cummax()
    drawdowns = cum_ret / max_cum_ret - 1
    max_dd = drawdowns.min()
    
    # Sharpe Ratio (assuming 0 risk-free rate for simplicity of excess return)
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0
    
    return {
        'Annualized Return': ann_ret,
        'Max Drawdown': max_dd,
        'Sharpe Ratio': sharpe,
        'Total Return': total_ret
    }

def get_sp500_benchmark(start_date, end_date):
    print(f"Downloading SPX data from akshare (.INX) for {start_date} to {end_date}...")
    try:
        spx = ak.index_us_stock_sina(symbol=".INX")
        if spx is None or spx.empty:
            return None
        
        spx['date'] = pd.to_datetime(spx['date'])
        spx.set_index('date', inplace=True)
        spx.sort_index(inplace=True)
        close = spx['close']
        ret = close.pct_change().dropna()
        ret.index = pd.to_datetime(ret.index).normalize()
        ret = ret.loc[start_date:end_date]
        return ret
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    base_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500")
    files = {
        'Quantalpha': base_dir / "Quantalpha_top150_daily_performance.csv",
        'alpha158': base_dir / "alpha158_top150_daily_performance.csv",
        'alpha360': base_dir / "alpha360_top150_daily_performance.csv"
    }
    start_date = "2022-01-01"
    end_date = "2025-12-23"
    
    spx_ret = get_sp500_benchmark(start_date, end_date)
    if spx_ret is None:
        return

    print(f"\nBenchmark (SPX) Metrics:")
    spx_metrics = calculate_metrics(spx_ret)
    for k, v in spx_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nStrategy Excess Return Metrics (Recalculated):")
    
    combined_df = pd.DataFrame(index=spx_ret.index)
    combined_df['spx_return'] = spx_ret

    for name, path in files.items():
        if not path.exists():
            continue
            
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if 'excess_return' not in df.columns:
            continue
            
        # Strategy absolute return (as we found Qlib missed benchmark)
        strat_ret = df['excess_return'].reindex(combined_df.index).fillna(0)
        
        # Calculate Excess Return
        excess_ret = strat_ret - combined_df['spx_return']
        excess_ret = excess_ret.fillna(0)
        
        metrics = calculate_metrics(excess_ret)
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
