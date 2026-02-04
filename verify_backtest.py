import qlib
from qlib.data import D
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.evaluate import risk_analysis
import pandas as pd
import numpy as np
import time

qlib.init(provider_uri="/home/tjxy/.qlib/qlib_data/cn_data", region="cn")
print("验证回测 (2022-01-01 ~ 2025-12-26)")
print("=" * 60)

# 获取股票列表
instruments = D.instruments("csi300")
stock_list = D.list_instruments(instruments, start_time="2022-01-01", end_time="2025-12-26", as_list=True)
print(f"股票数: {len(stock_list)}")

# 创建预测信号
dates = D.calendar(start_time="2022-01-01", end_time="2025-12-26")
print(f"交易日数: {len(dates)}")

index = pd.MultiIndex.from_product([dates, stock_list], names=['datetime', 'instrument'])
pred = pd.DataFrame({'score': np.random.randn(len(index))}, index=index)
print(f"预测形状: {pred.shape}")

print("\n开始回测...", flush=True)
start = time.time()

portfolio_metric_dict, indicator_dict = qlib_backtest(
    executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
              "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True,
                        "verbose": False, "indicator_config": {"show_indicator": False}}},
    strategy={"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy",
              "kwargs": {"signal": pred, "topk": 50, "n_drop": 5}},
    start_time="2022-01-01", end_time="2025-12-26", account=100000000, benchmark="SH000905",
    exchange_kwargs={"codes": stock_list, "limit_threshold": 0.095, "deal_price": "open",
                     "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5}
)

print(f"\n✓ 回测完成! 耗时: {time.time()-start:.2f}秒")

# 分析结果
if portfolio_metric_dict and "1day" in portfolio_metric_dict:
    report_df = portfolio_metric_dict["1day"][0]
    if 'return' in report_df.columns and 'bench' in report_df.columns:
        excess_return = report_df['return'] - report_df['bench']
        excess_return = excess_return.replace([np.inf, -np.inf], np.nan).dropna()
        
        analysis = risk_analysis(excess_return)
        if isinstance(analysis, pd.DataFrame) and 'risk' in analysis.columns:
            analysis = analysis['risk']
        
        print("\n📊 策略指标:")
        print(f"  年化收益: {analysis['annualized_return']:.6f}")
        print(f"  信息比率: {analysis['information_ratio']:.6f}")
        print(f"  最大回撤: {analysis['max_drawdown']:.6f}")
        if analysis['max_drawdown'] != 0:
            print(f"  Calmar比率: {analysis['annualized_return']/abs(analysis['max_drawdown']):.6f}")
