
import pandas as pd
import os
from pathlib import Path

# 配置
FACTOR_JSON = "/home/tjxy/quantagent/AlphaAgent/factor_library/AA_top80_RankIC_AA_claude_123_csi300.json"
FACTOR_NAME = Path(FACTOR_JSON).stem
COLUMN_NAME = "AA_top80_CSI300_Transfer"

# 路径配置
PATHS = [
    {
        "name": "CSI500",
        "output_dir": "/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_csi500",
        "target_csv": "/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_csi500/csi500_daily_excess_combined.csv"
    },
    {
        "name": "SP500",
        "output_dir": "/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500",
        "target_csv": "/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_sp500/sp500_top150_excess_comparison.csv"
    }
]

def process_and_merge(output_dir, target_csv, factor_name):
    output_path = Path(output_dir)
    # 查找生成的 CSV
    generated_csv = output_path / f"{factor_name}_daily_performance.csv"
    
    if not generated_csv.exists():
        print(f"❌ 未找到生成的 CSV: {generated_csv}")
        return False
        
    print(f"✅ 找到文件: {generated_csv}")
    
    # 读取数据
    df_new = pd.read_csv(generated_csv, index_col=0, parse_dates=True)
    
    # 计算累计超额收益
    if 'excess_return' not in df_new.columns:
        print(f"❌ 文件中缺少 excess_return 列: {generated_csv}")
        return False
        
    df_new['cumulative_excess'] = (1 + df_new['excess_return']).cumprod() - 1
    
    # 准备合并
    new_series = df_new['cumulative_excess']
    new_series.name = COLUMN_NAME
    
    # 读取目标 CSV
    if os.path.exists(target_csv):
        df_target = pd.read_csv(target_csv, index_col=0, parse_dates=True)
        print(f"  读取现有目标文件 ({df_target.shape})")
        
        # 检查是否已存在该列，如果存在则覆盖
        if COLUMN_NAME in df_target.columns:
            print(f"  ⚠️ 列 {COLUMN_NAME} 已存在，将被覆盖")
            df_target = df_target.drop(columns=[COLUMN_NAME])
            
        df_combined = df_target.join(new_series, how='outer')
    else:
        print("  目标文件不存在，创建新文件")
        df_combined = pd.DataFrame(new_series)
    
    # 保存
    df_combined.to_csv(target_csv)
    print(f"  🎉 合并完成: {target_csv}")
    return True

def main():
    print(f"开始合并数据... (因子: {FACTOR_NAME})")
    
    for item in PATHS:
        print(f"\n处理 {item['name']}...")
        process_and_merge(item['output_dir'], item['target_csv'], FACTOR_NAME)

if __name__ == "__main__":
    main()
