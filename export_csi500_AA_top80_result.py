
import pandas as pd
from pathlib import Path

def export_results():
    print("开始导出 CSI500 AA Top 80 回测结果...")
    
    # 1. 设置路径
    result_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_csi500_AA_top80")
    # 注意：文件名通常基于 experiment_name 或 json 文件名，backtest_runner 可能会使用 factor json 的 basename
    # 我们先尝试找 daily_performance.csv
    
    # 查找生成的 daily_performance csv 文件
    csv_files = list(result_dir.glob("*daily_performance.csv"))
    if not csv_files:
        print(f"错误: 在 {result_dir} 中找不到 *daily_performance.csv 文件")
        return
        
    # 取最新的一个（通常只有一个）
    raw_csv_path = sorted(csv_files)[-1]
    print(f"读取原始文件: {raw_csv_path}")
    
    # 输出路径
    output_csv_path = result_dir / "csi500_AA_top80_cumulative_excess.csv"
    
    df = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
    
    # 2. 检查列名
    if 'excess_return' not in df.columns:
        print("错误: CSV 中没有 'excess_return' 列")
        print(df.columns)
        return
        
    # 3. 计算累计超额收益 (几何复利)
    test_start = "2022-01-01"
    test_end = "2025-12-26"
    
    df_test = df.loc[test_start:test_end].copy()
    
    if df_test.empty:
        print("错误: 测试集数据为空")
        return
        
    daily_excess = df_test['excess_return'].fillna(0)
    cumulative_excess = (1 + daily_excess).cumprod() - 1
    
    # 4. 创建最终 DataFrame
    final_df = pd.DataFrame({
        'daily_excess_return': daily_excess,
        'cumulative_excess_return': cumulative_excess
    })
    
    # 5. 保存
    final_df.to_csv(output_csv_path, index_label='date')
    print(f"成功保存累计超额收益文件: {output_csv_path}")
    
    # 6. 打印最后几行预览
    print("\n数据预览 (最后5行):")
    print(final_df.tail())
    
    # 7. 计算并打印最终累计收益率
    final_return = cumulative_excess.iloc[-1]
    print(f"\n最终累计超额收益率: {final_return:.4f} ({final_return*100:.2f}%)")

if __name__ == "__main__":
    export_results()
