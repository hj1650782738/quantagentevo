
import pandas as pd
from pathlib import Path

def export_results():
    print("开始导出 CSI500 Round 11 回测结果...")
    
    # 1. 设置路径
    # 原始结果路径 (从回测日志中确认)
    result_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_csi500_round11")
    raw_csv_name = "RANKIC_desc_150_QA_round11_best_claude_123_csi300_daily_performance.csv"
    raw_csv_path = result_dir / raw_csv_name
    
    # 输出路径
    output_csv_path = result_dir / "csi500_round11_cumulative_excess.csv"
    
    if not raw_csv_path.exists():
        print(f"错误: 找不到原始结果文件: {raw_csv_path}")
        return

    print(f"读取原始文件: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
    
    # 2. 检查列名
    if 'excess_return' not in df.columns:
        print("错误: CSV 中没有 'excess_return' 列")
        print(df.columns)
        return
        
    # 3. 计算累计超额收益 (几何复利)
    # 逻辑: (1 + daily_excess).cumprod() - 1
    # 过滤测试集时间范围: 2022-01-01 到 2025-12-26 (回测输出已是此范围，再次确认)
    
    test_start = "2022-01-01"
    test_end = "2025-12-26"
    
    df_test = df.loc[test_start:test_end].copy()
    
    if df_test.empty:
        print("错误: 测试集数据为空")
        return
        
    daily_excess = df_test['excess_return'].fillna(0)
    cumulative_excess = (1 + daily_excess).cumprod() - 1
    
    # 4. 创建最终 DataFrame
    # 格式要求: date, cumulative_excess_return (中文列名?) 用户只说"输出成csv格式给我", "注意输出用中文"可能指回答用中文
    # 为了保险，CSV列名保持英文或拼音，但内容清晰
    
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
