
import time
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

def monitor_and_export_random80():
    target_dir = Path("/home/tjxy/quantagent/AlphaAgent/backtest_v2_results_csi500_AA_random80")
    expected_process_pattern = "Alphaagent_csi500_AA_random80"
    
    print(f"开始监控回测任务: {expected_process_pattern}")
    
    start_time = time.time()
    
    while True:
        # 1. Check if process is running
        try:
            result = subprocess.run(f"ps aux | grep '{expected_process_pattern}' | grep -v grep", shell=True, stdout=subprocess.PIPE, text=True)
            is_running = bool(result.stdout.strip())
        except Exception as e:
            print(f"Error checking process: {e}")
            is_running = False
            
        # 2. Check if output file exists
        csv_file = None
        if target_dir.exists():
            csv_files = list(target_dir.glob("*daily_performance.csv"))
            if csv_files:
                csv_file = csv_files[0]
        
        if csv_file and csv_file.exists():
            # Check if file size is stable
            size1 = csv_file.stat().st_size
            time.sleep(2)
            size2 = csv_file.stat().st_size
            
            if size1 == size2 and size1 > 0:
                print(f"\n检测到结果文件: {csv_file}")
                process_export(csv_file)
                break
        
        if not is_running and not csv_file:
            print("\n进程已结束，但未找到结果文件！可能回测失败。")
            break
            
        # Wait and retry
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r等待中... 已耗时: {elapsed:.0f}s | 进程运行中: {is_running} | 目录存在: {target_dir.exists()}")
        sys.stdout.flush()
        time.sleep(10)

def process_export(raw_csv_path):
    print("\n开始导出数据...")
    
    try:
        result_dir = raw_csv_path.parent
        output_csv_path = result_dir / "csi500_AA_random80_cumulative_excess.csv"
        
        df = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
        
        if 'excess_return' not in df.columns:
            print("错误: CSV 中没有 'excess_return' 列")
            return
            
        test_start = "2022-01-01"
        test_end = "2025-12-26"
        
        df_test = df.loc[test_start:test_end].copy()
        
        if df_test.empty:
            print("错误: 测试集数据为空")
            return
            
        daily_excess = df_test['excess_return'].fillna(0)
        cumulative_excess = (1 + daily_excess).cumprod() - 1
        
        final_df = pd.DataFrame({
            'daily_excess_return': daily_excess,
            'cumulative_excess_return': cumulative_excess
        })
        
        final_df.to_csv(output_csv_path, index_label='date')
        print(f"成功保存累计超额收益文件: {output_csv_path}")
        
        final_return = cumulative_excess.iloc[-1]
        print(f"最终累计超额收益率: {final_return:.4f} ({final_return*100:.2f}%)")
        
    except Exception as e:
        print(f"导出过程中出错: {e}")

if __name__ == "__main__":
    monitor_and_export_random80()
