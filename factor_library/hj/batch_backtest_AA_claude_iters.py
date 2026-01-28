#!/usr/bin/env python3
"""
批量回测 AA_claude random iter1-5 因子库
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# 配置
PROJECT_ROOT = Path("/home/tjxy/quantagent/AlphaAgent")
BACKTEST_SCRIPT = PROJECT_ROOT / "backtest_v2" / "run_backtest.py"
CONFIG_FILE = PROJECT_ROOT / "backtest_v2" / "config.yaml"
FACTOR_DIR = PROJECT_ROOT / "factor_library" / "hj"

# 回测任务
BACKTEST_TASKS = [
    {"name": "AA_claude_iter1", "file": "AA_claude_random_iter1_16.json"},
    {"name": "AA_claude_iter2", "file": "AA_claude_random_iter2_16.json"},
    {"name": "AA_claude_iter3", "file": "AA_claude_random_iter3_16.json"},
    {"name": "AA_claude_iter4", "file": "AA_claude_random_iter4_16.json"},
    {"name": "AA_claude_iter5", "file": "AA_claude_random_iter5_16.json"},
]


def run_backtest(task: dict) -> dict:
    """执行单个回测"""
    factor_file = FACTOR_DIR / task["file"]
    task_name = task["name"]
    
    if not factor_file.exists():
        return {
            "name": task_name,
            "success": False,
            "error": f"文件不存在: {factor_file}",
        }
    
    cmd = [
        sys.executable,
        str(BACKTEST_SCRIPT),
        "-c", str(CONFIG_FILE),
        "-s", "custom",
        "-j", str(factor_file),
        "-e", task_name,
    ]
    
    print(f"\n{'='*60}")
    print(f"开始回测: {task_name}")
    print(f"因子文件: {factor_file.name}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # 直接输出到终端
            timeout=1800,  # 30分钟超时
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        return {
            "name": task_name,
            "success": success,
            "duration": duration,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "name": task_name,
            "success": False,
            "error": "超时",
            "duration": 1800,
        }
    except Exception as e:
        return {
            "name": task_name,
            "success": False,
            "error": str(e),
        }


def main():
    print("="*60)
    print("批量回测: AA_claude random iter1-5")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = []
    total_start = time.time()
    
    for i, task in enumerate(BACKTEST_TASKS, 1):
        print(f"\n[{i}/{len(BACKTEST_TASKS)}] {task['name']}")
        result = run_backtest(task)
        results.append(result)
        
        status = "✅" if result["success"] else "❌"
        print(f"{status} {task['name']} 完成 (耗时: {result.get('duration', 0):.1f}s)")
    
    # 汇总
    total_duration = time.time() - total_start
    success_count = sum(1 for r in results if r["success"])
    
    print("\n" + "="*60)
    print("回测汇总")
    print("="*60)
    print(f"{'任务':<25} {'状态':<8} {'耗时':<10}")
    print("-"*45)
    for r in results:
        status = "✅" if r["success"] else "❌"
        duration = f"{r.get('duration', 0):.1f}s"
        print(f"{r['name']:<25} {status:<8} {duration:<10}")
    print("-"*45)
    print(f"成功: {success_count}/{len(results)}")
    print(f"总耗时: {total_duration/60:.1f} 分钟")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

