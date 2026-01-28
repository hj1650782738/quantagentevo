#!/usr/bin/env python3
"""
批量执行分年度回测

功能：
1. 读取配置索引
2. 依次执行每个年份的回测
3. 收集并汇总结果
4. 生成分年度IC对比报告
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YearlyBacktestRunner:
    """分年度回测执行器"""
    
    def __init__(self, config_index_path: str):
        self.config_index_path = Path(config_index_path)
        self.configs = self._load_config_index()
        self.results = []
    
    def _load_config_index(self) -> List[Dict]:
        """加载配置索引"""
        with open(self.config_index_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def run_single_backtest(self, config_info: Dict) -> Optional[Dict]:
        """执行单个回测"""
        from backtest_v2.backtest_runner import BacktestRunner
        
        library = config_info['library']
        year = config_info['year']
        config_path = config_info['config_path']
        factor_json = config_info['factor_json']
        
        print(f"\n{'='*70}")
        print(f"🚀 运行 {library} - {year} 年回测")
        print(f"{'='*70}")
        
        try:
            runner = BacktestRunner(config_path)
            
            output_name = f"{library}_{year}"
            metrics = runner.run(
                factor_source="custom",
                factor_json=[factor_json],
                experiment_name=output_name,
                output_name=output_name
            )
            
            result = {
                "library": library,
                "year": year,
                "metrics": metrics,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"回测失败 {library}-{year}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "library": library,
                "year": year,
                "metrics": {},
                "status": "failed",
                "error": str(e)
            }
    
    def run_all(self, libraries: Optional[List[str]] = None, 
                years: Optional[List[int]] = None) -> List[Dict]:
        """
        执行所有回测
        
        Args:
            libraries: 要运行的因子库列表，None 表示全部
            years: 要运行的年份列表，None 表示全部
        """
        filtered_configs = self.configs
        
        if libraries:
            filtered_configs = [c for c in filtered_configs if c['library'] in libraries]
        if years:
            filtered_configs = [c for c in filtered_configs if c['year'] in years]
        
        print(f"\n📊 待执行回测任务: {len(filtered_configs)} 个")
        for cfg in filtered_configs:
            print(f"  - {cfg['library']}-{cfg['year']}")
        
        start_time = time.time()
        
        for i, config_info in enumerate(filtered_configs):
            print(f"\n[{i+1}/{len(filtered_configs)}]", end="")
            result = self.run_single_backtest(config_info)
            if result:
                self.results.append(result)
        
        total_time = time.time() - start_time
        
        # 保存结果
        self._save_results()
        
        # 生成报告
        self._generate_report(total_time)
        
        return self.results
    
    def _save_results(self):
        """保存结果"""
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / "yearly_backtest_results.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 结果已保存: {results_path}")
    
    def _generate_report(self, total_time: float):
        """生成分析报告"""
        print(f"\n{'='*80}")
        print("📈 分年度 IC 对比报告")
        print(f"{'='*80}")
        
        # 按因子库分组
        by_library = {}
        for r in self.results:
            lib = r['library']
            if lib not in by_library:
                by_library[lib] = {}
            by_library[lib][r['year']] = r['metrics']
        
        # 打印表格
        print(f"\n{'Library':<8} {'Year':<6} {'IC':<12} {'ICIR':<12} {'Rank IC':<12} {'Rank ICIR':<12} {'ARR':<12} {'MDD':<12}")
        print("-" * 90)
        
        for lib in sorted(by_library.keys()):
            for year in sorted(by_library[lib].keys()):
                m = by_library[lib][year]
                ic = m.get('IC', 'N/A')
                icir = m.get('ICIR', 'N/A')
                ric = m.get('Rank IC', 'N/A')
                ricir = m.get('Rank ICIR', 'N/A')
                arr = m.get('annualized_return', 'N/A')
                mdd = m.get('max_drawdown', 'N/A')
                
                ic_str = f"{ic:.6f}" if isinstance(ic, float) else str(ic)
                icir_str = f"{icir:.6f}" if isinstance(icir, float) else str(icir)
                ric_str = f"{ric:.6f}" if isinstance(ric, float) else str(ric)
                ricir_str = f"{ricir:.6f}" if isinstance(ricir, float) else str(ricir)
                arr_str = f"{arr:.4f}" if isinstance(arr, float) else str(arr)
                mdd_str = f"{mdd:.4f}" if isinstance(mdd, float) else str(mdd)
                
                print(f"{lib:<8} {year:<6} {ic_str:<12} {icir_str:<12} {ric_str:<12} {ricir_str:<12} {arr_str:<12} {mdd_str:<12}")
            print()
        
        # IC 变化分析
        print(f"\n{'='*80}")
        print("📊 IC 年度变化分析")
        print(f"{'='*80}")
        
        for lib in sorted(by_library.keys()):
            years = sorted(by_library[lib].keys())
            if len(years) >= 2:
                print(f"\n{lib} 因子库:")
                
                ics = []
                for year in years:
                    ic = by_library[lib][year].get('Rank IC')
                    if isinstance(ic, (int, float)):
                        ics.append((year, ic))
                
                if len(ics) >= 2:
                    for i in range(1, len(ics)):
                        prev_year, prev_ic = ics[i-1]
                        curr_year, curr_ic = ics[i]
                        change = (curr_ic - prev_ic) / abs(prev_ic) * 100 if prev_ic != 0 else 0
                        arrow = "↑" if change > 0 else "↓"
                        color = "\033[92m" if change > 0 else "\033[91m"
                        reset = "\033[0m"
                        print(f"  {prev_year} → {curr_year}: {prev_ic:.6f} → {curr_ic:.6f} ({color}{arrow}{abs(change):.1f}%{reset})")
        
        # 2023年对比分析
        print(f"\n{'='*80}")
        print("🔍 2023年 AA vs QA 对比分析")
        print(f"{'='*80}")
        
        if 'AA' in by_library and 'QA' in by_library:
            aa_2023 = by_library['AA'].get(2023, {})
            qa_2023 = by_library['QA'].get(2023, {})
            
            if aa_2023 and qa_2023:
                aa_ic = aa_2023.get('Rank IC', 0)
                qa_ic = qa_2023.get('Rank IC', 0)
                
                print(f"\n  AA 2023 Rank IC: {aa_ic:.6f}" if isinstance(aa_ic, float) else f"\n  AA 2023 Rank IC: {aa_ic}")
                print(f"  QA 2023 Rank IC: {qa_ic:.6f}" if isinstance(qa_ic, float) else f"  QA 2023 Rank IC: {qa_ic}")
                
                if isinstance(aa_ic, float) and isinstance(qa_ic, float):
                    diff = qa_ic - aa_ic
                    print(f"\n  差异: QA 领先 {diff:.6f} ({diff/abs(aa_ic)*100:.1f}%)" if aa_ic != 0 else f"\n  差异: {diff:.6f}")
        
        print(f"\n⏱️  总耗时: {total_time/60:.1f} 分钟")
        print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分年度回测执行器')
    parser.add_argument('--libraries', '-l', nargs='+', choices=['AA', 'QA'],
                       help='指定因子库 (默认全部)')
    parser.add_argument('--years', '-y', nargs='+', type=int,
                       help='指定年份 (默认全部: 2021-2025)')
    parser.add_argument('--config-index', '-c', type=str,
                       default=None,
                       help='配置索引文件路径')
    
    args = parser.parse_args()
    
    # 默认配置索引路径
    if args.config_index is None:
        args.config_index = Path(__file__).parent.parent / "configs" / "config_index.yaml"
    
    # 检查配置是否存在
    if not Path(args.config_index).exists():
        print("⚠️  配置文件不存在，先生成配置...")
        from generate_yearly_configs import main as generate_configs
        generate_configs()
    
    runner = YearlyBacktestRunner(str(args.config_index))
    runner.run_all(libraries=args.libraries, years=args.years)


if __name__ == "__main__":
    main()

