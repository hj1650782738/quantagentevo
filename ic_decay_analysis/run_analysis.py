#!/usr/bin/env python3
"""
IC衰减分析主运行脚本

一键执行完整的分析流程：
1. 生成分年度回测配置
2. 执行分年度回测
3. 执行因子级别IC分析
4. 执行因子重要性分析
5. 生成综合分析报告

使用方法:
    # 完整分析（需要较长时间）
    python run_analysis.py --full
    
    # 快速分析（每个因子库最多分析20个因子）
    python run_analysis.py --quick
    
    # 仅执行回测
    python run_analysis.py --backtest-only
    
    # 仅生成报告（使用已有结果）
    python run_analysis.py --report-only
    
    # 指定因子库
    python run_analysis.py --libraries AA QA
    
    # 指定年份
    python run_analysis.py --years 2022 2023 2024
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加 scripts 目录
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "analysis.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def run_step(step_name: str, func, *args, **kwargs):
    """执行分析步骤并记录时间"""
    print(f"\n{'='*70}")
    print(f"🔄 {step_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"\n✅ {step_name} 完成 (耗时: {elapsed/60:.1f} 分钟)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {step_name} 失败: {e} (耗时: {elapsed/60:.1f} 分钟)")
        logger.error(f"{step_name} 失败", exc_info=True)
        return None


def step_generate_configs():
    """步骤1: 生成分年度回测配置"""
    from scripts.generate_yearly_configs import main as generate_configs
    generate_configs()


def step_run_backtests(libraries=None, years=None):
    """步骤2: 执行分年度回测"""
    from scripts.run_yearly_backtests import YearlyBacktestRunner
    
    config_index = Path(__file__).parent / "configs" / "config_index.yaml"
    
    if not config_index.exists():
        print("⚠️  配置文件不存在，先生成配置...")
        step_generate_configs()
    
    runner = YearlyBacktestRunner(str(config_index))
    return runner.run_all(libraries=libraries, years=years)


def step_factor_ic_analysis(max_factors=None, years=None):
    """步骤3: 因子级别IC分析"""
    from scripts.factor_level_ic_analysis import FactorLevelICAnalyzer
    
    analyzer = FactorLevelICAnalyzer()
    
    # 加载因子库
    analyzer.load_factor_library(
        "AA",
        "/home/tjxy/quantagent/AlphaAgent/factor_library/AA_top80_RankIC_AA_gpt_123_csi300.json"
    )
    analyzer.load_factor_library(
        "QA",
        "/home/tjxy/quantagent/AlphaAgent/factor_library/hj/RANKIC_desc_150_QA_round11_best_gpt_123_csi300.json"
    )
    
    if years is None:
        years = [2021, 2022, 2023, 2024, 2025]
    
    analyzer.analyze_all_factors(years=years, max_factors_per_lib=max_factors)
    analyzer.save_results()
    analyzer.print_analysis_report()
    
    return analyzer


def step_factor_importance_analysis(years=None):
    """步骤4: 因子重要性分析"""
    from scripts.factor_importance_analysis import FactorImportanceAnalyzer
    
    analyzer = FactorImportanceAnalyzer()
    
    # 加载因子库
    analyzer.load_factor_library(
        "AA",
        "/home/tjxy/quantagent/AlphaAgent/factor_library/AA_top80_RankIC_AA_gpt_123_csi300.json"
    )
    analyzer.load_factor_library(
        "QA",
        "/home/tjxy/quantagent/AlphaAgent/factor_library/hj/RANKIC_desc_150_QA_round11_best_gpt_123_csi300.json"
    )
    
    if years is None:
        years = [2021, 2022, 2023, 2024, 2025]
    
    analyzer.analyze_importance_by_year(years=years)
    analyzer.save_results()
    analyzer.print_analysis_report()
    
    return analyzer


def step_generate_report():
    """步骤5: 生成综合报告"""
    from scripts.comprehensive_report import ComprehensiveReportGenerator
    
    generator = ComprehensiveReportGenerator()
    generator.load_all_results()
    report_path = generator.save_report()
    
    # 打印简要总结
    analysis = generator.analyze_ic_decay_cause()
    
    print(f"\n{'='*70}")
    print("📊 分析总结")
    print(f"{'='*70}")
    
    if analysis['key_findings']:
        print("\n关键发现:")
        for i, finding in enumerate(analysis['key_findings'], 1):
            print(f"  {i}. {finding}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='AA vs QA 因子库 IC 衰减分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整分析
  python run_analysis.py --full
  
  # 快速测试（每个因子库最多分析20个因子）
  python run_analysis.py --quick
  
  # 仅执行回测
  python run_analysis.py --backtest-only --years 2022 2023
  
  # 仅生成报告
  python run_analysis.py --report-only
        """
    )
    
    # 运行模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--full', action='store_true',
                           help='执行完整分析流程')
    mode_group.add_argument('--quick', action='store_true',
                           help='快速分析模式（限制因子数量）')
    mode_group.add_argument('--backtest-only', action='store_true',
                           help='仅执行回测')
    mode_group.add_argument('--ic-only', action='store_true',
                           help='仅执行因子级别IC分析')
    mode_group.add_argument('--importance-only', action='store_true',
                           help='仅执行因子重要性分析')
    mode_group.add_argument('--report-only', action='store_true',
                           help='仅生成综合报告')
    
    # 过滤参数
    parser.add_argument('--libraries', '-l', nargs='+', choices=['AA', 'QA'],
                       default=['AA', 'QA'],
                       help='指定因子库')
    parser.add_argument('--years', '-y', nargs='+', type=int,
                       default=[2021, 2022, 2023, 2024, 2025],
                       help='指定年份')
    parser.add_argument('--max-factors', '-m', type=int, default=None,
                       help='每个因子库最多分析的因子数')
    
    args = parser.parse_args()
    
    # 快速模式
    if args.quick:
        args.max_factors = 20
    
    total_start = time.time()
    
    print(f"\n{'#'*70}")
    print("# AA vs QA 因子库 IC 衰减分析")
    print(f"# 因子库: {', '.join(args.libraries)}")
    print(f"# 年份: {', '.join(map(str, args.years))}")
    if args.max_factors:
        print(f"# 每个因子库最多分析: {args.max_factors} 个因子")
    print(f"{'#'*70}")
    
    try:
        if args.report_only:
            # 仅生成报告
            run_step("生成综合分析报告", step_generate_report)
            
        elif args.backtest_only:
            # 仅执行回测
            run_step("生成分年度回测配置", step_generate_configs)
            run_step("执行分年度回测", step_run_backtests, 
                    libraries=args.libraries, years=args.years)
            
        elif args.ic_only:
            # 仅执行因子级别IC分析
            run_step("因子级别IC分析", step_factor_ic_analysis,
                    max_factors=args.max_factors, years=args.years)
            
        elif args.importance_only:
            # 仅执行因子重要性分析
            run_step("因子重要性分析", step_factor_importance_analysis,
                    years=args.years)
            
        else:
            # 完整流程
            run_step("生成分年度回测配置", step_generate_configs)
            
            run_step("执行分年度回测", step_run_backtests,
                    libraries=args.libraries, years=args.years)
            
            run_step("因子级别IC分析", step_factor_ic_analysis,
                    max_factors=args.max_factors, years=args.years)
            
            run_step("因子重要性分析", step_factor_importance_analysis,
                    years=args.years)
            
            run_step("生成综合分析报告", step_generate_report)
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"✅ 分析完成！总耗时: {total_time/60:.1f} 分钟")
        print(f"📁 结果目录: {Path(__file__).parent / 'results'}")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断分析")
        sys.exit(130)
    except Exception as e:
        logger.error("分析过程出错", exc_info=True)
        print(f"\n❌ 分析失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

