#!/usr/bin/env python3
"""
因子级别 IC 分析

功能：
1. 计算每个因子在不同年份的单因子 IC
2. 识别 IC 衰减最严重的因子
3. 对比 AA 和 QA 因子库中因子的表现差异
4. 按因子类型/主题分组分析
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FactorLevelICAnalyzer:
    """因子级别 IC 分析器"""
    
    def __init__(self):
        self.factor_libraries = {}
        self.ic_results = {}
        self.qlib_initialized = False
    
    def _init_qlib(self):
        """初始化 Qlib"""
        if self.qlib_initialized:
            return
        
        import qlib
        qlib.init(provider_uri="/home/tjxy/.qlib/qlib_data/cn_data", region="cn")
        self.qlib_initialized = True
        logger.info("✓ Qlib 初始化完成")
    
    def load_factor_library(self, name: str, path: str):
        """加载因子库"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        factors = data.get('factors', {})
        self.factor_libraries[name] = {
            'path': path,
            'factors': factors,
            'metadata': data.get('metadata', {})
        }
        
        logger.info(f"✓ 加载因子库 {name}: {len(factors)} 个因子")
    
    def calculate_factor_ic_by_year(self, 
                                     factor_name: str,
                                     factor_expression: str,
                                     year: int,
                                     cache_location: Optional[Dict] = None) -> Dict:
        """
        计算单个因子在指定年份的 IC
        
        优先使用缓存的因子值，如果缓存不存在则实时计算
        """
        self._init_qlib()
        
        from qlib.data import D
        from scipy.stats import spearmanr, pearsonr
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        try:
            # 获取股票列表
            stock_list = D.instruments("csi300")
            
            # 尝试从缓存加载因子值
            factor_values = None
            
            if cache_location:
                result_h5_path = cache_location.get('result_h5_path')
                if result_h5_path and Path(result_h5_path).exists():
                    try:
                        factor_df = pd.read_hdf(result_h5_path, key='data')
                        
                        # 过滤指定年份
                        if isinstance(factor_df.index, pd.MultiIndex):
                            dates = factor_df.index.get_level_values('datetime')
                        else:
                            dates = factor_df.index
                        
                        mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
                        factor_values = factor_df[mask]
                        
                        if len(factor_values) > 0:
                            logger.debug(f"  从缓存加载因子值: {len(factor_values)} 行")
                    except Exception as e:
                        logger.warning(f"  缓存加载失败: {e}")
            
            # 如果缓存不存在，尝试使用 Qlib 计算
            if factor_values is None or len(factor_values) == 0:
                try:
                    factor_values = D.features(
                        stock_list,
                        [factor_expression],
                        start_time=start_date,
                        end_time=end_date,
                        freq='day'
                    )
                    factor_values.columns = [factor_name]
                except Exception as e:
                    logger.warning(f"  Qlib 计算失败: {e}")
                    # 使用自定义计算器
                    return self._calculate_factor_ic_custom(
                        factor_name, factor_expression, year
                    )
            
            # 获取收益率标签
            label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
            label_values = D.features(
                stock_list,
                [label_expr],
                start_time=start_date,
                end_time=end_date,
                freq='day'
            )
            label_values.columns = ['label']
            
            # 对齐数据
            if isinstance(factor_values, pd.Series):
                factor_values = factor_values.to_frame(name=factor_name)
            
            common_idx = factor_values.index.intersection(label_values.index)
            factor_values = factor_values.loc[common_idx]
            label_values = label_values.loc[common_idx]
            
            # 计算每日 IC
            daily_ics = []
            daily_rank_ics = []
            
            dates = factor_values.index.get_level_values('datetime').unique()
            
            for date in dates:
                try:
                    f_day = factor_values.xs(date, level='datetime')
                    l_day = label_values.xs(date, level='datetime')
                    
                    # 对齐股票
                    common_stocks = f_day.index.intersection(l_day.index)
                    f_day = f_day.loc[common_stocks]
                    l_day = l_day.loc[common_stocks]
                    
                    # 移除 NaN
                    mask = ~(f_day.iloc[:, 0].isna() | l_day.iloc[:, 0].isna())
                    f_day = f_day[mask]
                    l_day = l_day[mask]
                    
                    if len(f_day) >= 30:  # 至少30只股票
                        # Pearson IC
                        ic, _ = pearsonr(f_day.iloc[:, 0], l_day.iloc[:, 0])
                        daily_ics.append(ic)
                        
                        # Spearman Rank IC
                        rank_ic, _ = spearmanr(f_day.iloc[:, 0], l_day.iloc[:, 0])
                        daily_rank_ics.append(rank_ic)
                        
                except Exception:
                    continue
            
            if len(daily_ics) == 0:
                return {
                    'IC': None,
                    'ICIR': None,
                    'Rank_IC': None,
                    'Rank_ICIR': None,
                    'n_days': 0,
                    'status': 'no_data'
                }
            
            # 计算统计量
            ic_mean = np.nanmean(daily_ics)
            ic_std = np.nanstd(daily_ics)
            rank_ic_mean = np.nanmean(daily_rank_ics)
            rank_ic_std = np.nanstd(daily_rank_ics)
            
            return {
                'IC': float(ic_mean),
                'ICIR': float(ic_mean / ic_std) if ic_std > 0 else 0,
                'Rank_IC': float(rank_ic_mean),
                'Rank_ICIR': float(rank_ic_mean / rank_ic_std) if rank_ic_std > 0 else 0,
                'n_days': len(daily_ics),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"  计算失败: {e}")
            return {
                'IC': None,
                'ICIR': None,
                'Rank_IC': None,
                'Rank_ICIR': None,
                'n_days': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_factor_ic_custom(self, factor_name: str, 
                                    factor_expression: str, year: int) -> Dict:
        """使用自定义计算器计算因子 IC"""
        try:
            from backtest_v2.custom_factor_calculator import get_qlib_stock_data
            from alphaagent.components.coder.factor_coder.expr_parser import parse_expression, parse_symbol
            from alphaagent.components.coder.factor_coder.function_lib import *
            from scipy.stats import spearmanr, pearsonr
            
            # 加载数据
            config = {
                'data': {
                    'provider_uri': '/home/tjxy/.qlib/qlib_data/cn_data',
                    'region': 'cn',
                    'market': 'csi300',
                    'start_time': f'{year-1}-01-01',  # 多取一年用于计算
                    'end_time': f'{year}-12-31'
                }
            }
            
            df = get_qlib_stock_data(config)
            if df is None or df.empty:
                return {'status': 'no_data'}
            
            # 解析并计算因子
            expr = parse_symbol(factor_expression, df.columns)
            expr = parse_expression(expr)
            
            for col in df.columns:
                expr = expr.replace(col[1:], f"df['{col}']")
            
            df[factor_name] = eval(expr)
            
            # 计算收益率
            df['label'] = df.groupby(level='instrument')['$close'].shift(-2) / \
                         df.groupby(level='instrument')['$close'].shift(-1) - 1
            
            # 过滤到目标年份
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            dates = df.index.get_level_values('datetime')
            mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
            df = df[mask]
            
            # 计算每日 IC
            daily_ics = []
            daily_rank_ics = []
            
            for date in df.index.get_level_values('datetime').unique():
                day_data = df.xs(date, level='datetime')[[factor_name, 'label']].dropna()
                
                if len(day_data) >= 30:
                    ic, _ = pearsonr(day_data[factor_name], day_data['label'])
                    rank_ic, _ = spearmanr(day_data[factor_name], day_data['label'])
                    daily_ics.append(ic)
                    daily_rank_ics.append(rank_ic)
            
            if len(daily_ics) == 0:
                return {'status': 'no_data'}
            
            ic_mean = np.nanmean(daily_ics)
            ic_std = np.nanstd(daily_ics)
            rank_ic_mean = np.nanmean(daily_rank_ics)
            rank_ic_std = np.nanstd(daily_rank_ics)
            
            return {
                'IC': float(ic_mean),
                'ICIR': float(ic_mean / ic_std) if ic_std > 0 else 0,
                'Rank_IC': float(rank_ic_mean),
                'Rank_ICIR': float(rank_ic_mean / rank_ic_std) if rank_ic_std > 0 else 0,
                'n_days': len(daily_ics),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"自定义计算失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def analyze_all_factors(self, years: List[int] = [2021, 2022, 2023, 2024, 2025],
                           max_factors_per_lib: Optional[int] = None) -> Dict:
        """
        分析所有因子库中所有因子的年度 IC
        
        Args:
            years: 要分析的年份列表
            max_factors_per_lib: 每个库最多分析的因子数量（用于快速测试）
        """
        results = {}
        
        for lib_name, lib_data in self.factor_libraries.items():
            print(f"\n{'='*70}")
            print(f"📊 分析因子库: {lib_name}")
            print(f"{'='*70}")
            
            factors = lib_data['factors']
            factor_ids = list(factors.keys())
            
            if max_factors_per_lib:
                factor_ids = factor_ids[:max_factors_per_lib]
            
            lib_results = {}
            
            for i, factor_id in enumerate(factor_ids):
                factor = factors[factor_id]
                factor_name = factor.get('factor_name', factor_id)
                factor_expr = factor.get('factor_expression', '')
                cache_loc = factor.get('cache_location')
                
                print(f"\n[{i+1}/{len(factor_ids)}] {factor_name}")
                
                factor_results = {'years': {}}
                
                for year in years:
                    ic_result = self.calculate_factor_ic_by_year(
                        factor_name, factor_expr, year, cache_loc
                    )
                    factor_results['years'][year] = ic_result
                    
                    if ic_result['status'] == 'success':
                        print(f"  {year}: Rank IC = {ic_result['Rank_IC']:.6f}")
                
                # 计算 IC 变化
                if len(factor_results['years']) >= 2:
                    yearly_ics = []
                    for y in sorted(factor_results['years'].keys()):
                        ic = factor_results['years'][y].get('Rank_IC')
                        if ic is not None:
                            yearly_ics.append((y, ic))
                    
                    if len(yearly_ics) >= 2:
                        # 计算2022→2023的变化
                        ic_2022 = dict(yearly_ics).get(2022)
                        ic_2023 = dict(yearly_ics).get(2023)
                        
                        if ic_2022 is not None and ic_2023 is not None and ic_2022 != 0:
                            change_2022_2023 = (ic_2023 - ic_2022) / abs(ic_2022) * 100
                            factor_results['ic_change_2022_2023'] = change_2022_2023
                        
                        # 计算整体趋势
                        first_ic = yearly_ics[0][1]
                        last_ic = yearly_ics[-1][1]
                        if first_ic != 0:
                            factor_results['ic_change_total'] = (last_ic - first_ic) / abs(first_ic) * 100
                
                # 保存因子元信息
                factor_results['metadata'] = {
                    'factor_id': factor_id,
                    'factor_name': factor_name,
                    'expression': factor_expr[:200] + '...' if len(factor_expr) > 200 else factor_expr
                }
                
                lib_results[factor_id] = factor_results
            
            results[lib_name] = lib_results
        
        self.ic_results = results
        return results
    
    def identify_decaying_factors(self, threshold: float = -30.0) -> Dict:
        """
        识别 IC 衰减严重的因子
        
        Args:
            threshold: IC 变化阈值（百分比），默认 -30% 视为严重衰减
        """
        decaying_factors = {}
        
        for lib_name, lib_results in self.ic_results.items():
            lib_decaying = []
            
            for factor_id, factor_data in lib_results.items():
                change = factor_data.get('ic_change_2022_2023')
                
                if change is not None and change < threshold:
                    lib_decaying.append({
                        'factor_id': factor_id,
                        'factor_name': factor_data['metadata']['factor_name'],
                        'ic_change_2022_2023': change,
                        'ic_2022': factor_data['years'].get(2022, {}).get('Rank_IC'),
                        'ic_2023': factor_data['years'].get(2023, {}).get('Rank_IC')
                    })
            
            # 按衰减程度排序
            lib_decaying.sort(key=lambda x: x['ic_change_2022_2023'])
            decaying_factors[lib_name] = lib_decaying
        
        return decaying_factors
    
    def compare_libraries_by_year(self) -> pd.DataFrame:
        """对比两个因子库的年度平均 IC"""
        records = []
        
        for lib_name, lib_results in self.ic_results.items():
            yearly_ics = {}
            
            for factor_id, factor_data in lib_results.items():
                for year, year_data in factor_data.get('years', {}).items():
                    if year not in yearly_ics:
                        yearly_ics[year] = []
                    
                    rank_ic = year_data.get('Rank_IC')
                    if rank_ic is not None:
                        yearly_ics[year].append(rank_ic)
            
            for year, ics in yearly_ics.items():
                if ics:
                    records.append({
                        'Library': lib_name,
                        'Year': year,
                        'Mean_Rank_IC': np.mean(ics),
                        'Median_Rank_IC': np.median(ics),
                        'Std_Rank_IC': np.std(ics),
                        'N_Factors': len(ics)
                    })
        
        return pd.DataFrame(records)
    
    def save_results(self, output_dir: str = None):
        """保存分析结果"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        with open(output_dir / "factor_level_ic_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.ic_results, f, ensure_ascii=False, indent=2)
        
        # 保存衰减因子列表
        decaying = self.identify_decaying_factors()
        with open(output_dir / "decaying_factors.json", 'w', encoding='utf-8') as f:
            json.dump(decaying, f, ensure_ascii=False, indent=2)
        
        # 保存年度对比
        comparison_df = self.compare_libraries_by_year()
        comparison_df.to_csv(output_dir / "library_yearly_comparison.csv", index=False)
        
        print(f"\n✓ 结果已保存到: {output_dir}")
    
    def print_analysis_report(self):
        """打印分析报告"""
        print(f"\n{'='*80}")
        print("📈 因子级别 IC 分析报告")
        print(f"{'='*80}")
        
        # 年度对比
        comparison_df = self.compare_libraries_by_year()
        print("\n【各因子库年度平均 Rank IC】")
        print(comparison_df.to_string(index=False))
        
        # 衰减因子
        decaying = self.identify_decaying_factors()
        
        for lib_name, factors in decaying.items():
            print(f"\n【{lib_name} 因子库 IC 衰减 Top 10】")
            print(f"{'Factor Name':<50} {'2022 IC':<12} {'2023 IC':<12} {'Change %':<10}")
            print("-" * 84)
            
            for f in factors[:10]:
                name = f['factor_name'][:48]
                ic_2022 = f'{f["ic_2022"]:.6f}' if f['ic_2022'] else 'N/A'
                ic_2023 = f'{f["ic_2023"]:.6f}' if f['ic_2023'] else 'N/A'
                change = f'{f["ic_change_2022_2023"]:.1f}%'
                print(f"{name:<50} {ic_2022:<12} {ic_2023:<12} {change:<10}")
        
        # 关键发现
        print(f"\n{'='*80}")
        print("🔍 关键发现")
        print(f"{'='*80}")
        
        if 'AA' in decaying and 'QA' in decaying:
            aa_severe = len([f for f in decaying['AA'] if f['ic_change_2022_2023'] < -50])
            qa_severe = len([f for f in decaying['QA'] if f['ic_change_2022_2023'] < -50])
            
            print(f"\n  AA 因子库中 IC 下降超过50%的因子数: {aa_severe}")
            print(f"  QA 因子库中 IC 下降超过50%的因子数: {qa_severe}")
            
            if aa_severe > qa_severe:
                print(f"\n  ⚠️  AA 因子库在2023年的IC衰减更为严重")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='因子级别IC分析')
    parser.add_argument('--max-factors', '-m', type=int, default=None,
                       help='每个库最多分析的因子数（用于快速测试）')
    parser.add_argument('--years', '-y', nargs='+', type=int,
                       default=[2021, 2022, 2023, 2024, 2025],
                       help='要分析的年份')
    
    args = parser.parse_args()
    
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
    
    # 分析
    analyzer.analyze_all_factors(years=args.years, max_factors_per_lib=args.max_factors)
    
    # 保存结果
    analyzer.save_results()
    
    # 打印报告
    analyzer.print_analysis_report()


if __name__ == "__main__":
    main()

