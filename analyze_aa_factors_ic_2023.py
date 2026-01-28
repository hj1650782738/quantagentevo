#!/usr/bin/env python3
"""
计算 AA 因子库在 2023 年的年度 IC 指标

输出格式与 csi300_2023_ic_metrics.csv 一致:
- factor_name: 因子名称
- annual_ic: 年度平均 IC
- annual_rank_ic: 年度平均 Rank IC  
- ic_ir: IC IR (IC均值/IC标准差)
- rank_ic_ir: Rank IC IR (Rank IC均值/Rank IC标准差)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def init_qlib():
    """初始化 Qlib"""
    import qlib
    qlib.init(provider_uri="/home/tjxy/.qlib/qlib_data/cn_data", region="cn")
    print("✓ Qlib 初始化完成")


def load_factor_library(path: str) -> Dict:
    """加载因子库"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    factors = data.get('factors', {})
    print(f"✓ 加载因子库: {len(factors)} 个因子")
    return factors


def calculate_factor_ic_2023(factor_name: str, 
                              factor_expression: str,
                              cache_location: Optional[Dict] = None) -> Optional[Dict]:
    """
    计算单个因子在 2023 年的 IC 指标
    
    Returns:
        Dict with: annual_ic, annual_rank_ic, ic_ir, rank_ic_ir
    """
    from qlib.data import D
    
    year = 2023
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    try:
        stock_list = D.instruments("csi300")
        
        # 尝试从缓存加载因子值
        factor_values = None
        
        if cache_location:
            result_h5_path = cache_location.get('result_h5_path')
            if result_h5_path and Path(result_h5_path).exists():
                try:
                    factor_df = pd.read_hdf(result_h5_path, key='data')
                    
                    # 过滤到 2023 年
                    if isinstance(factor_df.index, pd.MultiIndex):
                        dates = factor_df.index.get_level_values('datetime')
                    else:
                        dates = factor_df.index
                    
                    mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
                    factor_values = factor_df[mask]
                    
                    if isinstance(factor_values, pd.Series):
                        factor_values = factor_values.to_frame(name=factor_name)
                    else:
                        factor_values.columns = [factor_name]
                    
                except Exception as e:
                    print(f"    缓存加载失败: {e}")
        
        # 如果缓存不存在，尝试用 Qlib 计算
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
                print(f"    Qlib 计算失败: {e}")
                return None
        
        if factor_values is None or len(factor_values) == 0:
            return None
        
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
                
                if len(f_day) >= 30:
                    # Pearson IC
                    ic, _ = pearsonr(f_day.iloc[:, 0], l_day.iloc[:, 0])
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                    
                    # Spearman Rank IC
                    rank_ic, _ = spearmanr(f_day.iloc[:, 0], l_day.iloc[:, 0])
                    if not np.isnan(rank_ic):
                        daily_rank_ics.append(rank_ic)
                    
            except Exception:
                continue
        
        if len(daily_ics) == 0:
            return None
        
        # 计算年度统计量
        ic_mean = np.mean(daily_ics)
        ic_std = np.std(daily_ics)
        rank_ic_mean = np.mean(daily_rank_ics)
        rank_ic_std = np.std(daily_rank_ics)
        
        return {
            'annual_ic': ic_mean,
            'annual_rank_ic': rank_ic_mean,
            'ic_ir': ic_mean / ic_std if ic_std > 0 else 0,
            'rank_ic_ir': rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0
        }
        
    except Exception as e:
        print(f"    计算错误: {e}")
        return None


def main():
    # 因子库路径
    factor_lib_path = "/home/tjxy/quantagent/AlphaAgent/factor_library/AA_top80_RankIC_AA_gpt_123_csi300.json"
    
    # 检查文件是否存在
    if not Path(factor_lib_path).exists():
        # 尝试 hj 目录
        alt_path = "/home/tjxy/quantagent/AlphaAgent/factor_library/hj/AA_top80_RankIC_AA_gpt_123_csi300.json"
        if Path(alt_path).exists():
            factor_lib_path = alt_path
        else:
            print(f"错误: 因子库文件不存在: {factor_lib_path}")
            return
    
    print(f"使用因子库: {factor_lib_path}")
    
    # 初始化 Qlib
    init_qlib()
    
    # 加载因子库
    factors = load_factor_library(factor_lib_path)
    
    # 计算每个因子的 IC
    results = []
    
    total = len(factors)
    for i, (factor_id, factor_info) in enumerate(factors.items()):
        factor_name = factor_info.get('factor_name', factor_id)
        factor_expr = factor_info.get('factor_expression', '')
        cache_loc = factor_info.get('cache_location')
        
        print(f"[{i+1}/{total}] {factor_name}...", end=" ")
        
        ic_result = calculate_factor_ic_2023(factor_name, factor_expr, cache_loc)
        
        if ic_result:
            results.append({
                'factor_name': factor_name,
                'annual_ic': ic_result['annual_ic'],
                'annual_rank_ic': ic_result['annual_rank_ic'],
                'ic_ir': ic_result['ic_ir'],
                'rank_ic_ir': ic_result['rank_ic_ir']
            })
            print(f"Rank IC = {ic_result['annual_rank_ic']:.6f}")
        else:
            print("跳过")
    
    # 保存结果
    if results:
        df = pd.DataFrame(results)
        
        # 按 Rank IC 降序排序
        df = df.sort_values('annual_rank_ic', ascending=False)
        
        output_path = Path(factor_lib_path).parent / "AA_csi300_2023_ic_metrics.csv"
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"✓ 结果已保存: {output_path}")
        print(f"✓ 共计算 {len(results)} 个因子")
        print(f"\n📊 统计摘要:")
        print(f"  平均 Rank IC: {df['annual_rank_ic'].mean():.6f}")
        print(f"  最大 Rank IC: {df['annual_rank_ic'].max():.6f}")
        print(f"  最小 Rank IC: {df['annual_rank_ic'].min():.6f}")
        print(f"  Rank IC > 0 的因子数: {(df['annual_rank_ic'] > 0).sum()}")
        print(f"  Rank IC < 0 的因子数: {(df['annual_rank_ic'] < 0).sum()}")
        
        print(f"\n📈 Top 10 因子 (by Rank IC):")
        for i, row in df.head(10).iterrows():
            print(f"  {row['factor_name'][:50]:<50} {row['annual_rank_ic']:.6f}")
        
        print(f"\n📉 Bottom 10 因子 (by Rank IC):")
        for i, row in df.tail(10).iterrows():
            print(f"  {row['factor_name'][:50]:<50} {row['annual_rank_ic']:.6f}")
    else:
        print("没有成功计算的因子")


if __name__ == "__main__":
    main()

