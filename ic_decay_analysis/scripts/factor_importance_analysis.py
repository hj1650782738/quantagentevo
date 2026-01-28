#!/usr/bin/env python3
"""
因子重要性分析

功能：
1. 训练 LightGBM 模型并提取因子重要性
2. 分析不同年份因子权重的变化
3. 识别主导因子及其在2023年的表现
4. 对比 AA 和 QA 因子库的重要性分布特征
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

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


class FactorImportanceAnalyzer:
    """因子重要性分析器"""
    
    def __init__(self):
        self.factor_libraries = {}
        self.importance_results = {}
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
    
    def _prepare_factor_data(self, lib_name: str, year: int) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        准备因子数据用于模型训练
        
        返回: (特征DataFrame, 标签Series)
        """
        self._init_qlib()
        
        from qlib.data import D
        from backtest_v2.custom_factor_calculator import get_qlib_stock_data
        
        lib_data = self.factor_libraries[lib_name]
        factors = lib_data['factors']
        
        # 训练时间范围：使用测试年份前两年作为训练集
        train_start = f"{year-3}-01-01"
        train_end = f"{year-1}-12-31"
        
        print(f"  训练数据范围: {train_start} ~ {train_end}")
        
        try:
            # 获取股票列表
            stock_list = D.instruments("csi300")
            
            # 收集所有因子数据
            all_factor_dfs = []
            factor_names = []
            
            for factor_id, factor_info in factors.items():
                factor_name = factor_info.get('factor_name', factor_id)
                cache_loc = factor_info.get('cache_location')
                
                factor_df = None
                
                # 尝试从缓存加载
                if cache_loc:
                    result_h5_path = cache_loc.get('result_h5_path')
                    if result_h5_path and Path(result_h5_path).exists():
                        try:
                            factor_df = pd.read_hdf(result_h5_path, key='data')
                            
                            # 过滤时间范围
                            if isinstance(factor_df.index, pd.MultiIndex):
                                dates = factor_df.index.get_level_values('datetime')
                            else:
                                dates = factor_df.index
                            
                            mask = (dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))
                            factor_df = factor_df[mask]
                            
                            if isinstance(factor_df, pd.Series):
                                factor_df = factor_df.to_frame(name=factor_name)
                            else:
                                factor_df.columns = [factor_name]
                            
                        except Exception as e:
                            logger.debug(f"  缓存加载失败 {factor_name}: {e}")
                
                if factor_df is not None and len(factor_df) > 0:
                    all_factor_dfs.append(factor_df)
                    factor_names.append(factor_name)
            
            if len(all_factor_dfs) == 0:
                logger.warning(f"  没有可用的因子数据")
                return None
            
            # 合并所有因子
            features_df = pd.concat(all_factor_dfs, axis=1)
            features_df = features_df.loc[:, ~features_df.columns.duplicated()]
            
            print(f"  加载 {len(features_df.columns)} 个因子, {len(features_df)} 行数据")
            
            # 获取标签
            label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
            label_df = D.features(
                stock_list,
                [label_expr],
                start_time=train_start,
                end_time=train_end,
                freq='day'
            )
            label_df.columns = ['label']
            
            # 对齐数据
            common_idx = features_df.index.intersection(label_df.index)
            features_df = features_df.loc[common_idx]
            label_series = label_df.loc[common_idx, 'label']
            
            # 数据预处理
            features_df = features_df.fillna(0)
            features_df = features_df.replace([np.inf, -np.inf], 0)
            
            # 移除标签为 NaN 的行
            valid_mask = ~label_series.isna()
            features_df = features_df[valid_mask]
            label_series = label_series[valid_mask]
            
            print(f"  预处理后: {len(features_df)} 行数据")
            
            return features_df, label_series
            
        except Exception as e:
            logger.error(f"  准备数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_and_get_importance(self, lib_name: str, year: int) -> Optional[Dict]:
        """
        训练 LightGBM 模型并获取因子重要性
        """
        import lightgbm as lgb
        
        print(f"\n训练 {lib_name} - {year} 年模型...")
        
        data = self._prepare_factor_data(lib_name, year)
        if data is None:
            return None
        
        features_df, label_series = data
        
        # 模型参数
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.1,
            'max_depth': 8,
            'num_leaves': 210,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'lambda_l1': 200,
            'lambda_l2': 500,
            'min_child_samples': 100,
            'verbose': -1,
            'seed': 42
        }
        
        # 创建数据集
        train_data = lgb.Dataset(features_df, label=label_series)
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(period=0)]  # 禁用日志
        )
        
        # 获取因子重要性
        importance_gain = model.feature_importance(importance_type='gain')
        importance_split = model.feature_importance(importance_type='split')
        
        feature_names = features_df.columns.tolist()
        
        # 整理结果
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_gain': importance_gain,
            'importance_split': importance_split
        })
        
        # 归一化重要性
        importance_df['importance_gain_norm'] = importance_df['importance_gain'] / importance_df['importance_gain'].sum()
        importance_df['importance_split_norm'] = importance_df['importance_split'] / importance_df['importance_split'].sum()
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance_gain', ascending=False)
        
        return {
            'importance_df': importance_df,
            'n_features': len(feature_names),
            'total_gain': importance_df['importance_gain'].sum(),
            'top_10_features': importance_df.head(10)['feature'].tolist(),
            'top_10_gain_pct': importance_df.head(10)['importance_gain_norm'].sum()
        }
    
    def analyze_importance_by_year(self, years: List[int] = [2021, 2022, 2023, 2024, 2025]):
        """分析各年份的因子重要性"""
        results = {}
        
        for lib_name in self.factor_libraries.keys():
            print(f"\n{'='*70}")
            print(f"📊 分析因子库: {lib_name}")
            print(f"{'='*70}")
            
            lib_results = {}
            
            for year in years:
                importance = self.train_and_get_importance(lib_name, year)
                if importance:
                    lib_results[year] = importance
                    
                    print(f"\n  {year}年 Top 10 因子 (占总重要性 {importance['top_10_gain_pct']*100:.1f}%):")
                    for i, name in enumerate(importance['top_10_features'][:5]):
                        print(f"    {i+1}. {name}")
            
            results[lib_name] = lib_results
        
        self.importance_results = results
        return results
    
    def analyze_dominant_factors(self) -> Dict:
        """
        分析主导因子
        
        识别在多个年份都排名靠前的因子
        """
        dominant_factors = {}
        
        for lib_name, lib_results in self.importance_results.items():
            # 统计每个因子在各年份的排名
            factor_ranks = {}
            
            for year, year_data in lib_results.items():
                imp_df = year_data['importance_df']
                
                for rank, (_, row) in enumerate(imp_df.iterrows()):
                    feature = row['feature']
                    if feature not in factor_ranks:
                        factor_ranks[feature] = {}
                    factor_ranks[feature][year] = {
                        'rank': rank + 1,
                        'importance_gain': row['importance_gain'],
                        'importance_pct': row['importance_gain_norm']
                    }
            
            # 计算平均排名和稳定性
            factor_stats = []
            for feature, yearly_data in factor_ranks.items():
                ranks = [v['rank'] for v in yearly_data.values()]
                gains = [v['importance_pct'] for v in yearly_data.values()]
                
                factor_stats.append({
                    'feature': feature,
                    'avg_rank': np.mean(ranks),
                    'min_rank': min(ranks),
                    'max_rank': max(ranks),
                    'rank_std': np.std(ranks),
                    'avg_importance_pct': np.mean(gains),
                    'years_in_top_20': sum(1 for r in ranks if r <= 20),
                    'yearly_data': yearly_data
                })
            
            # 按平均排名排序
            factor_stats.sort(key=lambda x: x['avg_rank'])
            
            # 取出稳定的高重要性因子
            dominant = [f for f in factor_stats if f['years_in_top_20'] >= 3]
            
            dominant_factors[lib_name] = {
                'all_factors': factor_stats,
                'dominant_factors': dominant[:20],
                'n_dominant': len(dominant)
            }
        
        return dominant_factors
    
    def compare_importance_shift(self) -> Dict:
        """
        对比因子重要性在2022→2023的变化
        """
        shifts = {}
        
        for lib_name, lib_results in self.importance_results.items():
            if 2022 not in lib_results or 2023 not in lib_results:
                continue
            
            imp_2022 = lib_results[2022]['importance_df'].set_index('feature')
            imp_2023 = lib_results[2023]['importance_df'].set_index('feature')
            
            # 合并比较
            common_features = set(imp_2022.index) & set(imp_2023.index)
            
            comparison = []
            for feature in common_features:
                gain_2022 = imp_2022.loc[feature, 'importance_gain_norm']
                gain_2023 = imp_2023.loc[feature, 'importance_gain_norm']
                
                rank_2022 = imp_2022.index.get_loc(feature) + 1 if feature in imp_2022.index else None
                rank_2023 = imp_2023.index.get_loc(feature) + 1 if feature in imp_2023.index else None
                
                change = (gain_2023 - gain_2022) / gain_2022 * 100 if gain_2022 > 0 else 0
                
                comparison.append({
                    'feature': feature,
                    'gain_2022': gain_2022,
                    'gain_2023': gain_2023,
                    'gain_change_pct': change,
                    'rank_2022': rank_2022,
                    'rank_2023': rank_2023
                })
            
            # 按重要性变化排序（识别上升和下降最多的因子）
            comparison.sort(key=lambda x: x['gain_change_pct'])
            
            shifts[lib_name] = {
                'declining': comparison[:10],  # 重要性下降最多
                'rising': comparison[-10:][::-1],  # 重要性上升最多
                'all': comparison
            }
        
        return shifts
    
    def save_results(self, output_dir: str = None):
        """保存分析结果"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存年度重要性
        yearly_importance = {}
        for lib_name, lib_results in self.importance_results.items():
            yearly_importance[lib_name] = {}
            for year, year_data in lib_results.items():
                yearly_importance[lib_name][year] = {
                    'top_20': year_data['importance_df'].head(20).to_dict('records'),
                    'n_features': year_data['n_features'],
                    'top_10_gain_pct': year_data['top_10_gain_pct']
                }
        
        with open(output_dir / "factor_importance_by_year.json", 'w', encoding='utf-8') as f:
            json.dump(yearly_importance, f, ensure_ascii=False, indent=2)
        
        # 保存主导因子
        dominant = self.analyze_dominant_factors()
        dominant_simplified = {}
        for lib_name, data in dominant.items():
            dominant_simplified[lib_name] = {
                'dominant_factors': [
                    {k: v for k, v in f.items() if k != 'yearly_data'}
                    for f in data['dominant_factors']
                ],
                'n_dominant': data['n_dominant']
            }
        
        with open(output_dir / "dominant_factors.json", 'w', encoding='utf-8') as f:
            json.dump(dominant_simplified, f, ensure_ascii=False, indent=2)
        
        # 保存重要性变化
        shifts = self.compare_importance_shift()
        with open(output_dir / "importance_shift_2022_2023.json", 'w', encoding='utf-8') as f:
            json.dump(shifts, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 结果已保存到: {output_dir}")
    
    def print_analysis_report(self):
        """打印分析报告"""
        print(f"\n{'='*80}")
        print("📈 因子重要性分析报告")
        print(f"{'='*80}")
        
        # 主导因子
        dominant = self.analyze_dominant_factors()
        
        for lib_name, data in dominant.items():
            print(f"\n【{lib_name} 因子库 - 稳定高重要性因子 Top 10】")
            print(f"{'Factor Name':<50} {'Avg Rank':<10} {'Avg Imp%':<10} {'Years Top20':<12}")
            print("-" * 82)
            
            for f in data['dominant_factors'][:10]:
                name = f['feature'][:48]
                avg_rank = f'{f["avg_rank"]:.1f}'
                avg_imp = f'{f["avg_importance_pct"]*100:.2f}%'
                years_top = str(f['years_in_top_20'])
                print(f"{name:<50} {avg_rank:<10} {avg_imp:<10} {years_top:<12}")
        
        # 重要性变化
        shifts = self.compare_importance_shift()
        
        for lib_name, data in shifts.items():
            print(f"\n【{lib_name} 因子库 - 2022→2023 重要性下降最多的因子】")
            print(f"{'Factor Name':<50} {'2022 Imp%':<12} {'2023 Imp%':<12} {'Change':<10}")
            print("-" * 84)
            
            for f in data['declining'][:5]:
                name = f['feature'][:48]
                imp_2022 = f'{f["gain_2022"]*100:.2f}%'
                imp_2023 = f'{f["gain_2023"]*100:.2f}%'
                change = f'{f["gain_change_pct"]:.1f}%'
                print(f"{name:<50} {imp_2022:<12} {imp_2023:<12} {change:<10}")
            
            print(f"\n【{lib_name} 因子库 - 2022→2023 重要性上升最多的因子】")
            for f in data['rising'][:5]:
                name = f['feature'][:48]
                imp_2022 = f'{f["gain_2022"]*100:.2f}%'
                imp_2023 = f'{f["gain_2023"]*100:.2f}%'
                change = f'+{f["gain_change_pct"]:.1f}%'
                print(f"{name:<50} {imp_2022:<12} {imp_2023:<12} {change:<10}")
        
        # 关键发现
        print(f"\n{'='*80}")
        print("🔍 关键发现")
        print(f"{'='*80}")
        
        if 'AA' in dominant and 'QA' in dominant:
            aa_dominant = dominant['AA']['n_dominant']
            qa_dominant = dominant['QA']['n_dominant']
            
            print(f"\n  AA 因子库稳定高重要性因子数: {aa_dominant}")
            print(f"  QA 因子库稳定高重要性因子数: {qa_dominant}")
            
            if aa_dominant < qa_dominant:
                print(f"\n  ⚠️  AA 因子库的因子重要性更不稳定，可能导致在市场变化时表现下降")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='因子重要性分析')
    parser.add_argument('--years', '-y', nargs='+', type=int,
                       default=[2021, 2022, 2023, 2024, 2025],
                       help='要分析的年份')
    
    args = parser.parse_args()
    
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
    
    # 分析
    analyzer.analyze_importance_by_year(years=args.years)
    
    # 保存结果
    analyzer.save_results()
    
    # 打印报告
    analyzer.print_analysis_report()


if __name__ == "__main__":
    main()

