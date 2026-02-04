#!/usr/bin/env python3
"""
年度因子 IC 计算工具

功能：
1. 从因子库JSON文件读取因子列表
2. 优先级加载因子值：
   - cache_location.result_h5_path（因子库指定的缓存）
   - MD5 缓存（factor_cache 目录）
   - 实时计算（使用 AlphaAgent 表达式解析器）
3. 计算指定年度的 IC、Rank IC、IC IR、Rank IC IR
4. 输出 CSV 文件

使用方式:
    python backtest_v2/yearly_ic_calculator.py \
        --factor-json /path/to/factors.json \
        --year 2023 \
        --market csi300 \
        --output /path/to/output.csv
"""

import argparse
import hashlib
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认缓存目录
DEFAULT_CACHE_DIR = Path("/mnt/DATA/quantagent/AlphaAgent/factor_cache")


class YearlyICCalculator:
    """年度 IC 计算器"""
    
    def __init__(self, 
                 market: str = "csi300", 
                 provider_uri: str = "/home/tjxy/.qlib/qlib_data/cn_data",
                 cache_dir: Optional[Path] = None):
        self.market = market
        self.provider_uri = provider_uri
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._qlib_initialized = False
        self._label_cache = {}  # 缓存标签数据
        self._data_df_cache = {}  # 缓存股票数据
    
    def _init_qlib(self):
        """初始化 Qlib"""
        if self._qlib_initialized:
            return
        
        import qlib
        qlib.init(provider_uri=self.provider_uri, region="cn")
        self._qlib_initialized = True
        logger.info(f"✓ Qlib 初始化完成: {self.provider_uri}")
    
    def load_factor_library(self, json_path: str) -> Dict:
        """加载因子库"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        factors = data.get('factors', {})
        logger.info(f"✓ 加载因子库: {len(factors)} 个因子")
        return factors
    
    def _get_cache_key(self, expr: str) -> str:
        """生成 MD5 缓存键"""
        return hashlib.md5(expr.encode()).hexdigest()
    
    def load_factor_from_cache_location(self, cache_location: Dict, year: int) -> Optional[pd.Series]:
        """
        从 cache_location 字段指定的路径加载因子值
        
        Args:
            cache_location: 缓存位置信息，包含 result_h5_path
            year: 目标年份
            
        Returns:
            过滤后的因子值 Series
        """
        if not cache_location:
            return None
        
        result_h5_path = cache_location.get('result_h5_path', '')
        if not result_h5_path or not Path(result_h5_path).exists():
            return None
        
        try:
            # 读取 HDF5 文件
            factor_df = pd.read_hdf(result_h5_path, key='data')
            return self._filter_factor_by_year(factor_df, year, result_h5_path)
        except Exception as e:
            logger.debug(f"从 cache_location 加载失败 [{result_h5_path}]: {e}")
            return None
    
    def load_factor_from_md5_cache(self, factor_expr: str, year: int) -> Optional[pd.Series]:
        """
        从 MD5 缓存加载因子值
        
        Args:
            factor_expr: 因子表达式
            year: 目标年份
            
        Returns:
            过滤后的因子值 Series
        """
        cache_key = self._get_cache_key(factor_expr)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            result = pd.read_pickle(cache_file)
            return self._filter_factor_by_year(result, year, str(cache_file))
        except Exception as e:
            logger.debug(f"从 MD5 缓存加载失败 [{cache_file}]: {e}")
            return None
    
    def _filter_factor_by_year(self, factor_data: Any, year: int, source: str) -> Optional[pd.Series]:
        """
        将因子数据过滤到指定年份
        
        Args:
            factor_data: 原始因子数据 (DataFrame 或 Series)
            year: 目标年份
            source: 数据来源（用于日志）
            
        Returns:
            过滤后的因子值 Series
        """
        try:
            # 处理 DataFrame 格式
            if isinstance(factor_data, pd.DataFrame):
                if len(factor_data.columns) == 1:
                    factor_series = factor_data.iloc[:, 0]
                elif 'factor' in factor_data.columns:
                    factor_series = factor_data['factor']
                else:
                    factor_series = factor_data.iloc[:, 0]
            else:
                factor_series = factor_data
            
            # 过滤到指定年份
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            if isinstance(factor_series.index, pd.MultiIndex):
                # MultiIndex: (instrument, datetime) 或 (datetime, instrument)
                idx_names = list(factor_series.index.names)
                
                # 找到 datetime 所在的 level
                datetime_level = None
                for i, name in enumerate(idx_names):
                    if name == 'datetime':
                        datetime_level = i
                        break
                    level_values = factor_series.index.get_level_values(i)
                    if pd.api.types.is_datetime64_any_dtype(level_values):
                        datetime_level = i
                        break
                
                if datetime_level is None:
                    datetime_level = 0
                
                dates = factor_series.index.get_level_values(datetime_level)
                mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
                factor_series = factor_series[mask]
            
            if len(factor_series) == 0:
                return None
            
            return factor_series
            
        except Exception as e:
            logger.debug(f"过滤因子数据失败 [{source}]: {e}")
            return None
    
    def calculate_factor_realtime(self, factor_name: str, factor_expr: str, year: int) -> Optional[pd.Series]:
        """
        实时计算因子值
        
        Args:
            factor_name: 因子名称
            factor_expr: 因子表达式
            year: 目标年份
            
        Returns:
            计算得到的因子值 Series
        """
        try:
            # 获取股票数据
            data_df = self._get_stock_data(year)
            if data_df is None or data_df.empty:
                return None
            
            # 导入计算器
            import io
            import sys as _sys
            from joblib import parallel_backend
            from alphaagent.components.coder.factor_coder.expr_parser import (
                parse_expression, parse_symbol
            )
            import alphaagent.components.coder.factor_coder.function_lib as func_lib
            
            df = data_df.copy()
            
            # 添加 $return 列 (如果不存在)
            if '$return' not in df.columns:
                df['$return'] = df.groupby('instrument')['$close'].transform(
                    lambda x: x / x.shift(1) - 1
                )
            
            # 解析表达式
            expr = parse_symbol(factor_expr, df.columns)
            
            # 静默解析
            old_stdout = _sys.stdout
            _sys.stdout = io.StringIO()
            try:
                expr = parse_expression(expr)
            finally:
                _sys.stdout = old_stdout
            
            # 替换变量
            for col in df.columns:
                if col.startswith('$'):
                    expr = expr.replace(col[1:], f"df['{col}']")
            
            # 构建执行环境
            exec_globals = {'df': df, 'np': np, 'pd': pd}
            for name in dir(func_lib):
                if not name.startswith('_'):
                    obj = getattr(func_lib, name)
                    if callable(obj):
                        exec_globals[name] = obj
            
            # 计算
            with parallel_backend('threading', n_jobs=1):
                result = eval(expr, exec_globals)
            
            if isinstance(result, pd.DataFrame):
                result = result.iloc[:, 0]
            
            if isinstance(result, pd.Series):
                result.name = factor_name
                return result.astype(np.float64)
            else:
                return pd.Series(result, index=df.index, name=factor_name).astype(np.float64)
            
        except Exception as e:
            logger.debug(f"实时计算因子失败 [{factor_name}]: {str(e)[:100]}")
            return None
    
    def _get_stock_data(self, year: int) -> Optional[pd.DataFrame]:
        """获取指定年份的股票数据"""
        if year in self._data_df_cache:
            return self._data_df_cache[year]
        
        self._init_qlib()
        from qlib.data import D
        
        # 扩展数据范围以支持需要历史数据的因子
        start_date = f"{year-1}-01-01"  # 多加载1年历史数据
        end_date = f"{year}-12-31"
        
        stock_list = D.instruments(self.market)
        
        fields = ['$open', '$high', '$low', '$close', '$volume', '$vwap']
        df = D.features(
            stock_list,
            fields,
            start_time=start_date,
            end_time=end_date,
            freq='day'
        )
        df.columns = fields
        
        self._data_df_cache[year] = df
        logger.info(f"  加载{year}年股票数据: {len(df)} 行")
        
        return df
    
    def get_label_data(self, year: int) -> pd.DataFrame:
        """
        获取指定年份的标签数据（收益率）
        
        Returns:
            DataFrame with MultiIndex (instrument, datetime) and column 'label'
        """
        if year in self._label_cache:
            return self._label_cache[year]
        
        self._init_qlib()
        from qlib.data import D
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        stock_list = D.instruments(self.market)
        
        # 标签: T+2收益率
        label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
        label_df = D.features(
            stock_list,
            [label_expr],
            start_time=start_date,
            end_time=end_date,
            freq='day'
        )
        label_df.columns = ['label']
        
        self._label_cache[year] = label_df
        logger.info(f"✓ 加载{year}年标签数据: {len(label_df)} 行")
        
        return label_df
    
    def load_factor_with_fallback(self, factor_info: Dict, year: int) -> Optional[pd.Series]:
        """
        加载因子值，按优先级尝试多种来源
        
        优先级:
        1. cache_location.result_h5_path
        2. MD5 缓存 (factor_cache 目录)
        3. 实时计算
        
        Returns:
            Tuple[factor_series, source_type]
        """
        factor_name = factor_info.get('factor_name', 'unknown')
        factor_expr = factor_info.get('factor_expression', '')
        cache_location = factor_info.get('cache_location')
        
        # 1. 尝试从 cache_location 加载
        if cache_location:
            result = self.load_factor_from_cache_location(cache_location, year)
            if result is not None and len(result) > 0:
                return result, 'cache_location'
        
        # 2. 尝试从 MD5 缓存加载
        if factor_expr:
            result = self.load_factor_from_md5_cache(factor_expr, year)
            if result is not None and len(result) > 0:
                return result, 'md5_cache'
        
        # 3. 实时计算
        if factor_expr:
            result = self.calculate_factor_realtime(factor_name, factor_expr, year)
            if result is not None and len(result) > 0:
                # 过滤到目标年份
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                
                if isinstance(result.index, pd.MultiIndex):
                    idx_names = list(result.index.names)
                    datetime_level = None
                    for i, name in enumerate(idx_names):
                        if name == 'datetime':
                            datetime_level = i
                            break
                        level_values = result.index.get_level_values(i)
                        if pd.api.types.is_datetime64_any_dtype(level_values):
                            datetime_level = i
                            break
                    
                    if datetime_level is not None:
                        dates = result.index.get_level_values(datetime_level)
                        mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
                        result = result[mask]
                
                if len(result) > 0:
                    return result, 'calculated'
        
        return None, None
    
    def calculate_factor_ic(self, factor_series: pd.Series, label_df: pd.DataFrame) -> Optional[Dict]:
        """
        计算因子的 IC 指标
        
        Returns:
            Dict with: annual_ic, annual_rank_ic, ic_ir, rank_ic_ir
        """
        try:
            # 对齐因子和标签的索引
            if isinstance(factor_series.index, pd.MultiIndex):
                factor_idx_names = list(factor_series.index.names)
                label_idx_names = list(label_df.index.names)
                
                # 如果索引顺序不同，调整因子的索引顺序
                if factor_idx_names != label_idx_names:
                    if set(factor_idx_names) == set(label_idx_names):
                        factor_series = factor_series.swaplevel()
                        factor_series = factor_series.sort_index()
            
            # 找到共同索引
            common_idx = factor_series.index.intersection(label_df.index)
            
            if len(common_idx) < 100:
                logger.debug(f"共同索引过少: {len(common_idx)}")
                return None
            
            factor_aligned = factor_series.loc[common_idx]
            label_aligned = label_df.loc[common_idx, 'label']
            
            # 获取所有交易日
            if isinstance(factor_aligned.index, pd.MultiIndex):
                idx_names = factor_aligned.index.names
                datetime_level = None
                for name in idx_names:
                    if name == 'datetime':
                        datetime_level = name
                        break
                    level_values = factor_aligned.index.get_level_values(name)
                    if pd.api.types.is_datetime64_any_dtype(level_values):
                        datetime_level = name
                        break
                
                if datetime_level is None:
                    datetime_level = idx_names[0]
                
                dates = factor_aligned.index.get_level_values(datetime_level).unique()
            else:
                dates = factor_aligned.index.unique()
            
            # 计算每日 IC
            daily_ics = []
            daily_rank_ics = []
            
            for date in dates:
                try:
                    # 获取当日数据
                    if isinstance(factor_aligned.index, pd.MultiIndex):
                        f_day = factor_aligned.xs(date, level=datetime_level)
                        l_day = label_aligned.xs(date, level=datetime_level)
                    else:
                        f_day = factor_aligned.loc[date]
                        l_day = label_aligned.loc[date]
                    
                    # 对齐股票
                    if isinstance(f_day, pd.Series) and isinstance(l_day, pd.Series):
                        common_stocks = f_day.index.intersection(l_day.index)
                        f_day = f_day.loc[common_stocks]
                        l_day = l_day.loc[common_stocks]
                    
                    # 移除 NaN
                    mask = ~(pd.isna(f_day) | pd.isna(l_day))
                    f_day = f_day[mask]
                    l_day = l_day[mask]
                    
                    if len(f_day) >= 30:
                        # Pearson IC
                        ic, _ = pearsonr(f_day.values, l_day.values)
                        if not np.isnan(ic):
                            daily_ics.append(ic)
                        
                        # Spearman Rank IC
                        rank_ic, _ = spearmanr(f_day.values, l_day.values)
                        if not np.isnan(rank_ic):
                            daily_rank_ics.append(rank_ic)
                
                except Exception:
                    continue
            
            if len(daily_ics) < 20:
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
                'rank_ic_ir': rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0,
                'n_days': len(daily_ics)
            }
            
        except Exception as e:
            logger.debug(f"IC计算错误: {e}")
            return None
    
    def calculate_all_factors(self, factors: Dict, year: int) -> pd.DataFrame:
        """
        计算所有因子的年度 IC
        
        Args:
            factors: 因子字典
            year: 目标年份
            
        Returns:
            DataFrame with columns: factor_name, annual_ic, annual_rank_ic, ic_ir, rank_ic_ir
        """
        # 获取标签数据
        label_df = self.get_label_data(year)
        
        results = []
        total = len(factors)
        success_count = 0
        cache_location_hit = 0
        md5_cache_hit = 0
        calculated_count = 0
        failed_count = 0
        
        for i, (factor_id, factor_info) in enumerate(factors.items()):
            factor_name = factor_info.get('factor_name', factor_id)
            
            # 进度显示
            if (i + 1) % 20 == 0 or i == 0:
                logger.info(f"  进度: {i+1}/{total}")
            
            # 加载因子值（按优先级尝试）
            factor_series, source = self.load_factor_with_fallback(factor_info, year)
            
            if factor_series is None or len(factor_series) == 0:
                failed_count += 1
                logger.debug(f"  跳过 {factor_name}: 无法获取因子值")
                results.append({
                    'factor_name': factor_name,
                    'annual_ic': None,
                    'annual_rank_ic': None,
                    'ic_ir': None,
                    'rank_ic_ir': None
                })
                continue
            
            # 统计来源
            if source == 'cache_location':
                cache_location_hit += 1
            elif source == 'md5_cache':
                md5_cache_hit += 1
            elif source == 'calculated':
                calculated_count += 1
            
            # 计算 IC
            ic_result = self.calculate_factor_ic(factor_series, label_df)
            
            if ic_result:
                results.append({
                    'factor_name': factor_name,
                    'annual_ic': ic_result['annual_ic'],
                    'annual_rank_ic': ic_result['annual_rank_ic'],
                    'ic_ir': ic_result['ic_ir'],
                    'rank_ic_ir': ic_result['rank_ic_ir']
                })
                success_count += 1
            else:
                results.append({
                    'factor_name': factor_name,
                    'annual_ic': None,
                    'annual_rank_ic': None,
                    'ic_ir': None,
                    'rank_ic_ir': None
                })
        
        logger.info(f"\n✓ IC计算完成:")
        logger.info(f"  成功: {success_count}/{total}")
        logger.info(f"  - cache_location 命中: {cache_location_hit}")
        logger.info(f"  - MD5 缓存命中: {md5_cache_hit}")
        logger.info(f"  - 实时计算: {calculated_count}")
        logger.info(f"  - 失败: {failed_count}")
        
        return pd.DataFrame(results)
    
    def run(self, factor_json: str, year: int, output_path: str) -> pd.DataFrame:
        """
        主运行方法
        
        Args:
            factor_json: 因子库 JSON 文件路径
            year: 目标年份
            output_path: 输出 CSV 文件路径
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 年度 IC 计算工具")
        logger.info(f"  因子库: {factor_json}")
        logger.info(f"  年份: {year}")
        logger.info(f"  市场: {self.market}")
        logger.info(f"  缓存目录: {self.cache_dir}")
        logger.info(f"{'='*60}\n")
        
        # 加载因子库
        factors = self.load_factor_library(factor_json)
        
        # 计算 IC
        result_df = self.calculate_all_factors(factors, year)
        
        # 按 Rank IC 降序排序
        result_df = result_df.sort_values('annual_rank_ic', ascending=False, na_position='last')
        
        # 保存结果
        result_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ 结果已保存: {output_path}")
        
        # 打印统计摘要
        valid_df = result_df.dropna(subset=['annual_rank_ic'])
        
        if len(valid_df) > 0:
            print(f"\n📈 统计摘要:")
            print(f"  有效因子数: {len(valid_df)}/{len(result_df)}")
            print(f"  平均 Rank IC: {valid_df['annual_rank_ic'].mean():.6f}")
            print(f"  最大 Rank IC: {valid_df['annual_rank_ic'].max():.6f}")
            print(f"  最小 Rank IC: {valid_df['annual_rank_ic'].min():.6f}")
            print(f"  Rank IC > 0: {(valid_df['annual_rank_ic'] > 0).sum()} 个")
            print(f"  Rank IC < 0: {(valid_df['annual_rank_ic'] < 0).sum()} 个")
            
            print(f"\n📈 Top 10 因子:")
            for _, row in valid_df.head(10).iterrows():
                name = row['factor_name'][:45]
                ric = row['annual_rank_ic']
                print(f"  {name:<45} Rank IC: {ric:.6f}")
            
            print(f"\n📉 Bottom 10 因子:")
            for _, row in valid_df.tail(10).iterrows():
                name = row['factor_name'][:45]
                ric = row['annual_rank_ic']
                print(f"  {name:<45} Rank IC: {ric:.6f}")
        
        return result_df


def main():
    parser = argparse.ArgumentParser(
        description='年度因子 IC 计算工具（支持多种缓存来源 + 实时计算）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算 2023 年 CSI300 上的 IC
  python yearly_ic_calculator.py \\
      --factor-json /path/to/factors.json \\
      --year 2023 \\
      --market csi300 \\
      --output /path/to/output.csv
        """
    )
    
    parser.add_argument(
        '-j', '--factor-json',
        type=str,
        required=True,
        help='因子库 JSON 文件路径'
    )
    
    parser.add_argument(
        '-y', '--year',
        type=int,
        required=True,
        help='目标年份 (e.g., 2023)'
    )
    
    parser.add_argument(
        '-m', '--market',
        type=str,
        default='csi300',
        help='股票池 (默认: csi300)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出 CSV 文件路径 (默认: 因子库同目录下的 {market}_{year}_ic_metrics.csv)'
    )
    
    parser.add_argument(
        '--provider-uri',
        type=str,
        default='/home/tjxy/.qlib/qlib_data/cn_data',
        help='Qlib 数据目录'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help=f'MD5 缓存目录 (默认: {DEFAULT_CACHE_DIR})'
    )
    
    args = parser.parse_args()
    
    # 默认输出路径
    if args.output is None:
        factor_dir = Path(args.factor_json).parent
        args.output = str(factor_dir / f"{args.market}_{args.year}_ic_metrics.csv")
    
    # 创建计算器并运行
    calculator = YearlyICCalculator(
        market=args.market,
        provider_uri=args.provider_uri,
        cache_dir=Path(args.cache_dir)
    )
    
    calculator.run(
        factor_json=args.factor_json,
        year=args.year,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
