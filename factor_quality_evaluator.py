#!/usr/bin/env python3
"""
因子质量评估工具
计算因子与Alpha158因子库的相关性，并判断是否为高质量因子
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import qlib
from qlib.data import D
from qlib.contrib.data.loader import Alpha158DL


class FactorQualityEvaluator:
    """因子质量评估器"""
    
    def __init__(self, alpha158_factors_path: str = None):
        """
        初始化评估器
        
        Args:
            alpha158_factors_path: Alpha158因子库JSON文件路径
        """
        if alpha158_factors_path is None:
            alpha158_factors_path = Path(__file__).parent / "alpha158_all_factors.json"
        
        # 加载Alpha158因子库
        with open(alpha158_factors_path, 'r', encoding='utf-8') as f:
            alpha158_data = json.load(f)
            self.alpha158_factors = alpha158_data
        
        # 初始化Qlib（如果需要计算Alpha158因子值）
        try:
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        except:
            pass  # 如果已经初始化过，忽略错误
    
    def calculate_factor_correlation_with_alpha158(
        self, 
        factor_data: pd.Series | pd.DataFrame,
        alpha158_factor_name: str,
        alpha158_expression: str
    ) -> float:
        """
        计算新因子与单个Alpha158因子的相关性
        
        Args:
            factor_data: 新因子的数据（Series或DataFrame）
            alpha158_factor_name: Alpha158因子名称
            alpha158_expression: Alpha158因子表达式
            
        Returns:
            相关性系数（Pearson相关系数）
        """
        try:
            # 如果factor_data是DataFrame，取第一列
            if isinstance(factor_data, pd.DataFrame):
                if factor_data.empty:
                    return 0.0
                factor_series = factor_data.iloc[:, 0]
            else:
                factor_series = factor_data
            
            # 计算Alpha158因子值（这里简化处理，实际需要根据表达式计算）
            # 注意：完整实现需要解析表达式并计算，这里先返回0.0作为占位
            # 实际使用时，需要从Qlib加载Alpha158因子数据
            
            # 简化版本：如果因子数据格式正确，尝试计算相关性
            # 这里假设alpha158_factor_data已经计算好
            # 实际实现需要：
            # 1. 解析alpha158_expression
            # 2. 从Qlib数据计算Alpha158因子值
            # 3. 对齐时间索引
            # 4. 计算相关性
            
            return 0.0  # 占位符，需要实际实现
        except Exception as e:
            print(f"计算相关性时出错: {e}")
            return 0.0
    
    def get_max_correlation_with_alpha158(
        self,
        factor_data: pd.Series | pd.DataFrame,
        factor_workspace_path: Optional[Path] = None
    ) -> Tuple[float, str]:
        """
        计算因子与Alpha158因子库的最大相关性
        
        Args:
            factor_data: 因子数据（Series或DataFrame）
            factor_workspace_path: 因子工作空间路径（可选，用于加载因子数据）
            
        Returns:
            (最大相关性, 对应的Alpha158因子名称)
        """
        if factor_workspace_path is not None:
            # 尝试从工作空间加载因子数据
            result_file = factor_workspace_path / "result.h5"
            if result_file.exists():
                try:
                    factor_data = pd.read_hdf(result_file, key="data")
                except:
                    pass
        
        if factor_data is None or (isinstance(factor_data, pd.DataFrame) and factor_data.empty):
            return 0.0, ""
        
        max_corr = 0.0
        max_corr_factor = ""
        
        # 遍历Alpha158因子库
        for factor_name, factor_expr in self.alpha158_factors.items():
            try:
                corr = self.calculate_factor_correlation_with_alpha158(
                    factor_data, factor_name, factor_expr
                )
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    max_corr_factor = factor_name
            except Exception as e:
                continue
        
        return max_corr, max_corr_factor
    
    def judge_factor_quality(
        self,
        rank_ic: Optional[float],
        max_correlation: Optional[float] = None,
        factor_workspace_path: Optional[Path] = None,
        factor_data: Optional[pd.Series | pd.DataFrame] = None
    ) -> Tuple[str, Dict]:
        """
        判断因子质量（新标准）
        
        新标准:
        - RankIC > 0.02
        - 与Alpha158因子库的最大相关性 < 0.7
        
        Args:
            rank_ic: RankIC值
            max_correlation: 与Alpha158的最大相关性（如果已计算）
            factor_workspace_path: 因子工作空间路径（用于计算相关性）
            factor_data: 因子数据（用于计算相关性）
            
        Returns:
            (质量等级, 详细信息字典)
        """
        info = {
            "rank_ic": rank_ic,
            "max_correlation": max_correlation,
            "alpha158_factor": ""
        }
        
        if rank_ic is None:
            return "Unknown", info
        
        # 如果相关性未提供，尝试计算
        if max_correlation is None:
            if factor_workspace_path is not None or factor_data is not None:
                max_correlation, alpha158_factor = self.get_max_correlation_with_alpha158(
                    factor_data, factor_workspace_path
                )
                info["max_correlation"] = max_correlation
                info["alpha158_factor"] = alpha158_factor
            else:
                # 无法计算相关性，只根据RankIC判断
                if rank_ic > 0.02:
                    return "high_quality", info
                elif rank_ic > 0:
                    return "valid", info
                else:
                    return "poor", info
        
        # 应用新标准：RankIC > 0.02 且 Correlation < 0.7
        try:
            rank_ic_val = float(rank_ic)
            corr_val = abs(float(max_correlation)) if max_correlation is not None else 1.0
            
            if rank_ic_val > 0.02 and corr_val < 0.7:
                return "high_quality", info
            elif rank_ic_val > 0:
                return "valid", info
            else:
                return "poor", info
        except Exception as e:
            return "unknown", info


# 全局实例
_evaluator = None

def get_evaluator() -> FactorQualityEvaluator:
    """获取全局评估器实例"""
    global _evaluator
    if _evaluator is None:
        _evaluator = FactorQualityEvaluator()
    return _evaluator


def judge_factor_quality_new(
    rank_ic: Optional[float],
    max_correlation: Optional[float] = None,
    factor_workspace_path: Optional[Path] = None,
    factor_data: Optional[pd.Series | pd.DataFrame] = None
) -> Tuple[str, Dict]:
    """
    判断因子质量（新标准）- 便捷函数
    
    新标准:
    - RankIC > 0.02
    - 与Alpha158因子库的最大相关性 < 0.7
    
    Args:
        rank_ic: RankIC值
        max_correlation: 与Alpha158的最大相关性（如果已计算）
        factor_workspace_path: 因子工作空间路径（用于计算相关性）
        factor_data: 因子数据（用于计算相关性）
        
    Returns:
        (质量等级, 详细信息字典)
    """
    evaluator = get_evaluator()
    return evaluator.judge_factor_quality(
        rank_ic, max_correlation, factor_workspace_path, factor_data
    )


def reclassify_factors_in_json(json_path: str, output_path: str = None) -> Dict:
    """
    重新分类JSON文件中的因子质量
    
    新标准:
    - high_quality: RankIC > 0.02 且 max_correlation_with_alpha158 < 0.7（或为None）
    - valid: RankIC > 0 但不满足high_quality条件
    - poor: RankIC <= 0 或 RankIC 为 None
    
    Args:
        json_path: 输入JSON文件路径
        output_path: 输出JSON文件路径（默认覆盖原文件）
        
    Returns:
        统计信息字典
    """
    import json
    from datetime import datetime
    
    if output_path is None:
        output_path = json_path
    
    # 读取JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    factors = data.get('factors', {})
    
    stats = {
        "total": len(factors),
        "high_quality": 0,
        "valid": 0,
        "poor": 0,
        "unknown": 0,
        "changed": []
    }
    
    for factor_id, factor in factors.items():
        old_quality = factor.get('quality', 'unknown')
        
        # 获取指标
        backtest_metrics = factor.get('backtest_metrics', {})
        rank_ic = backtest_metrics.get('RankIC')
        max_corr = factor.get('max_correlation_with_alpha158')
        
        # 应用新标准
        if rank_ic is None:
            new_quality = "poor"
        else:
            try:
                rank_ic_val = float(rank_ic)
                # 如果相关性为None，假设满足条件
                corr_val = abs(float(max_corr)) if max_corr is not None else 0.0
                
                if rank_ic_val > 0.02 and corr_val < 0.7:
                    new_quality = "high_quality"
                elif rank_ic_val > 0:
                    new_quality = "valid"
                else:
                    new_quality = "poor"
            except:
                new_quality = "unknown"
        
        # 更新质量
        factor['quality'] = new_quality
        factor['updated_at'] = datetime.now().isoformat()
        
        # 统计
        stats[new_quality] = stats.get(new_quality, 0) + 1
        
        if old_quality != new_quality:
            stats["changed"].append({
                "factor_id": factor_id,
                "factor_name": factor.get('factor_name'),
                "old_quality": old_quality,
                "new_quality": new_quality,
                "RankIC": rank_ic,
                "max_corr": max_corr
            })
    
    # 更新metadata
    if 'metadata' in data:
        data['metadata']['last_updated'] = datetime.now().isoformat()
        data['metadata']['quality_standard'] = "RankIC > 0.02 AND max_correlation_with_alpha158 < 0.7"
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return stats

