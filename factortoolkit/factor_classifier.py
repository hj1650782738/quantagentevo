#!/usr/bin/env python3
"""
因子库分类脚本
支持三种分类方式：
1. quality - 按RankIC值分类（>0.02: high_quality, 0-0.02: valid, <0: poor）
2. round_number - 按round_number值分类
3. initial_direction - 按initial_direction分类
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


class FactorClassifier:
    MAX_FACTORS_PER_FILE = 60  # 每个文件最多包含的因子数量
    
    def __init__(self, input_file: str, output_base_path: str):
        """
        初始化分类器
        
        Args:
            input_file: 输入因子库JSON文件路径
            output_base_path: 输出基础路径
        """
        self.input_file = input_file
        self.output_base_path = Path(output_base_path)
        self.factors = {}
        self.load_factors()
    
    def load_factors(self):
        """加载因子库"""
        print(f"📖 正在加载因子库: {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.factors = data.get('factors', {})
        print(f"✅ 已加载 {len(self.factors)} 个因子")
    
    def save_factors_to_files(self, factors_list: List[Dict[str, Any]], base_file_path: Path, 
                              metadata: Dict[str, Any], category_name: str = ""):
        """
        将因子列表保存到文件，如果超过MAX_FACTORS_PER_FILE则分多个文件
        
        Args:
            factors_list: 因子列表
            base_file_path: 基础文件路径（不含后缀，如 "high_quality"）
            metadata: 元数据字典
            category_name: 分类名称（用于日志输出）
        
        Returns:
            保存的文件数量
        """
        total_factors = len(factors_list)
        if total_factors == 0:
            return 0
        
        # 如果因子数量不超过限制，直接保存
        if total_factors <= self.MAX_FACTORS_PER_FILE:
            output_file = base_file_path.with_suffix('.json')
            result = {
                "metadata": {**metadata, "total_factors": total_factors},
                "factors": {f["factor_id"]: f for f in factors_list}
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ {output_file.name}: {total_factors} 个因子")
            return 1
        
        # 需要分成多个文件
        num_files = (total_factors + self.MAX_FACTORS_PER_FILE - 1) // self.MAX_FACTORS_PER_FILE
        
        for file_idx in range(num_files):
            start_idx = file_idx * self.MAX_FACTORS_PER_FILE
            end_idx = min(start_idx + self.MAX_FACTORS_PER_FILE, total_factors)
            chunk = factors_list[start_idx:end_idx]
            
            # 生成文件名：如果有多个文件，添加 "_1"、"_2" 等后缀
            if num_files > 1:
                file_name = f"{base_file_path.stem}_{file_idx + 1}.json"
            else:
                file_name = f"{base_file_path.stem}.json"
            
            output_file = base_file_path.parent / file_name
            
            result = {
                "metadata": {
                    **metadata,
                    "total_factors": len(chunk),
                    "file_index": file_idx + 1,
                    "total_files": num_files,
                    "is_split": num_files > 1
                },
                "factors": {f["factor_id"]: f for f in chunk}
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ {file_name}: {len(chunk)} 个因子" + 
                  (f" (共 {num_files} 个文件，第 {file_idx + 1} 个)" if num_files > 1 else ""))
        
        return num_files
    
    def classify_by_quality(self):
        """
        按quality分类（实际按RankIC值）
        - RankIC > 0.02 → high_quality.json
        - 0 <= RankIC <= 0.02 → valid.json
        - RankIC < 0 → poor.json
        """
        print("\n📊 开始按 quality (RankIC) 分类...")
        
        output_dir = self.output_base_path / "quality"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        categories = {
            'high_quality': [],  # RankIC > 0.02
            'valid': [],         # 0 <= RankIC <= 0.02
            'poor': []           # RankIC < 0
        }
        
        null_count = 0
        
        for factor_id, factor in self.factors.items():
            rankic = factor.get('backtest_metrics', {}).get('RankIC')
            
            if rankic is None:
                null_count += 1
                # None值不分类，跳过
                continue
            elif rankic > 0.02:
                categories['high_quality'].append(factor)
            elif rankic >= 0:
                categories['valid'].append(factor)
            else:  # rankic < 0
                categories['poor'].append(factor)
        
        # 保存分类结果
        for category, factors_list in categories.items():
            base_file_path = output_dir / category
            metadata = {
                "classification_type": "quality",
                "classification_rule": "RankIC",
                "category": category
            }
            self.save_factors_to_files(factors_list, base_file_path, metadata, category)
        
        if null_count > 0:
            print(f"  ⚠️  跳过 {null_count} 个 RankIC 为 null 的因子")
        
        print(f"✅ quality 分类完成，输出目录: {output_dir}")
    
    def classify_by_round_number(self):
        """
        按round_number分类
        相同round_number值的因子放在同一个JSON文件中
        """
        print("\n📊 开始按 round_number 分类...")
        
        output_dir = self.output_base_path / "round_number"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 按round_number分组
        grouped = defaultdict(list)
        null_count = 0
        
        for factor_id, factor in self.factors.items():
            round_num = factor.get('round_number')
            if round_num is None:
                null_count += 1
                continue
            grouped[round_num].append(factor)
        
        # 保存每个round_number的文件
        for round_num in sorted(grouped.keys()):
            factors_list = grouped[round_num]
            base_file_path = output_dir / f"round_{round_num}"
            metadata = {
                "classification_type": "round_number",
                "round_number": round_num
            }
            self.save_factors_to_files(factors_list, base_file_path, metadata, f"round_{round_num}")
        
        if null_count > 0:
            print(f"  ⚠️  跳过 {null_count} 个 round_number 为 null 的因子")
        
        print(f"✅ round_number 分类完成，输出目录: {output_dir}")
    
    def classify_by_initial_direction(self):
        """
        按initial_direction分类
        - 没有该字段的因子放入"无方向.json"
        - 有该字段的按值分组，值太长的用"初始1"、"初始2"等命名
        """
        print("\n📊 开始按 initial_direction 分类...")
        
        output_dir = self.output_base_path / "initial_direction"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 按initial_direction分组
        grouped = defaultdict(list)
        no_direction = []
        
        for factor_id, factor in self.factors.items():
            initial_dir = factor.get('initial_direction')
            if initial_dir is None:
                no_direction.append(factor)
            else:
                grouped[initial_dir].append(factor)
        
        # 保存无方向的因子
        if no_direction:
            base_file_path = output_dir / "无方向"
            metadata = {
                "classification_type": "initial_direction",
                "category": "无方向"
            }
            self.save_factors_to_files(no_direction, base_file_path, metadata, "无方向")
        
        # 保存有方向的因子
        # 为每个唯一的initial_direction值创建文件
        # 如果值太长，使用"初始1"、"初始2"等命名
        direction_to_index = {}
        index = 1
        
        for direction_value in sorted(grouped.keys()):
            factors_list = grouped[direction_value]
            
            # 判断值是否太长（超过50个字符）
            if len(direction_value) > 50:
                if direction_value not in direction_to_index:
                    direction_to_index[direction_value] = index
                    index += 1
                base_file_name = f"初始{direction_to_index[direction_value]}"
            else:
                # 使用值作为文件名（清理特殊字符）
                safe_name = direction_value.replace('/', '_').replace('\\', '_').replace(':', '_')
                safe_name = safe_name.replace('*', '_').replace('?', '_').replace('"', '_')
                safe_name = safe_name.replace('<', '_').replace('>', '_').replace('|', '_')
                # 限制文件名长度
                if len(safe_name) > 100:
                    safe_name = safe_name[:100]
                base_file_name = safe_name
            
            # 构建基础文件路径（不含.json后缀）
            base_file_path = output_dir / base_file_name
            metadata = {
                "classification_type": "initial_direction",
                "initial_direction": direction_value
            }
            
            # 用于显示的原始文件名（带.json后缀）
            display_name = f"{base_file_name}.json"
            
            num_files = self.save_factors_to_files(factors_list, base_file_path, metadata, display_name)
            
            if len(direction_value) > 50 and num_files > 0:
                print(f"      (原始值: {direction_value[:80]}...)")
        
        print(f"✅ initial_direction 分类完成，输出目录: {output_dir}")
    
    def classify_all(self, methods: List[str] = None):
        """
        执行所有分类
        
        Args:
            methods: 要执行的分类方法列表，None表示执行所有
        """
        if methods is None:
            methods = ['quality', 'round_number', 'initial_direction']
        
        print("=" * 60)
        print("🚀 开始因子库分类")
        print("=" * 60)
        
        if 'quality' in methods:
            self.classify_by_quality()
        
        if 'round_number' in methods:
            self.classify_by_round_number()
        
        if 'initial_direction' in methods:
            self.classify_by_initial_direction()
        
        print("\n" + "=" * 60)
        print("✅ 所有分类完成！")
        print(f"📁 输出目录: {self.output_base_path}")
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='因子库分类工具')
    parser.add_argument(
        '--input',
        type=str,
        default='/home/tjxy/quantagent/AlphaAgent/all_factors_library.json',
        help='输入因子库JSON文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/tjxy/.qlib/factor_data',
        help='输出基础路径'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['quality', 'round_number', 'initial_direction', 'all'],
        default='all',
        help='分类方法: quality, round_number, initial_direction, 或 all'
    )
    
    args = parser.parse_args()
    
    classifier = FactorClassifier(args.input, args.output)
    
    if args.method == 'all':
        classifier.classify_all()
    else:
        classifier.classify_all([args.method])


if __name__ == '__main__':
    main()

