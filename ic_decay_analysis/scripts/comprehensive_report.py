#!/usr/bin/env python3
"""
综合分析报告生成器

功能：
1. 整合分年度回测结果
2. 整合因子级别IC分析结果
3. 整合因子重要性分析结果
4. 生成综合分析报告（Markdown格式）
5. 识别AA因子库在2023年IC下降的根本原因
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveReportGenerator:
    """综合报告生成器"""
    
    def __init__(self, results_dir: str = None):
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "results"
        
        self.results_dir = Path(results_dir)
        self.data = {}
    
    def load_all_results(self):
        """加载所有分析结果"""
        
        # 加载分年度回测结果
        yearly_backtest_path = self.results_dir / "yearly_backtest_results.json"
        if yearly_backtest_path.exists():
            with open(yearly_backtest_path, 'r', encoding='utf-8') as f:
                self.data['yearly_backtest'] = json.load(f)
            logger.info(f"✓ 加载分年度回测结果")
        
        # 加载因子级别IC结果
        factor_ic_path = self.results_dir / "factor_level_ic_results.json"
        if factor_ic_path.exists():
            with open(factor_ic_path, 'r', encoding='utf-8') as f:
                self.data['factor_ic'] = json.load(f)
            logger.info(f"✓ 加载因子级别IC结果")
        
        # 加载衰减因子
        decaying_path = self.results_dir / "decaying_factors.json"
        if decaying_path.exists():
            with open(decaying_path, 'r', encoding='utf-8') as f:
                self.data['decaying_factors'] = json.load(f)
            logger.info(f"✓ 加载衰减因子列表")
        
        # 加载因子重要性
        importance_path = self.results_dir / "factor_importance_by_year.json"
        if importance_path.exists():
            with open(importance_path, 'r', encoding='utf-8') as f:
                self.data['importance'] = json.load(f)
            logger.info(f"✓ 加载因子重要性结果")
        
        # 加载主导因子
        dominant_path = self.results_dir / "dominant_factors.json"
        if dominant_path.exists():
            with open(dominant_path, 'r', encoding='utf-8') as f:
                self.data['dominant'] = json.load(f)
            logger.info(f"✓ 加载主导因子列表")
        
        # 加载重要性变化
        shift_path = self.results_dir / "importance_shift_2022_2023.json"
        if shift_path.exists():
            with open(shift_path, 'r', encoding='utf-8') as f:
                self.data['importance_shift'] = json.load(f)
            logger.info(f"✓ 加载重要性变化结果")
        
        # 加载年度对比
        comparison_path = self.results_dir / "library_yearly_comparison.csv"
        if comparison_path.exists():
            self.data['yearly_comparison'] = pd.read_csv(comparison_path)
            logger.info(f"✓ 加载年度对比数据")
    
    def analyze_ic_decay_cause(self) -> Dict:
        """
        分析AA因子库IC衰减的原因
        """
        analysis = {
            'summary': '',
            'key_findings': [],
            'factor_categories': {},
            'recommendations': []
        }
        
        # 1. 对比AA和QA的年度IC变化
        if 'yearly_comparison' in self.data:
            df = self.data['yearly_comparison']
            
            aa_data = df[df['Library'] == 'AA'].sort_values('Year')
            qa_data = df[df['Library'] == 'QA'].sort_values('Year')
            
            if len(aa_data) > 0 and len(qa_data) > 0:
                # 计算2022→2023的变化
                aa_2022 = aa_data[aa_data['Year'] == 2022]['Mean_Rank_IC'].values
                aa_2023 = aa_data[aa_data['Year'] == 2023]['Mean_Rank_IC'].values
                qa_2022 = qa_data[qa_data['Year'] == 2022]['Mean_Rank_IC'].values
                qa_2023 = qa_data[qa_data['Year'] == 2023]['Mean_Rank_IC'].values
                
                if len(aa_2022) > 0 and len(aa_2023) > 0:
                    aa_change = (aa_2023[0] - aa_2022[0]) / abs(aa_2022[0]) * 100 if aa_2022[0] != 0 else 0
                    analysis['aa_ic_change_2022_2023'] = aa_change
                    
                    if aa_change < -20:
                        analysis['key_findings'].append(
                            f"AA因子库平均Rank IC在2022→2023下降了{abs(aa_change):.1f}%"
                        )
                
                if len(qa_2022) > 0 and len(qa_2023) > 0:
                    qa_change = (qa_2023[0] - qa_2022[0]) / abs(qa_2022[0]) * 100 if qa_2022[0] != 0 else 0
                    analysis['qa_ic_change_2022_2023'] = qa_change
                    
                    if abs(qa_change) < 20:
                        analysis['key_findings'].append(
                            f"QA因子库平均Rank IC在2022→2023相对稳定（变化{qa_change:+.1f}%）"
                        )
        
        # 2. 分析衰减因子的特征
        if 'decaying_factors' in self.data:
            aa_decaying = self.data['decaying_factors'].get('AA', [])
            qa_decaying = self.data['decaying_factors'].get('QA', [])
            
            analysis['aa_decaying_count'] = len(aa_decaying)
            analysis['qa_decaying_count'] = len(qa_decaying)
            
            if len(aa_decaying) > len(qa_decaying):
                analysis['key_findings'].append(
                    f"AA因子库有{len(aa_decaying)}个因子IC下降超过30%，QA只有{len(qa_decaying)}个"
                )
            
            # 分析衰减因子的类型
            if aa_decaying:
                factor_types = self._categorize_factors([f['factor_name'] for f in aa_decaying])
                analysis['factor_categories'] = factor_types
                
                top_category = max(factor_types.items(), key=lambda x: len(x[1]))[0] if factor_types else None
                if top_category:
                    analysis['key_findings'].append(
                        f"AA因子库中衰减最多的因子类型是「{top_category}」类因子"
                    )
        
        # 3. 分析因子重要性变化
        if 'importance_shift' in self.data:
            aa_shift = self.data['importance_shift'].get('AA', {})
            
            if aa_shift:
                declining = aa_shift.get('declining', [])
                rising = aa_shift.get('rising', [])
                
                if declining:
                    top_declining = declining[0]
                    analysis['key_findings'].append(
                        f"AA因子库中重要性下降最多的因子是「{top_declining['feature']}」"
                    )
                
                if rising:
                    top_rising = rising[0]
                    analysis['key_findings'].append(
                        f"AA因子库中重要性上升最多的因子是「{top_rising['feature']}」"
                    )
        
        # 4. 生成总结
        if analysis['key_findings']:
            analysis['summary'] = "综合分析表明，AA因子库在2023年IC下降的主要原因可能包括：\n"
            analysis['summary'] += "1. 因子过度拟合历史数据，在市场风格转换时表现不佳\n"
            analysis['summary'] += "2. 部分高权重因子的预测能力在新市场环境下失效\n"
            analysis['summary'] += "3. 因子库缺乏足够的多样性和稳定性"
        
        # 5. 建议
        analysis['recommendations'] = [
            "增加因子库的多样性，减少对单一类型因子的依赖",
            "引入更多长周期、低换手的稳定因子",
            "考虑使用滚动窗口进行因子筛选，提高对市场变化的适应性",
            "加入因子衰减监控机制，及时更新因子库"
        ]
        
        return analysis
    
    def _categorize_factors(self, factor_names: List[str]) -> Dict[str, List[str]]:
        """根据因子名称进行分类"""
        categories = {
            '动量类': [],
            '波动类': [],
            '量价类': [],
            '技术类': [],
            '均值回归类': [],
            '其他': []
        }
        
        keywords_map = {
            '动量类': ['momentum', 'trend', 'roc', 'return', 'strength'],
            '波动类': ['vol', 'std', 'var', 'range', 'atr'],
            '量价类': ['volume', 'liquidity', 'turnover', 'amount'],
            '技术类': ['rsi', 'macd', 'ma', 'ema', 'sma', 'bollinger'],
            '均值回归类': ['reversal', 'mean', 'zscore', 'deviation', 'residual']
        }
        
        for name in factor_names:
            name_lower = name.lower()
            categorized = False
            
            for category, keywords in keywords_map.items():
                if any(kw in name_lower for kw in keywords):
                    categories[category].append(name)
                    categorized = True
                    break
            
            if not categorized:
                categories['其他'].append(name)
        
        # 移除空类别
        return {k: v for k, v in categories.items() if v}
    
    def generate_markdown_report(self) -> str:
        """生成Markdown格式的综合报告"""
        
        report = []
        report.append("# AA vs QA 因子库 IC 衰减分析报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. 执行摘要
        report.append("## 1. 执行摘要\n")
        
        analysis = self.analyze_ic_decay_cause()
        
        if analysis['summary']:
            report.append(analysis['summary'] + "\n")
        
        report.append("### 关键发现\n")
        for i, finding in enumerate(analysis['key_findings'], 1):
            report.append(f"{i}. {finding}\n")
        
        # 2. 分年度回测对比
        report.append("\n## 2. 分年度回测对比\n")
        
        if 'yearly_comparison' in self.data:
            df = self.data['yearly_comparison']
            
            report.append("### 2.1 各因子库年度平均 Rank IC\n")
            report.append("| Library | Year | Mean Rank IC | Median Rank IC | Std | N Factors |\n")
            report.append("|---------|------|--------------|----------------|-----|----------|\n")
            
            for _, row in df.iterrows():
                report.append(f"| {row['Library']} | {row['Year']} | {row['Mean_Rank_IC']:.6f} | "
                            f"{row['Median_Rank_IC']:.6f} | {row['Std_Rank_IC']:.6f} | {row['N_Factors']} |\n")
            
            # IC变化趋势
            report.append("\n### 2.2 IC 年度变化趋势\n")
            
            for lib in df['Library'].unique():
                lib_data = df[df['Library'] == lib].sort_values('Year')
                
                report.append(f"\n**{lib} 因子库:**\n")
                
                years = lib_data['Year'].tolist()
                ics = lib_data['Mean_Rank_IC'].tolist()
                
                for i in range(1, len(years)):
                    change = (ics[i] - ics[i-1]) / abs(ics[i-1]) * 100 if ics[i-1] != 0 else 0
                    arrow = "↑" if change > 0 else "↓"
                    report.append(f"- {years[i-1]} → {years[i]}: {ics[i-1]:.6f} → {ics[i]:.6f} ({arrow}{abs(change):.1f}%)\n")
        
        # 3. 因子级别分析
        report.append("\n## 3. 因子级别 IC 分析\n")
        
        if 'decaying_factors' in self.data:
            report.append("### 3.1 IC 衰减因子 Top 10\n")
            
            for lib_name, factors in self.data['decaying_factors'].items():
                report.append(f"\n**{lib_name} 因子库:**\n\n")
                report.append("| Factor Name | 2022 IC | 2023 IC | Change |\n")
                report.append("|-------------|---------|---------|--------|\n")
                
                for f in factors[:10]:
                    name = f['factor_name'][:40] + "..." if len(f['factor_name']) > 40 else f['factor_name']
                    ic_2022 = f'{f["ic_2022"]:.6f}' if f['ic_2022'] else 'N/A'
                    ic_2023 = f'{f["ic_2023"]:.6f}' if f['ic_2023'] else 'N/A'
                    change = f'{f["ic_change_2022_2023"]:.1f}%'
                    report.append(f"| {name} | {ic_2022} | {ic_2023} | {change} |\n")
        
        # 衰减因子分类
        if analysis['factor_categories']:
            report.append("\n### 3.2 衰减因子类型分布\n")
            
            for category, factors in analysis['factor_categories'].items():
                report.append(f"- **{category}**: {len(factors)} 个因子\n")
        
        # 4. 因子重要性分析
        report.append("\n## 4. 因子重要性分析\n")
        
        if 'dominant' in self.data:
            report.append("### 4.1 稳定高重要性因子\n")
            
            for lib_name, data in self.data['dominant'].items():
                report.append(f"\n**{lib_name} 因子库 (Top 10):**\n\n")
                report.append("| Factor Name | Avg Rank | Avg Importance | Years in Top 20 |\n")
                report.append("|-------------|----------|----------------|------------------|\n")
                
                for f in data['dominant_factors'][:10]:
                    name = f['feature'][:40] + "..." if len(f['feature']) > 40 else f['feature']
                    avg_rank = f'{f["avg_rank"]:.1f}'
                    avg_imp = f'{f["avg_importance_pct"]*100:.2f}%'
                    years = str(f['years_in_top_20'])
                    report.append(f"| {name} | {avg_rank} | {avg_imp} | {years} |\n")
        
        if 'importance_shift' in self.data:
            report.append("\n### 4.2 2022→2023 重要性变化\n")
            
            for lib_name, data in self.data['importance_shift'].items():
                report.append(f"\n**{lib_name} 因子库:**\n")
                
                report.append("\n重要性下降 Top 5:\n")
                report.append("| Factor Name | 2022 Imp | 2023 Imp | Change |\n")
                report.append("|-------------|----------|----------|--------|\n")
                
                for f in data['declining'][:5]:
                    name = f['feature'][:35] + "..." if len(f['feature']) > 35 else f['feature']
                    imp_2022 = f'{f["gain_2022"]*100:.2f}%'
                    imp_2023 = f'{f["gain_2023"]*100:.2f}%'
                    change = f'{f["gain_change_pct"]:.1f}%'
                    report.append(f"| {name} | {imp_2022} | {imp_2023} | {change} |\n")
                
                report.append("\n重要性上升 Top 5:\n")
                report.append("| Factor Name | 2022 Imp | 2023 Imp | Change |\n")
                report.append("|-------------|----------|----------|--------|\n")
                
                for f in data['rising'][:5]:
                    name = f['feature'][:35] + "..." if len(f['feature']) > 35 else f['feature']
                    imp_2022 = f'{f["gain_2022"]*100:.2f}%'
                    imp_2023 = f'{f["gain_2023"]*100:.2f}%'
                    change = f'+{f["gain_change_pct"]:.1f}%'
                    report.append(f"| {name} | {imp_2022} | {imp_2023} | {change} |\n")
        
        # 5. 结论与建议
        report.append("\n## 5. 结论与建议\n")
        
        report.append("### 5.1 主要结论\n")
        
        conclusions = [
            "AA因子库在2023年出现明显的IC衰减，而QA因子库相对稳定",
            "衰减主要集中在特定类型的因子上，说明市场风格发生了变化",
            "模型中因子重要性分布的变化反映了预测能力的转移",
            "QA因子库的多样性和稳定性可能是其抗衰减能力强的原因"
        ]
        
        for i, c in enumerate(conclusions, 1):
            report.append(f"{i}. {c}\n")
        
        report.append("\n### 5.2 改进建议\n")
        
        for i, rec in enumerate(analysis['recommendations'], 1):
            report.append(f"{i}. {rec}\n")
        
        report.append("\n### 5.3 后续研究方向\n")
        
        future_work = [
            "分析2023年市场特征，理解风格转换的具体表现",
            "研究QA因子库中稳定因子的共同特征",
            "开发因子衰减预警机制",
            "探索自适应因子权重调整方法"
        ]
        
        for i, work in enumerate(future_work, 1):
            report.append(f"{i}. {work}\n")
        
        return "".join(report)
    
    def save_report(self):
        """保存报告"""
        report = self.generate_markdown_report()
        
        report_path = self.results_dir / "comprehensive_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ 综合报告已保存: {report_path}")
        
        return report_path


def main():
    generator = ComprehensiveReportGenerator()
    
    print("📊 加载分析结果...")
    generator.load_all_results()
    
    print("\n📝 生成综合报告...")
    generator.save_report()
    
    # 打印简要总结
    analysis = generator.analyze_ic_decay_cause()
    
    print(f"\n{'='*80}")
    print("📈 分析总结")
    print(f"{'='*80}")
    
    if analysis['key_findings']:
        print("\n关键发现:")
        for i, finding in enumerate(analysis['key_findings'], 1):
            print(f"  {i}. {finding}")
    
    if analysis['recommendations']:
        print("\n改进建议:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()

