#!/usr/bin/env python3
"""
因子库分析工具 (Factor Library Analyzer)

分析因子库文件中的各种字段，生成统计报表和可视化图表。

使用方法:
    python tools/factor_analyzer.py <factor_library.json> [--output_dir <dir>]
    
示例:
    python tools/factor_analyzer.py all_factors_library.json
    python tools/factor_analyzer.py all_factors_library.json --output_dir analysis_output
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.font_manager import FontProperties, findfont, fontManager
    matplotlib.use('Agg')  # 非交互式后端
    
    # 中文字体配置函数
    def setup_chinese_font():
        """设置中文字体，自动检测系统可用字体"""
        # 常见中文字体列表（按优先级）
        chinese_font_names = [
            'SimHei',           # 黑体 (Windows)
            'Microsoft YaHei',  # 微软雅黑 (Windows)
            'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
            'WenQuanYi Zen Hei',    # 文泉驿正黑 (Linux)
            'Noto Sans CJK SC',     # Noto 字体简体中文 (Linux/Mac)
            'Noto Sans CJK TC',     # Noto 字体繁体中文
            'Noto Sans CJK JP',     # Noto 字体日文（也支持中文）
            'Source Han Sans CN',   # 思源黑体 (Linux/Mac)
            'STHeiti',          # 华文黑体 (Mac)
            'PingFang SC',      # 苹方 (Mac)
            'Arial Unicode MS', # Arial Unicode (跨平台)
        ]
        
        # 检测系统可用字体（包括字体名称和文件路径）
        available_fonts = {}
        cjk_font_files = []
        
        for font in fontManager.ttflist:
            font_name = font.name
            font_file = font.fname
            
            # 检查是否是 CJK 字体（支持中文）
            if any(kw in font_name for kw in ['CJK', 'SC', 'CN', 'Chinese', 'SimHei', 'YaHei', 'WenQuan']):
                if font_name not in cjk_font_files:
                    cjk_font_files.append((font_name, font_file))
            
            available_fonts[font_name] = font_file
        
        # 优先查找 CJK 字体
        selected_font = None
        selected_font_file = None
        
        # 1. 先尝试按名称匹配
        for font_name in chinese_font_names:
            if font_name in available_fonts:
                selected_font = font_name
                selected_font_file = available_fonts[font_name]
                break
        
        # 2. 如果没找到，尝试查找任何包含 CJK 的字体
        if not selected_font and cjk_font_files:
            # 优先选择 SC (简体中文) 或包含 SC 的
            for font_name, font_file in cjk_font_files:
                if 'SC' in font_name or 'CN' in font_name:
                    selected_font = font_name
                    selected_font_file = font_file
                    break
            
            # 如果还没找到，使用第一个 CJK 字体（JP/TC 也支持中文）
            if not selected_font:
                selected_font, selected_font_file = cjk_font_files[0]
        
        # 3. 如果找到了字体，直接加载字体文件
        if selected_font_file:
            try:
                # 使用字体文件路径直接加载
                font_prop = FontProperties(fname=selected_font_file)
                plt.rcParams['font.family'] = font_prop.get_name()
                # 同时设置字体列表
                plt.rcParams['font.sans-serif'] = [selected_font] + chinese_font_names
                print(f"  使用中文字体: {selected_font}")
                print(f"  字体路径: {selected_font_file}")
            except Exception as e:
                # 如果直接加载失败，使用字体名称
                plt.rcParams['font.sans-serif'] = [selected_font] + chinese_font_names
                print(f"  使用中文字体: {selected_font} (通过名称)")
        else:
            # 如果找不到中文字体，尝试系统默认
            try:
                default_font = findfont(FontProperties(family='sans-serif'))
                plt.rcParams['font.sans-serif'] = chinese_font_names + ['DejaVu Sans']
                print(f"  警告: 未找到中文字体，使用系统默认: {default_font}")
                print("  提示: 图表中的中文可能显示为方块，建议安装中文字体")
            except:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                print("  警告: 未找到中文字体，图表标签将使用英文")
        
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 测试中文显示
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontsize=10)
            plt.close(fig)
        except Exception as e:
            print(f"  警告: 中文字体测试失败: {e}")
    
    # 初始化字体
    setup_chinese_font()
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将跳过图表生成")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ============================================================
# 核心指标定义
# ============================================================
KEY_METRICS = ['RankIC', 'annualized_return', 'max_drawdown', 'information_ratio']
ALL_METRICS = ['IC', 'ICIR', 'RankIC', 'RankICIR', 'annualized_return', 'information_ratio', 'max_drawdown']


# ============================================================
# 数据加载
# ============================================================
def load_factor_library(filepath: str) -> tuple[dict, pd.DataFrame]:
    """
    加载因子库文件并转换为 DataFrame。
    
    Returns:
        (metadata, factors_df)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    factors_dict = data.get('factors', {})
    
    # 转换为 DataFrame
    records = []
    for fid, factor in factors_dict.items():
        record = {
            'factor_id': fid,
            'factor_name': factor.get('factor_name', ''),
            'factor_expression': factor.get('factor_expression', ''),
            'round_number': factor.get('round_number'),
            'evolution_phase': factor.get('evolution_phase', 'N/A'),
            'initial_direction': factor.get('initial_direction', ''),
            'planning_direction': factor.get('planning_direction', ''),
            'quality': factor.get('quality', 'N/A'),
            'is_sota': factor.get('is_sota', False),
            'trajectory_id': factor.get('trajectory_id', ''),
            'experiment_id': factor.get('experiment_id', ''),
            'added_at': factor.get('added_at', ''),
        }
        
        # 提取回测指标
        metrics = factor.get('backtest_metrics', {})
        for m in ALL_METRICS:
            record[m] = metrics.get(m)
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # 转换数据类型
    if 'round_number' in df.columns:
        df['round_number'] = pd.to_numeric(df['round_number'], errors='coerce')
    
    return metadata, df


# ============================================================
# 统计分析函数
# ============================================================
def analyze_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """按 round_number 分析因子指标。"""
    if df.empty or 'round_number' not in df.columns:
        return pd.DataFrame()
    
    grouped = df.groupby('round_number')
    
    stats = []
    for rn, group in grouped:
        stat = {'round_number': int(rn) if pd.notna(rn) else 'N/A'}
        stat['count'] = len(group)
        
        for m in KEY_METRICS:
            if m in group.columns:
                values = group[m].dropna()
                if len(values) > 0:
                    stat[f'{m}_mean'] = values.mean()
                    stat[f'{m}_std'] = values.std()
                    stat[f'{m}_min'] = values.min()
                    stat[f'{m}_max'] = values.max()
                    stat[f'{m}_median'] = values.median()
        
        stats.append(stat)
    
    return pd.DataFrame(stats).sort_values('round_number')


def analyze_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """按 evolution_phase 分析因子指标。"""
    if df.empty or 'evolution_phase' not in df.columns:
        return pd.DataFrame()
    
    grouped = df.groupby('evolution_phase')
    
    stats = []
    for phase, group in grouped:
        stat = {'evolution_phase': phase}
        stat['count'] = len(group)
        
        for m in KEY_METRICS:
            if m in group.columns:
                values = group[m].dropna()
                if len(values) > 0:
                    stat[f'{m}_mean'] = values.mean()
                    stat[f'{m}_std'] = values.std()
                    stat[f'{m}_median'] = values.median()
        
        stats.append(stat)
    
    return pd.DataFrame(stats)


def analyze_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    """按 quality 分析因子指标。"""
    if df.empty or 'quality' not in df.columns:
        return pd.DataFrame()
    
    grouped = df.groupby('quality')
    
    stats = []
    for quality, group in grouped:
        stat = {'quality': quality}
        stat['count'] = len(group)
        stat['percentage'] = f"{len(group) / len(df) * 100:.1f}%"
        
        for m in KEY_METRICS:
            if m in group.columns:
                values = group[m].dropna()
                if len(values) > 0:
                    stat[f'{m}_mean'] = values.mean()
                    stat[f'{m}_std'] = values.std()
        
        stats.append(stat)
    
    return pd.DataFrame(stats)


def analyze_by_direction(df: pd.DataFrame) -> pd.DataFrame:
    """按 initial_direction 分析因子指标。"""
    if df.empty or 'initial_direction' not in df.columns:
        return pd.DataFrame()
    
    # 截取方向文本前80字符作为分组键
    df = df.copy()
    df['direction_key'] = df['initial_direction'].fillna('N/A').str[:80]
    
    grouped = df.groupby('direction_key')
    
    stats = []
    for direction, group in grouped:
        stat = {'initial_direction': direction + '...' if len(direction) == 80 else direction}
        stat['count'] = len(group)
        
        for m in KEY_METRICS:
            if m in group.columns:
                values = group[m].dropna()
                if len(values) > 0:
                    stat[f'{m}_mean'] = values.mean()
                    stat[f'{m}_std'] = values.std()
        
        stats.append(stat)
    
    result = pd.DataFrame(stats)
    if 'RankIC_mean' in result.columns:
        result = result.sort_values('RankIC_mean', ascending=False)
    return result


def get_top_factors(df: pd.DataFrame, metric: str = 'RankIC', top_n: int = 10) -> pd.DataFrame:
    """获取指定指标的 Top N 因子。"""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    
    sorted_df = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    
    cols = ['factor_name', 'round_number', 'evolution_phase', 'quality']
    cols.extend([m for m in KEY_METRICS if m in df.columns])
    
    return sorted_df[cols].head(top_n)


def get_overall_statistics(df: pd.DataFrame) -> dict:
    """获取整体统计信息。"""
    stats = {
        'total_factors': len(df),
        'unique_rounds': df['round_number'].nunique() if 'round_number' in df.columns else 0,
        'unique_phases': df['evolution_phase'].nunique() if 'evolution_phase' in df.columns else 0,
        'unique_directions': df['initial_direction'].nunique() if 'initial_direction' in df.columns else 0,
    }
    
    for m in KEY_METRICS:
        if m in df.columns:
            values = df[m].dropna()
            if len(values) > 0:
                stats[f'{m}_overall_mean'] = values.mean()
                stats[f'{m}_overall_std'] = values.std()
                stats[f'{m}_overall_min'] = values.min()
                stats[f'{m}_overall_max'] = values.max()
    
    # 质量分布
    if 'quality' in df.columns:
        quality_counts = df['quality'].value_counts().to_dict()
        stats['quality_distribution'] = quality_counts
    
    return stats


# ============================================================
# 可视化函数
# ============================================================
def plot_round_analysis(df: pd.DataFrame, output_dir: Path):
    """绘制按轮次分析的图表。"""
    if not HAS_MATPLOTLIB or df.empty:
        return
    
    # 确保字体已正确设置
    try:
        setup_chinese_font()
    except:
        pass
    
    round_stats = analyze_by_round(df)
    if round_stats.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('按进化轮次分析 (Round Number Analysis)', fontsize=14, fontweight='bold')
    
    rounds = round_stats['round_number'].astype(str)
    
    # 1. 因子数量随轮次变化
    ax1 = axes[0, 0]
    ax1.bar(rounds, round_stats['count'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Round Number')
    ax1.set_ylabel('Factor Count')
    ax1.set_title('因子数量 vs 轮次')
    for i, v in enumerate(round_stats['count']):
        ax1.text(i, v + 0.5, str(v), ha='center', fontsize=9)
    
    # 2. RankIC 均值随轮次变化
    ax2 = axes[0, 1]
    if 'RankIC_mean' in round_stats.columns:
        ax2.plot(rounds, round_stats['RankIC_mean'], 'o-', color='green', linewidth=2, markersize=8)
        if 'RankIC_std' in round_stats.columns:
            ax2.fill_between(
                range(len(rounds)),
                round_stats['RankIC_mean'] - round_stats['RankIC_std'],
                round_stats['RankIC_mean'] + round_stats['RankIC_std'],
                alpha=0.2, color='green'
            )
        ax2.set_xlabel('Round Number')
        ax2.set_ylabel('RankIC Mean')
        ax2.set_title('RankIC 均值 vs 轮次 (±1 std)')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. 年化收益率随轮次变化
    ax3 = axes[1, 0]
    if 'annualized_return_mean' in round_stats.columns:
        ax3.bar(rounds, round_stats['annualized_return_mean'] * 100, color='coral', alpha=0.7)
        ax3.set_xlabel('Round Number')
        ax3.set_ylabel('Annualized Return (%)')
        ax3.set_title('年化收益率均值 vs 轮次')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 4. Information Ratio 随轮次变化
    ax4 = axes[1, 1]
    if 'information_ratio_mean' in round_stats.columns:
        ax4.plot(rounds, round_stats['information_ratio_mean'], 's-', color='purple', linewidth=2, markersize=8)
        ax4.set_xlabel('Round Number')
        ax4.set_ylabel('Information Ratio')
        ax4.set_title('信息比率均值 vs 轮次')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'round_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已生成: round_analysis.png")


def plot_phase_analysis(df: pd.DataFrame, output_dir: Path):
    """绘制按进化阶段分析的图表。"""
    if not HAS_MATPLOTLIB or df.empty:
        return
    
    # 确保字体已正确设置
    try:
        setup_chinese_font()
    except:
        pass
    
    phase_stats = analyze_by_phase(df)
    if phase_stats.empty or len(phase_stats) < 2:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('按进化阶段分析 (Evolution Phase Analysis)', fontsize=14, fontweight='bold')
    
    phases = phase_stats['evolution_phase']
    colors = {'original': '#3498db', 'mutation': '#e74c3c', 'crossover': '#2ecc71', 'N/A': '#95a5a6'}
    bar_colors = [colors.get(p, '#95a5a6') for p in phases]
    
    # 1. 因子数量饼图
    ax1 = axes[0]
    ax1.pie(phase_stats['count'], labels=phases, autopct='%1.1f%%', colors=bar_colors, startangle=90)
    ax1.set_title('因子数量分布')
    
    # 2. RankIC 对比
    ax2 = axes[1]
    if 'RankIC_mean' in phase_stats.columns:
        bars = ax2.bar(phases, phase_stats['RankIC_mean'], color=bar_colors, alpha=0.8)
        if 'RankIC_std' in phase_stats.columns:
            ax2.errorbar(phases, phase_stats['RankIC_mean'], yerr=phase_stats['RankIC_std'], 
                        fmt='none', color='black', capsize=5)
        ax2.set_ylabel('RankIC Mean')
        ax2.set_title('RankIC 均值对比')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. 年化收益对比
    ax3 = axes[2]
    if 'annualized_return_mean' in phase_stats.columns:
        ax3.bar(phases, phase_stats['annualized_return_mean'] * 100, color=bar_colors, alpha=0.8)
        ax3.set_ylabel('Annualized Return (%)')
        ax3.set_title('年化收益率均值对比')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已生成: phase_analysis.png")


def plot_quality_analysis(df: pd.DataFrame, output_dir: Path):
    """绘制按质量分析的图表。"""
    if not HAS_MATPLOTLIB or df.empty:
        return
    
    # 确保字体已正确设置
    try:
        setup_chinese_font()
    except:
        pass
    
    quality_stats = analyze_by_quality(df)
    if quality_stats.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('按因子质量分析 (Quality Analysis)', fontsize=14, fontweight='bold')
    
    qualities = quality_stats['quality']
    colors = {'high_quality': '#27ae60', 'Poor': '#e74c3c', 'N/A': '#95a5a6'}
    bar_colors = [colors.get(q, '#95a5a6') for q in qualities]
    
    # 1. 质量分布饼图
    ax1 = axes[0]
    ax1.pie(quality_stats['count'], labels=qualities, autopct='%1.1f%%', colors=bar_colors, startangle=90)
    ax1.set_title('因子质量分布')
    
    # 2. 不同质量的指标对比
    ax2 = axes[1]
    x = np.arange(len(qualities))
    width = 0.35
    
    if 'RankIC_mean' in quality_stats.columns and 'information_ratio_mean' in quality_stats.columns:
        # 归一化以便在同一图上显示
        rankic = quality_stats['RankIC_mean'].fillna(0) * 100  # 放大100倍
        ir = quality_stats['information_ratio_mean'].fillna(0)
        
        bars1 = ax2.bar(x - width/2, rankic, width, label='RankIC × 100', color='#3498db', alpha=0.8)
        bars2 = ax2.bar(x + width/2, ir, width, label='Information Ratio', color='#e74c3c', alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(qualities)
        ax2.set_ylabel('Value')
        ax2.set_title('不同质量因子的指标对比')
        ax2.legend()
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已生成: quality_analysis.png")


def plot_metrics_distribution(df: pd.DataFrame, output_dir: Path):
    """绘制关键指标的分布图。"""
    if not HAS_MATPLOTLIB or df.empty:
        return
    
    # 确保字体已正确设置
    try:
        setup_chinese_font()
    except:
        pass
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('关键指标分布 (Key Metrics Distribution)', fontsize=14, fontweight='bold')
    
    metrics_config = [
        ('RankIC', 'RankIC 分布', '#3498db'),
        ('annualized_return', '年化收益率分布', '#27ae60'),
        ('max_drawdown', '最大回撤分布', '#e74c3c'),
        ('information_ratio', '信息比率分布', '#9b59b6'),
    ]
    
    for ax, (metric, title, color) in zip(axes.flat, metrics_config):
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                if HAS_SEABORN:
                    sns.histplot(values, kde=True, ax=ax, color=color, alpha=0.7)
                else:
                    ax.hist(values, bins=20, color=color, alpha=0.7, edgecolor='white')
                
                ax.axvline(values.mean(), color='red', linestyle='--', label=f'Mean: {values.mean():.4f}')
                ax.axvline(values.median(), color='orange', linestyle=':', label=f'Median: {values.median():.4f}')
                ax.set_xlabel(metric)
                ax.set_ylabel('Frequency')
                ax.set_title(title)
                ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已生成: metrics_distribution.png")


def plot_direction_analysis(df: pd.DataFrame, output_dir: Path, top_n: int = 10):
    """绘制按初始方向分析的图表。"""
    if not HAS_MATPLOTLIB or df.empty:
        return
    
    # 确保字体已正确设置
    try:
        setup_chinese_font()
    except:
        pass
    
    direction_stats = analyze_by_direction(df)
    if direction_stats.empty:
        return
    
    # 取 Top N
    top_directions = direction_stats.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f'Top {top_n} 初始方向按 RankIC 均值排名', fontsize=14, fontweight='bold')
    
    # 截短方向名称用于显示
    labels = [d[:40] + '...' if len(d) > 40 else d for d in top_directions['initial_direction']]
    
    if 'RankIC_mean' in top_directions.columns:
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, top_directions['RankIC_mean'], color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('RankIC Mean')
        ax.set_title('不同初始方向的 RankIC 均值')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # 添加数值标签
        for i, (v, c) in enumerate(zip(top_directions['RankIC_mean'], top_directions['count'])):
            ax.text(v + 0.001, i, f'{v:.4f} (n={c})', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'direction_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已生成: direction_analysis.png")


def plot_metrics_correlation(df: pd.DataFrame, output_dir: Path):
    """绘制指标相关性热力图。"""
    if not HAS_MATPLOTLIB or df.empty:
        return
    
    # 确保字体已正确设置
    try:
        setup_chinese_font()
    except:
        pass
    
    metrics_cols = [m for m in ALL_METRICS if m in df.columns]
    if len(metrics_cols) < 2:
        return
    
    corr_matrix = df[metrics_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
    else:
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(len(metrics_cols)))
        ax.set_yticks(range(len(metrics_cols)))
        ax.set_xticklabels(metrics_cols, rotation=45, ha='right')
        ax.set_yticklabels(metrics_cols)
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 添加数值
        for i in range(len(metrics_cols)):
            for j in range(len(metrics_cols)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    ax.set_title('回测指标相关性矩阵', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已生成: metrics_correlation.png")


def plot_cumulative_round_performance(df: pd.DataFrame, output_dir: Path):
    """绘制累计轮次性能趋势。"""
    if not HAS_MATPLOTLIB or df.empty:
        return
    
    # 确保字体已正确设置
    try:
        setup_chinese_font()
    except:
        pass
    
    round_stats = analyze_by_round(df)
    if round_stats.empty or len(round_stats) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rounds = round_stats['round_number'].values
    cumulative_count = round_stats['count'].cumsum()
    
    # 双Y轴
    ax2 = ax.twinx()
    
    # 累计因子数量
    ax.fill_between(rounds, 0, cumulative_count, alpha=0.3, color='steelblue', label='累计因子数')
    ax.plot(rounds, cumulative_count, 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel('Round Number')
    ax.set_ylabel('累计因子数量', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    
    # RankIC 累计均值
    if 'RankIC_mean' in round_stats.columns:
        cumulative_rankic = round_stats['RankIC_mean'].expanding().mean()
        ax2.plot(rounds, cumulative_rankic, 's--', color='green', linewidth=2, label='累计 RankIC 均值')
        ax2.set_ylabel('累计 RankIC 均值', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    
    ax.set_title('累计轮次性能趋势', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已生成: cumulative_performance.png")


# ============================================================
# 报告生成
# ============================================================
def generate_report(metadata: dict, df: pd.DataFrame, output_dir: Path):
    """生成完整的分析报告。"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("因子库分析报告 (Factor Library Analysis Report)")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. 元数据
    report_lines.append("-" * 40)
    report_lines.append("1. 因子库元数据")
    report_lines.append("-" * 40)
    for k, v in metadata.items():
        report_lines.append(f"  {k}: {v}")
    report_lines.append("")
    
    # 2. 整体统计
    report_lines.append("-" * 40)
    report_lines.append("2. 整体统计")
    report_lines.append("-" * 40)
    overall = get_overall_statistics(df)
    report_lines.append(f"  因子总数: {overall['total_factors']}")
    report_lines.append(f"  唯一轮次数: {overall['unique_rounds']}")
    report_lines.append(f"  唯一进化阶段数: {overall['unique_phases']}")
    report_lines.append(f"  唯一初始方向数: {overall['unique_directions']}")
    report_lines.append("")
    report_lines.append("  关键指标汇总:")
    for m in KEY_METRICS:
        if f'{m}_overall_mean' in overall:
            report_lines.append(f"    {m}:")
            report_lines.append(f"      均值: {overall[f'{m}_overall_mean']:.6f}")
            report_lines.append(f"      标准差: {overall[f'{m}_overall_std']:.6f}")
            report_lines.append(f"      最小值: {overall[f'{m}_overall_min']:.6f}")
            report_lines.append(f"      最大值: {overall[f'{m}_overall_max']:.6f}")
    report_lines.append("")
    
    if 'quality_distribution' in overall:
        report_lines.append("  质量分布:")
        for q, c in overall['quality_distribution'].items():
            report_lines.append(f"    {q}: {c} ({c/overall['total_factors']*100:.1f}%)")
    report_lines.append("")
    
    # 3. 按轮次分析
    report_lines.append("-" * 40)
    report_lines.append("3. 按轮次分析 (Round Number)")
    report_lines.append("-" * 40)
    round_stats = analyze_by_round(df)
    if not round_stats.empty:
        report_lines.append(round_stats.to_string(index=False))
    report_lines.append("")
    
    # 4. 按进化阶段分析
    report_lines.append("-" * 40)
    report_lines.append("4. 按进化阶段分析 (Evolution Phase)")
    report_lines.append("-" * 40)
    phase_stats = analyze_by_phase(df)
    if not phase_stats.empty:
        report_lines.append(phase_stats.to_string(index=False))
    report_lines.append("")
    
    # 5. 按质量分析
    report_lines.append("-" * 40)
    report_lines.append("5. 按质量分析 (Quality)")
    report_lines.append("-" * 40)
    quality_stats = analyze_by_quality(df)
    if not quality_stats.empty:
        report_lines.append(quality_stats.to_string(index=False))
    report_lines.append("")
    
    # 6. 按初始方向分析 (Top 10)
    report_lines.append("-" * 40)
    report_lines.append("6. 按初始方向分析 (Top 10 by RankIC)")
    report_lines.append("-" * 40)
    direction_stats = analyze_by_direction(df)
    if not direction_stats.empty:
        report_lines.append(direction_stats.head(10).to_string(index=False))
    report_lines.append("")
    
    # 7. Top 10 因子
    report_lines.append("-" * 40)
    report_lines.append("7. Top 10 因子 (by RankIC)")
    report_lines.append("-" * 40)
    top_factors = get_top_factors(df, 'RankIC', 10)
    if not top_factors.empty:
        report_lines.append(top_factors.to_string(index=False))
    report_lines.append("")
    
    # 8. 图表说明
    report_lines.append("-" * 40)
    report_lines.append("8. 生成的可视化图表")
    report_lines.append("-" * 40)
    report_lines.append("  - round_analysis.png: 按轮次分析图表")
    report_lines.append("  - phase_analysis.png: 按进化阶段分析图表")
    report_lines.append("  - quality_analysis.png: 按质量分析图表")
    report_lines.append("  - metrics_distribution.png: 关键指标分布图")
    report_lines.append("  - direction_analysis.png: 按初始方向分析图表")
    report_lines.append("  - metrics_correlation.png: 指标相关性热力图")
    report_lines.append("  - cumulative_performance.png: 累计轮次性能趋势")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("报告结束")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # 保存报告
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text


# ============================================================
# 主函数
# ============================================================
def analyze_factor_library(
    factor_library_path: str,
    output_dir: Optional[str] = None
):
    """
    分析因子库文件。
    
    Args:
        factor_library_path: 因子库 JSON 文件路径
        output_dir: 输出目录，默认为工具所在目录下的 analysis_<timestamp> 子文件夹
    """
    factor_path = Path(factor_library_path)
    if not factor_path.exists():
        print(f"错误: 文件不存在 - {factor_library_path}")
        return
    
    # 设置输出目录
    # 获取工具所在目录（factor_analyzer.py 的父目录）
    tool_dir = Path(__file__).parent
    
    if output_dir:
        # 如果用户指定了输出目录，在该目录下创建带时间戳的子文件夹
        base_output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = base_output_dir / f"analysis_{timestamp}"
    else:
        # 默认输出到工具所在目录，每次运行创建带时间戳的子文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = tool_dir / f"analysis_{timestamp}"
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在分析因子库: {factor_path}")
    print(f"输出目录: {out_path}")
    print()
    
    # 加载数据
    metadata, df = load_factor_library(factor_library_path)
    print(f"已加载 {len(df)} 个因子")
    print()
    
    # 生成图表
    print("生成可视化图表...")
    plot_round_analysis(df, out_path)
    plot_phase_analysis(df, out_path)
    plot_quality_analysis(df, out_path)
    plot_metrics_distribution(df, out_path)
    plot_direction_analysis(df, out_path)
    plot_metrics_correlation(df, out_path)
    plot_cumulative_round_performance(df, out_path)
    print()
    
    # 生成报告
    print("生成分析报告...")
    report = generate_report(metadata, df, out_path)
    print(f"  ✓ 已生成: analysis_report.txt")
    print()
    
    # 保存详细数据为 CSV
    df.to_csv(out_path / "factors_detail.csv", index=False, encoding='utf-8-sig')
    print(f"  ✓ 已生成: factors_detail.csv")
    
    # 打印报告摘要
    print()
    print("=" * 60)
    print("分析完成！报告摘要:")
    print("=" * 60)
    
    overall = get_overall_statistics(df)
    print(f"因子总数: {overall['total_factors']}")
    print(f"唯一轮次数: {overall['unique_rounds']}")
    
    for m in ['RankIC', 'annualized_return', 'information_ratio']:
        if f'{m}_overall_mean' in overall:
            print(f"{m} 均值: {overall[f'{m}_overall_mean']:.6f}")
    
    print()
    print(f"详细报告已保存到: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='因子库分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 默认输出到工具目录下的 analysis_<timestamp> 文件夹
    python factor_analyzer.py all_factors_library.json
    
    # 指定输出目录（会在该目录下创建 analysis_<timestamp> 子文件夹）
    python factor_analyzer.py all_factors_library.json --output_dir /path/to/output
    
注意:
    - 默认输出目录: tools/factor_analyzer/analysis_<timestamp>/
    - 每次运行都会创建新的带时间戳的文件夹，不会覆盖之前的分析结果
        """
    )
    parser.add_argument('factor_library', help='因子库 JSON 文件路径')
    parser.add_argument('--output_dir', '-o', help='输出目录路径（会在该目录下创建带时间戳的子文件夹）')
    
    args = parser.parse_args()
    
    analyze_factor_library(args.factor_library, args.output_dir)


if __name__ == '__main__':
    main()
