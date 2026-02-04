#!/usr/bin/env python3
"""
绘制 Round 回测结果折线图

用法:
    python tools/plot_round_backtest_results.py --input backtest_v2_results/round_claude_20260120_030739/summary.json --output results.png
    python tools/plot_round_backtest_results.py --data-inline  # 使用内置数据测试
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path


def load_results_from_json(json_path: str) -> list:
    """从 JSON 文件加载回测结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_round_results(results: list, 
                       output_path: str, 
                       title: str = "Factor Mining - Round Backtest Results",
                       baseline: dict = None):
    """
    绘制 Round 回测结果折线图
    
    Args:
        results: 回测结果列表，每个元素包含 round, Rank_IC, annualized_return, information_ratio, max_drawdown
        output_path: 输出图片路径
        title: 图表标题
        baseline: 基准数据字典 (可选)
    """
    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 默认基准数据 (alpha158_20)
    if baseline is None:
        baseline = {
            'Rank_IC': 0.018385,
            'Rank_ICIR': 0.1177,
            'annualized_return': 0.0463,
            'information_ratio': 0.5044,
            'max_drawdown': -0.2219
        }
    
    # 提取数据
    rounds = [r['round'] for r in results]
    rank_ic = [r.get('Rank_IC', 0) or 0 for r in results]
    rank_icir = [r.get('Rank_ICIR', 0) or 0 for r in results]
    ann_ret = [(r.get('annualized_return', 0) or 0) * 100 for r in results]
    ir = [r.get('information_ratio', 0) or 0 for r in results]
    mdd = [abs(r.get('max_drawdown', 0) or 0) * 100 for r in results]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title}\n(vs Alpha158_20 Baseline)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    marker_style = dict(marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    # 1. Rank IC
    ax1 = axes[0, 0]
    ax1.plot(rounds, rank_ic, color='#2E86AB', linewidth=2.5, label='Mined Factors', **marker_style, markeredgecolor='#2E86AB')
    ax1.axhline(y=baseline['Rank_IC'], color='#E74C3C', linestyle='--', linewidth=2, label=f'Baseline: {baseline["Rank_IC"]:.4f}')
    ax1.fill_between(rounds, baseline['Rank_IC'], rank_ic, alpha=0.2, color='#2E86AB')
    ax1.set_ylabel('Rank IC', fontsize=11, fontweight='bold')
    ax1.set_title('Rank IC by Round', fontsize=12, fontweight='bold')
    ax1.set_xticks(rounds)
    ax1.set_xlabel('Round', fontsize=10)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0, max(rank_ic) * 1.15 if rank_ic else 0.1)
    # 标注最大值
    if rank_ic:
        max_idx = rank_ic.index(max(rank_ic))
        ax1.annotate(f'Max: {max(rank_ic):.4f}', xy=(rounds[max_idx], max(rank_ic)), 
                     xytext=(rounds[max_idx]+0.5, max(rank_ic)+0.003),
                     fontsize=9, color='#2E86AB', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1))
    
    # 2. Annualized Return
    ax2 = axes[0, 1]
    ax2.plot(rounds, ann_ret, color='#27AE60', linewidth=2.5, label='Mined Factors', **marker_style, markeredgecolor='#27AE60')
    ax2.axhline(y=baseline['annualized_return']*100, color='#E74C3C', linestyle='--', linewidth=2, 
                label=f'Baseline: {baseline["annualized_return"]*100:.1f}%')
    ax2.fill_between(rounds, baseline['annualized_return']*100, ann_ret, alpha=0.2, color='#27AE60')
    ax2.set_ylabel('Annualized Return (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Annualized Return by Round', fontsize=12, fontweight='bold')
    ax2.set_xticks(rounds)
    ax2.set_xlabel('Round', fontsize=10)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(0, max(ann_ret) * 1.15 if ann_ret else 20)
    if ann_ret:
        max_idx = ann_ret.index(max(ann_ret))
        ax2.annotate(f'Max: {max(ann_ret):.1f}%', xy=(rounds[max_idx], max(ann_ret)), 
                     xytext=(rounds[max_idx]-1.5, max(ann_ret)+1),
                     fontsize=9, color='#27AE60', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1))
    
    # 3. Information Ratio
    ax3 = axes[1, 0]
    ax3.plot(rounds, ir, color='#9B59B6', linewidth=2.5, label='Mined Factors', **marker_style, markeredgecolor='#9B59B6')
    ax3.axhline(y=baseline['information_ratio'], color='#E74C3C', linestyle='--', linewidth=2, 
                label=f'Baseline: {baseline["information_ratio"]:.2f}')
    ax3.fill_between(rounds, baseline['information_ratio'], ir, alpha=0.2, color='#9B59B6')
    ax3.set_ylabel('Information Ratio', fontsize=11, fontweight='bold')
    ax3.set_title('Information Ratio by Round', fontsize=12, fontweight='bold')
    ax3.set_xticks(rounds)
    ax3.set_xlabel('Round', fontsize=10)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_ylim(0, max(ir) * 1.15 if ir else 2)
    if ir:
        max_idx = ir.index(max(ir))
        ax3.annotate(f'Max: {max(ir):.2f}', xy=(rounds[max_idx], max(ir)), 
                     xytext=(rounds[max_idx]+0.5, max(ir)+0.08),
                     fontsize=9, color='#9B59B6', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1))
    
    # 4. Max Drawdown (lower is better)
    ax4 = axes[1, 1]
    ax4.plot(rounds, mdd, color='#E67E22', linewidth=2.5, label='Mined Factors', **marker_style, markeredgecolor='#E67E22')
    ax4.axhline(y=abs(baseline['max_drawdown'])*100, color='#E74C3C', linestyle='--', linewidth=2, 
                label=f'Baseline: {abs(baseline["max_drawdown"])*100:.1f}%')
    ax4.fill_between(rounds, abs(baseline['max_drawdown'])*100, mdd, alpha=0.2, color='#E67E22')
    ax4.set_ylabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Max Drawdown by Round (lower is better)', fontsize=12, fontweight='bold')
    ax4.set_xticks(rounds)
    ax4.set_xlabel('Round', fontsize=10)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_ylim(0, max(mdd) * 1.25 if mdd else 25)
    if mdd:
        min_idx = mdd.index(min(mdd))
        ax4.annotate(f'Min: {min(mdd):.1f}%', xy=(rounds[min_idx], min(mdd)), 
                     xytext=(rounds[min_idx]+0.5, min(mdd)+2),
                     fontsize=9, color='#E67E22', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#E67E22', lw=1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 折线图已保存到: {output_path}")
    
    plt.close()


def get_sample_data():
    """获取示例数据 (Claude Round 0-10)"""
    return [
        {'round': 0, 'num_factors': 30, 'Rank_IC': 0.067353, 'Rank_ICIR': 0.4970, 'annualized_return': 0.1134, 'information_ratio': 1.5571, 'max_drawdown': -0.0856},
        {'round': 1, 'num_factors': 30, 'Rank_IC': 0.063938, 'Rank_ICIR': 0.3317, 'annualized_return': 0.0896, 'information_ratio': 0.9333, 'max_drawdown': -0.1944},
        {'round': 2, 'num_factors': 30, 'Rank_IC': 0.066001, 'Rank_ICIR': 0.3573, 'annualized_return': 0.0742, 'information_ratio': 0.7850, 'max_drawdown': -0.1680},
        {'round': 3, 'num_factors': 27, 'Rank_IC': 0.074280, 'Rank_ICIR': 0.4079, 'annualized_return': 0.1174, 'information_ratio': 1.2861, 'max_drawdown': -0.1468},
        {'round': 4, 'num_factors': 30, 'Rank_IC': 0.063128, 'Rank_ICIR': 0.3169, 'annualized_return': 0.0999, 'information_ratio': 1.0024, 'max_drawdown': -0.1639},
        {'round': 5, 'num_factors': 30, 'Rank_IC': 0.071348, 'Rank_ICIR': 0.4357, 'annualized_return': 0.1297, 'information_ratio': 1.6803, 'max_drawdown': -0.1168},
        {'round': 6, 'num_factors': 27, 'Rank_IC': 0.061487, 'Rank_ICIR': 0.3026, 'annualized_return': 0.1146, 'information_ratio': 1.1200, 'max_drawdown': -0.1502},
        {'round': 7, 'num_factors': 27, 'Rank_IC': 0.073417, 'Rank_ICIR': 0.4119, 'annualized_return': 0.1331, 'information_ratio': 1.4599, 'max_drawdown': -0.1152},
        {'round': 8, 'num_factors': 30, 'Rank_IC': 0.074393, 'Rank_ICIR': 0.3961, 'annualized_return': 0.1461, 'information_ratio': 1.5784, 'max_drawdown': -0.1228},
        {'round': 9, 'num_factors': 27, 'Rank_IC': 0.068506, 'Rank_ICIR': 0.4574, 'annualized_return': 0.1056, 'information_ratio': 1.4590, 'max_drawdown': -0.1289},
        {'round': 10, 'num_factors': 30, 'Rank_IC': 0.068531, 'Rank_ICIR': 0.3618, 'annualized_return': 0.1091, 'information_ratio': 1.2300, 'max_drawdown': -0.0981},
    ]


def main():
    parser = argparse.ArgumentParser(description='绘制 Round 回测结果折线图')
    parser.add_argument('--input', '-i', type=str, help='输入 JSON 文件路径 (summary.json 格式)')
    parser.add_argument('--output', '-o', type=str, default='round_backtest_results.png', help='输出图片路径')
    parser.add_argument('--title', '-t', type=str, default='Factor Mining - Round Backtest Results', help='图表标题')
    parser.add_argument('--data-inline', action='store_true', help='使用内置示例数据')
    
    args = parser.parse_args()
    
    if args.data_inline:
        results = get_sample_data()
        print("使用内置示例数据 (Claude Round 0-10)")
    elif args.input:
        results = load_results_from_json(args.input)
        print(f"从 {args.input} 加载了 {len(results)} 条记录")
    else:
        print("错误: 请指定 --input 或 --data-inline")
        return
    
    plot_round_results(results, args.output, title=args.title)


if __name__ == '__main__':
    main()

