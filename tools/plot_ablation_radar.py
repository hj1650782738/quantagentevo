#!/usr/bin/env python3
"""
绘制消融实验雷达图

用法:
    python tools/plot_ablation_radar.py --output ablation_radar.png
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from math import pi


def plot_ablation_radar(output_path: str, title: str = "Ablation Study - Radar Chart"):
    """
    绘制消融实验雷达图
    
    Args:
        output_path: 输出图片路径
        title: 图表标题
    """
    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 消融实验数据
    data = {
        'w/ All': {
            'IC': 0.1493, 'ICIR': 0.9263, 'Rank IC': 0.1458, 
            'Rank ICIR': 0.9093, 'ARR': 0.2899, 'MDD': -0.0942
        },
        'w/o Planning': {
            'IC': 0.1488, 'ICIR': 0.8782, 'Rank IC': 0.1452, 
            'Rank ICIR': 0.8570, 'ARR': 0.2121, 'MDD': -0.1215
        },
        'w/o Mutation': {
            'IC': 0.1201, 'ICIR': 0.7704, 'Rank IC': 0.1174, 
            'Rank ICIR': 0.7555, 'ARR': 0.1918, 'MDD': -0.0985
        },
        'w/o Crossover': {
            'IC': 0.1423, 'ICIR': 0.8673, 'Rank IC': 0.1381, 
            'Rank ICIR': 0.8412, 'ARR': 0.2617, 'MDD': -0.1063
        }
    }
    
    # 维度标签
    categories = ['IC', 'Rank IC', 'ICIR', 'Rank ICIR', 'ARR', '1-|MDD|']
    num_vars = len(categories)
    
    # 配色方案（参考 plot_round_backtest_results.py）
    colors = {
        'w/ All': '#2E86AB',       # 蓝色
        'w/o Planning': '#27AE60',  # 绿色
        'w/o Mutation': '#9B59B6',  # 紫色
        'w/o Crossover': '#E67E22'  # 橙色
    }
    
    # 数据归一化 - 使用自定义范围来控制各指标差异显示程度
    # 对于MDD，转换为 1-|MDD| 使其越大越好
    raw_values = {
        'IC': [d['IC'] for d in data.values()],
        'ICIR': [d['ICIR'] for d in data.values()],
        'Rank IC': [d['Rank IC'] for d in data.values()],
        'Rank ICIR': [d['Rank ICIR'] for d in data.values()],
        'ARR': [d['ARR'] for d in data.values()],
        'MDD': [1 - abs(d['MDD']) for d in data.values()]
    }
    
    # 计算每个维度的min和max
    data_min = {k: min(v) for k, v in raw_values.items()}
    data_max = {k: max(v) for k, v in raw_values.items()}
    
    # 自定义归一化范围 - 调整各指标的差异显示程度
    # 设置 (ref_min, ref_max)：范围越窄，差异显示越大；范围越宽，差异显示越小
    # IC, Rank IC: 数据范围约0.029，设置更窄的参考范围来放大差异
    # MDD: 设置更宽的参考范围来缩小差异
    custom_ranges = {
        'IC': (0.10, 0.16),          # 数据实际 0.12~0.15，范围0.06，放大差异
        'Rank IC': (0.10, 0.16),     # 数据实际 0.12~0.15，范围0.06，放大差异
        'ICIR': (0.70, 1.00),        # 数据实际 0.77~0.93，适中
        'Rank ICIR': (0.70, 1.00),   # 数据实际 0.76~0.91，适中
        'ARR': (0.15, 0.35),         # 数据实际 0.19~0.29，适中
        'MDD': (0.80, 0.95),         # 数据实际 0.88~0.91，扩大范围缩小差异
    }
    
    # 归一化函数：将原始值映射到 [0.55, 1.0] 区间
    # 这样圆更贴近六边形边缘，同时保持差异可见
    def normalize(val, ref_min, ref_max):
        if ref_max == ref_min:
            return 0.75
        # 先clip到参考范围内
        val_clipped = max(ref_min, min(ref_max, val))
        # 映射到 [0.55, 1.0]
        return 0.55 + 0.45 * (val_clipped - ref_min) / (ref_max - ref_min)
    
    # 归一化数据
    normalized_data = {}
    for name, metrics in data.items():
        normalized_data[name] = [
            normalize(metrics['IC'], *custom_ranges['IC']),
            normalize(metrics['Rank IC'], *custom_ranges['Rank IC']),
            normalize(metrics['ICIR'], *custom_ranges['ICIR']),
            normalize(metrics['Rank ICIR'], *custom_ranges['Rank ICIR']),
            normalize(metrics['ARR'], *custom_ranges['ARR']),
            normalize(1 - abs(metrics['MDD']), *custom_ranges['MDD'])
        ]
    
    # 计算角度
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # 闭合
    
    # 准备原始数值用于标注
    raw_data_for_labels = {}
    for name, metrics in data.items():
        raw_data_for_labels[name] = [
            metrics['IC'],
            metrics['Rank IC'],
            metrics['ICIR'],
            metrics['Rank ICIR'],
            metrics['ARR'],
            abs(metrics['MDD'])  # MDD显示绝对值
        ]
    
    # 创建图形（不绘制标题）
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # 设置起始角度和方向
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # 绘制各条线并标注数值
    # 定义每个配置的标注偏移量，避免重叠
    label_offsets = {
        'w/ All': 0.06,
        'w/o Planning': 0.04,
        'w/o Mutation': 0.02,
        'w/o Crossover': 0.00
    }
    
    for idx, (name, values) in enumerate(normalized_data.items()):
        values_plot = values + values[:1]  # 闭合
        ax.plot(angles, values_plot, 'o-', linewidth=2.5, label=name, 
                color=colors[name], markersize=8)
        ax.fill(angles, values_plot, alpha=0.15, color=colors[name])
        
        # 在每个数据点上标注原始数值
        raw_vals = raw_data_for_labels[name]
        for i, (angle, norm_val, raw_val) in enumerate(zip(angles[:-1], values, raw_vals)):
            # 根据角度调整标注位置
            offset = label_offsets[name]
            r_offset = norm_val + offset
            
            # 格式化数值：MDD显示为负数
            if i == 5:  # MDD
                label_text = f"-{raw_val:.2f}"
            elif i in [0, 1]:  # IC, Rank IC
                label_text = f"{raw_val:.4f}"
            elif i in [2, 3]:  # ICIR, Rank ICIR
                label_text = f"{raw_val:.2f}"
            else:  # ARR
                label_text = f"{raw_val:.2f}"
            
            ax.annotate(label_text, xy=(angle, r_offset), 
                       fontsize=8, color=colors[name], fontweight='bold',
                       ha='center', va='bottom')
    
    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
    
    # 设置径向刻度（隐藏数值刻度，因为已经标注了原始值）
    # 调整范围让图更紧凑，数据都在 [0.55, 1.0] 区间
    ax.set_ylim(0.4, 1.15)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['', '', '', '', '', ''], fontsize=9, color='gray')
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=11, 
              frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 雷达图已保存到: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制消融实验雷达图')
    parser.add_argument('--output', '-o', type=str, default='ablation_radar.png', help='输出图片路径')
    parser.add_argument('--title', '-t', type=str, default='Ablation Study - Radar Chart', help='图表标题')
    
    args = parser.parse_args()
    
    plot_ablation_radar(args.output, title=args.title)


if __name__ == '__main__':
    main()

