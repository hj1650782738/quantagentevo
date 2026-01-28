#!/usr/bin/env python3
"""
按 RankIC 排序提取 Top-K 因子

从 all_factors_library JSON 中按 Rank IC 降序排列，
依次提取 top500, top550, top600, top650, top700 的因子，
输出到 factor_library/hj/ 目录下。
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def extract_topk_factors(
    input_json: str,
    output_dir: str,
    topk_list: list,
    rank_ic_key: str = "Rank IC"
):
    """
    按 RankIC 提取 Top-K 因子
    
    Args:
        input_json: 输入的因子库JSON路径
        output_dir: 输出目录
        topk_list: 要提取的topk列表，如 [500, 550, 600, 650, 700]
        rank_ic_key: backtest_results中RankIC的键名
    """
    print(f"[INFO] 读取因子库: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    factors = data.get('factors', {})
    print(f"[INFO] 总因子数: {len(factors)}")
    
    # 提取有RankIC的因子
    factor_list = []
    for fid, info in factors.items():
        bt = info.get('backtest_results', {})
        rank_ic = bt.get(rank_ic_key)
        if rank_ic is not None and not (isinstance(rank_ic, float) and (rank_ic != rank_ic)):  # 排除NaN
            factor_list.append({
                'factor_id': fid,
                'rank_ic': float(rank_ic),
                'info': info
            })
    
    print(f"[INFO] 有效因子数 (有RankIC): {len(factor_list)}")
    
    # 按RankIC降序排序
    factor_list.sort(key=lambda x: x['rank_ic'], reverse=True)
    
    # 打印前10个因子的RankIC
    print(f"\n[INFO] Top 10 因子 (RankIC):")
    for i, f in enumerate(factor_list[:10]):
        print(f"  {i+1}. {f['info'].get('factor_name', f['factor_id'][:16])}: {f['rank_ic']:.6f}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件的基础名
    input_name = Path(input_json).stem
    
    # 提取各个topk
    output_files = []
    for topk in topk_list:
        if topk > len(factor_list):
            print(f"[WARN] topk={topk} 超过有效因子数 {len(factor_list)}，使用全部因子")
            topk = len(factor_list)
        
        selected = factor_list[:topk]
        
        # 构建输出JSON
        output_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source_file": str(input_json),
                "selection_method": f"Top {topk} by {rank_ic_key} (descending)",
                "total_factors": topk,
                "rank_ic_range": {
                    "max": selected[0]['rank_ic'] if selected else None,
                    "min": selected[-1]['rank_ic'] if selected else None,
                }
            },
            "factors": {}
        }
        
        for f in selected:
            output_data["factors"][f['factor_id']] = f['info']
        
        # 输出文件名
        output_file = output_path / f"RANKIC_top{topk}_{input_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] 输出 top{topk} 因子 -> {output_file}")
        print(f"     RankIC 范围: [{selected[-1]['rank_ic']:.6f}, {selected[0]['rank_ic']:.6f}]")
        output_files.append(str(output_file))
    
    return output_files


def main():
    parser = argparse.ArgumentParser(description='按 RankIC 提取 Top-K 因子')
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入的因子库JSON文件'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='factor_library/hj',
        help='输出目录 (默认: factor_library/hj)'
    )
    parser.add_argument(
        '-k', '--topk',
        type=int,
        nargs='+',
        default=[500, 550, 600, 650, 700],
        help='要提取的topk列表 (默认: 500 550 600 650 700)'
    )
    parser.add_argument(
        '--rank-ic-key',
        type=str,
        default='Rank IC',
        help='backtest_results中RankIC的键名 (默认: "Rank IC")'
    )
    
    args = parser.parse_args()
    
    output_files = extract_topk_factors(
        input_json=args.input,
        output_dir=args.output_dir,
        topk_list=args.topk,
        rank_ic_key=args.rank_ic_key
    )
    
    print(f"\n[DONE] 共生成 {len(output_files)} 个因子库文件")
    for f in output_files:
        print(f"  - {f}")


if __name__ == '__main__':
    main()

