import json
import argparse
from pathlib import Path

def filter_factors(input_path, output_path, top_k=150):
    print(f"Loading factors from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    factors = data.get('factors', {})
    print(f"Total factors found: {len(factors)}")
    
    # Extract factors with their Rank IC
    factor_list = []
    for fid, f_data in factors.items():
        rank_ic = -float('inf')
        if 'backtest_results' in f_data and f_data['backtest_results']:
            # Try different key variations just in case
            if 'Rank IC' in f_data['backtest_results']:
                rank_ic = f_data['backtest_results']['Rank IC']
            elif 'rank_ic' in f_data['backtest_results']:
                rank_ic = f_data['backtest_results']['rank_ic']
        
        factor_list.append({
            'id': fid,
            'data': f_data,
            'rank_ic': rank_ic
        })
    
    # Sort by Rank IC descending
    factor_list.sort(key=lambda x: x['rank_ic'], reverse=True)
    
    # Select top K
    top_factors = factor_list[:top_k]
    print(f"Selected top {len(top_factors)} factors based on Rank IC")
    
    if top_factors:
        print(f"Top 1 Rank IC: {top_factors[0]['rank_ic']}")
        print(f"Top {len(top_factors)} Rank IC: {top_factors[-1]['rank_ic']}")
    
    # Construct new data
    new_factors = {item['id']: item['data'] for item in top_factors}
    
    new_data = {
        'metadata': data.get('metadata', {}),
        'factors': new_factors
    }
    
    # Update metadata if possible
    if 'metadata' in new_data:
        new_data['metadata']['total_factors'] = len(new_factors)
        new_data['metadata']['description'] = f"Top {len(new_factors)} factors filtered by Rank IC from {Path(input_path).name}"
        
    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter top factors by Rank IC')
    parser.add_argument('input', help='Input JSON file path')
    parser.add_argument('output', help='Output JSON file path')
    parser.add_argument('--top-k', type=int, default=150, help='Number of top factors to keep')
    
    args = parser.parse_args()
    
    filter_factors(args.input, args.output, args.top_k)
