#!/usr/bin/env python3
"""
查看Alpha158因子库和20个精选因子
"""

import json
from qlib.contrib.data.loader import Alpha158DL

# Alpha158默认配置
default_config = {
    "kbar": {},
    "price": {
        "windows": [0],
        "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
    },
    "rolling": {},
}

# 20个精选因子
selected_factors = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
    "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
    "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
]

print("=" * 70)
print("Alpha158 因子库分析")
print("=" * 70)

# 获取所有因子
fields, names = Alpha158DL.get_feature_config(default_config)

print(f"\n📊 Alpha158 总因子数量: {len(names)}")
print(f"\n所有因子列表:")
for i, (field, name) in enumerate(zip(fields, names), 1):
    print(f"{i:3d}. {name:15s} = {field}")

# 创建因子字典
factor_dict = dict(zip(names, fields))

# 检查精选因子
print("\n" + "=" * 70)
print("20个精选因子检查")
print("=" * 70)

selected_info = []
missing_factors = []

for factor in selected_factors:
    if factor in factor_dict:
        selected_info.append({
            "name": factor,
            "expression": factor_dict[factor],
            "status": "✅ 存在"
        })
    else:
        missing_factors.append(factor)
        selected_info.append({
            "name": factor,
            "expression": None,
            "status": "❌ 不存在"
        })

print(f"\n找到 {len(selected_factors) - len(missing_factors)}/{len(selected_factors)} 个精选因子\n")

for item in selected_info:
    if item["expression"]:
        print(f"✅ {item['name']:10s} = {item['expression']}")
    else:
        print(f"❌ {item['name']:10s} = 未找到")

if missing_factors:
    print(f"\n⚠️  缺失的因子: {', '.join(missing_factors)}")
    print("\n可能原因:")
    print("1. 这些因子需要特定的rolling配置才能生成")
    print("2. 需要检查Alpha158的默认配置是否包含这些因子")

# 保存为JSON
output_data = {
    "total_factors": len(names),
    "all_factors": {name: expr for name, expr in zip(names, fields)},
    "selected_20_factors": selected_info,
    "missing_factors": missing_factors
}

json_file = "alpha158_factors.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 已保存到: {json_file}")

# 创建表格格式的CSV
import csv
csv_file = "alpha158_factors.csv"
with open(csv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["因子名称", "表达式", "状态"])
    for item in selected_info:
        writer.writerow([
            item["name"],
            item["expression"] or "N/A",
            item["status"]
        ])

print(f"✅ 已保存到: {csv_file}")

# 尝试使用完整的rolling配置来生成所有因子
print("\n" + "=" * 70)
print("尝试使用完整配置生成所有因子...")
print("=" * 70)

full_config = {
    "kbar": {},
    "price": {
        "windows": [0],
        "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
    },
    "rolling": {
        "windows": [5, 10, 20, 30, 60],
        "include": None,  # 包含所有算子
        "exclude": []  # 不排除任何算子
    }
}

full_fields, full_names = Alpha158DL.get_feature_config(full_config)
full_factor_dict = dict(zip(full_names, full_fields))

print(f"\n完整配置下的因子数量: {len(full_names)}")

# 再次检查精选因子
print("\n使用完整配置检查精选因子:")
found_count = 0
for factor in selected_factors:
    if factor in full_factor_dict:
        found_count += 1
        print(f"✅ {factor:10s} = {full_factor_dict[factor]}")
    else:
        print(f"❌ {factor:10s} = 未找到")

print(f"\n找到 {found_count}/{len(selected_factors)} 个精选因子")

# 更新JSON文件
output_data["full_config_factors"] = {
    "total": len(full_names),
    "factors": {name: expr for name, expr in zip(full_names, full_fields)},
    "selected_found": found_count
}

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 已更新: {json_file}")

