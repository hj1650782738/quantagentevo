#!/usr/bin/env python3
"""
分年度回测配置生成器

为 AA 和 QA 两个因子库分别生成 2021-2025 年的回测配置
注意事项：
1. 训练集使用测试年份之前的数据
2. 验证集使用测试年份的前一年
3. 测试集使用指定年份
4. 确保缓存路径正确配置
"""

import os
import yaml
from pathlib import Path


# 因子库配置
FACTOR_LIBRARIES = {
    "AA": "/home/tjxy/quantagent/AlphaAgent/factor_library/AA_top80_RankIC_AA_gpt_123_csi300.json",
    "QA": "/home/tjxy/quantagent/AlphaAgent/factor_library/hj/RANKIC_desc_150_QA_round11_best_gpt_123_csi300.json"
}

# 回测年份
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

# 基础配置模板
BASE_CONFIG = {
    "random_seed": 42,
    "experiment": {
        "name": "yearly_ic_analysis",
        "recorder": "yearly_recorder",
        "output_dir": "./ic_decay_analysis/results",
        "output_metrics_file": "backtest_metrics.json"
    },
    "factor_source": {
        "type": "custom",
        "custom": {
            "json_files": [],
            "quality_filter": None,
            "max_factors": None,
            "use_llm_for_incompatible": True
        },
        "combined": {
            "official_source": "alpha158_20",
            "include_custom": True
        }
    },
    "data": {
        "provider_uri": "/home/tjxy/.qlib/qlib_data/cn_data",
        "region": "cn",
        "market": "csi300",
        "start_time": "2016-01-01",
        "end_time": "2025-12-31"
    },
    "dataset": {
        "label": "Ref($close, -2) / Ref($close, -1) - 1",
        "learn_processors": [
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            {"class": "ProcessInf"},
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
        ],
        "infer_processors": [
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            {"class": "ProcessInf"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
        ],
        "segments": {
            "train": ["2016-01-01", "2019-12-31"],
            "valid": ["2020-01-01", "2020-12-31"],
            "test": ["2021-01-01", "2021-12-31"]
        }
    },
    "model": {
        "type": "lgb",
        "params": {
            "loss": "mse",
            "learning_rate": 0.1,
            "max_depth": 8,
            "num_leaves": 210,
            "colsample_bytree": 0.8879,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "num_threads": 20,
            "seed": 42,
            "random_state": 42,
            "early_stopping_round": 50,
            "num_boost_round": 500,
            "min_child_samples": 100,
            "feature_fraction_bynode": 0.8
        }
    },
    "backtest": {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": 50,
                "n_drop": 5
            }
        },
        "backtest": {
            "start_time": "2021-01-01",
            "end_time": "2021-12-31",
            "account": 100000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "open",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5
            }
        }
    },
    "llm": {
        "enabled": True,
        "timeout": 300,
        "max_retries": 3,
        "cache_results": True,
        "cache_dir": "/mnt/DATA/quantagent/AlphaAgent/factor_cache",
        "auto_extract_cache": False,
        "debug": False
    },
    "factor_calculation": {
        "output_dir": "/mnt/DATA/quantagent/AlphaAgent/computed_factors",
        "save_intermediate": True,
        "n_jobs": 4,
        "data_file": None
    }
}


def generate_config_for_year(library_name: str, library_path: str, test_year: int) -> dict:
    """
    为指定年份生成配置
    
    训练集：2016 ~ (test_year - 2)
    验证集：(test_year - 1)
    测试集：test_year
    """
    config = yaml.safe_load(yaml.dump(BASE_CONFIG))  # Deep copy
    
    # 设置因子库
    config["factor_source"]["custom"]["json_files"] = [library_path]
    
    # 设置实验名称
    config["experiment"]["name"] = f"{library_name}_{test_year}"
    config["experiment"]["output_dir"] = f"./ic_decay_analysis/results/{library_name}"
    
    # 设置数据时间范围
    config["data"]["start_time"] = "2016-01-01"
    config["data"]["end_time"] = f"{test_year}-12-31"
    
    # 设置数据集划分
    train_end_year = test_year - 2
    valid_year = test_year - 1
    
    config["dataset"]["segments"] = {
        "train": ["2016-01-01", f"{train_end_year}-12-31"],
        "valid": [f"{valid_year}-01-01", f"{valid_year}-12-31"],
        "test": [f"{test_year}-01-01", f"{test_year}-12-31"]
    }
    
    # 设置回测时间
    config["backtest"]["backtest"]["start_time"] = f"{test_year}-01-01"
    config["backtest"]["backtest"]["end_time"] = f"{test_year}-12-31"
    
    return config


def main():
    """生成所有配置文件"""
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    generated_configs = []
    
    for lib_name, lib_path in FACTOR_LIBRARIES.items():
        print(f"\n📁 生成 {lib_name} 因子库配置...")
        
        for year in TEST_YEARS:
            config = generate_config_for_year(lib_name, lib_path, year)
            
            # 保存配置
            config_filename = f"config_{lib_name}_{year}.yaml"
            config_path = config_dir / config_filename
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"  ✓ {config_filename}")
            
            generated_configs.append({
                "library": lib_name,
                "year": year,
                "config_path": str(config_path),
                "factor_json": lib_path
            })
    
    # 保存配置索引
    index_path = config_dir / "config_index.yaml"
    with open(index_path, 'w', encoding='utf-8') as f:
        yaml.dump(generated_configs, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ 共生成 {len(generated_configs)} 个配置文件")
    print(f"📋 配置索引: {index_path}")


if __name__ == "__main__":
    main()

