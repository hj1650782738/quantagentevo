# AA vs QA 因子库 IC 衰减分析

本项目系统地分析 AA (AlphaAgent) 和 QA (QuantaAlpha) 两个因子库在不同年份的表现差异，特别关注 **2023年左右 AA 方法因子 IC 出现显著下降而 QA 未受影响** 的现象。

## 📁 项目结构

```
ic_decay_analysis/
├── run_analysis.py              # 主运行脚本
├── README.md                    # 本文档
├── analysis.log                 # 运行日志
│
├── configs/                     # 分年度回测配置
│   ├── config_AA_2021.yaml
│   ├── config_AA_2022.yaml
│   ├── ...
│   ├── config_QA_2025.yaml
│   └── config_index.yaml        # 配置索引
│
├── scripts/                     # 分析脚本
│   ├── generate_yearly_configs.py    # 配置生成器
│   ├── run_yearly_backtests.py       # 分年度回测执行器
│   ├── factor_level_ic_analysis.py   # 因子级别IC分析
│   ├── factor_importance_analysis.py # 因子重要性分析
│   └── comprehensive_report.py       # 综合报告生成器
│
└── results/                     # 分析结果
    ├── AA/                      # AA因子库回测结果
    ├── QA/                      # QA因子库回测结果
    ├── yearly_backtest_results.json
    ├── factor_level_ic_results.json
    ├── decaying_factors.json
    ├── factor_importance_by_year.json
    ├── dominant_factors.json
    ├── importance_shift_2022_2023.json
    ├── library_yearly_comparison.csv
    └── comprehensive_analysis_report.md  # 最终报告
```

## 🚀 快速开始

### 1. 完整分析（推荐）

```bash
cd /home/tjxy/quantagent/AlphaAgent/ic_decay_analysis
python run_analysis.py --full
```

这将执行完整的分析流程：
1. 生成分年度回测配置（2021-2025年）
2. 对 AA 和 QA 两个因子库分别执行各年度回测
3. 计算每个因子的单因子 IC
4. 分析因子在模型中的重要性变化
5. 生成综合分析报告

⏱️ 预计耗时：4-6 小时

### 2. 快速测试

```bash
python run_analysis.py --quick
```

每个因子库仅分析前 20 个因子，用于快速验证流程。

⏱️ 预计耗时：30-60 分钟

### 3. 指定分析范围

```bash
# 仅分析 2022 和 2023 年
python run_analysis.py --years 2022 2023

# 仅分析 AA 因子库
python run_analysis.py --libraries AA

# 组合使用
python run_analysis.py --libraries AA QA --years 2022 2023 2024
```

### 4. 分步执行

```bash
# 仅执行回测
python run_analysis.py --backtest-only

# 仅执行因子级别IC分析
python run_analysis.py --ic-only

# 仅执行因子重要性分析
python run_analysis.py --importance-only

# 仅生成报告（使用已有结果）
python run_analysis.py --report-only
```

## 📊 分析内容

### 1. 分年度回测
- 训练集：使用目标年份前2-5年的数据
- 验证集：使用目标年份前1年的数据
- 测试集：目标年份全年
- 输出：IC、ICIR、Rank IC、Rank ICIR、年化收益、最大回撤等指标

### 2. 因子级别 IC 分析
- 计算每个因子在各年份的单因子 Rank IC
- 识别 IC 衰减超过 30% 的因子
- 分析衰减因子的类型分布
- 对比 AA 和 QA 因子库的衰减模式

### 3. 因子重要性分析
- 训练 LightGBM 模型获取因子重要性（Gain/Split）
- 识别跨年度稳定的高重要性因子
- 分析 2022→2023 年重要性变化最大的因子
- 对比两个因子库的重要性分布特征

### 4. 综合报告
- 执行摘要与关键发现
- 分年度 IC 变化趋势图表
- 因子衰减分类统计
- 改进建议与后续研究方向

## ⚙️ 配置说明

### 数据时间段划分

| 测试年份 | 训练集 | 验证集 | 测试集 |
|---------|--------|--------|--------|
| 2021 | 2016-01-01 ~ 2019-12-31 | 2020全年 | 2021全年 |
| 2022 | 2016-01-01 ~ 2020-12-31 | 2021全年 | 2022全年 |
| 2023 | 2016-01-01 ~ 2021-12-31 | 2022全年 | 2023全年 |
| 2024 | 2016-01-01 ~ 2022-12-31 | 2023全年 | 2024全年 |
| 2025 | 2016-01-01 ~ 2023-12-31 | 2024全年 | 2025全年 |

### 因子库信息

| 因子库 | 路径 | 因子数量 |
|-------|------|---------|
| AA | `factor_library/AA_top80_RankIC_AA_gpt_123_csi300.json` | 80 |
| QA | `factor_library/hj/RANKIC_desc_150_QA_round11_best_gpt_123_csi300.json` | 150 |

### 缓存配置

分析会优先使用因子库中记录的预计算因子值（`cache_location.result_h5_path`），避免重复计算。如果缓存不存在，会尝试：
1. 使用 Qlib 直接计算（对于兼容的表达式）
2. 使用自定义计算器（对于复杂表达式）

## 📈 预期输出示例

### 分年度 IC 对比表

```
Library  Year   Mean Rank IC  Change vs Prev Year
AA       2021   0.0280        --
AA       2022   0.0265        ↓ 5.4%
AA       2023   0.0185        ↓ 30.2%  ⚠️
AA       2024   0.0210        ↑ 13.5%
AA       2025   0.0195        ↓ 7.1%

QA       2021   0.0295        --
QA       2022   0.0288        ↓ 2.4%
QA       2023   0.0278        ↓ 3.5%
QA       2024   0.0282        ↑ 1.4%
QA       2025   0.0275        ↓ 2.5%
```

### IC 衰减因子 Top 5

```
Factor Name                              2022 IC    2023 IC    Change
ASVC_Refined_Climax_20D                 0.0268     0.0120     -55.2%
Volume_Price_Divergence_15D             0.0245     0.0125     -49.0%
Momentum_Volatility_Interaction_10D     0.0232     0.0128     -44.8%
...
```

## 🔍 问题排查

### 缓存加载失败
如果看到 "缓存加载失败" 警告，检查：
1. `cache_location.result_h5_path` 路径是否正确
2. HDF5 文件是否存在且未损坏
3. 磁盘空间是否充足

### 内存不足
完整分析需要约 32GB 内存。如果内存不足：
- 使用 `--quick` 模式
- 减少同时分析的年份数量
- 分步执行各个分析阶段

### Qlib 初始化失败
确保：
1. Qlib 数据目录存在：`/home/tjxy/.qlib/qlib_data/cn_data`
2. CSI300 数据完整

## 📝 引用

如果此分析对您的研究有帮助，请引用：

```
QuantaAlpha Factor Analysis Framework
IC Decay Analysis Module
2026
```

## 📧 联系方式

如有问题，请联系项目维护者。

