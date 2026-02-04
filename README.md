# QuantaAlpha

面向因子挖掘与回测的 LLM 驱动实验框架，支持一致性/冗余度/复杂度质量门控与可复现实验流程。

---

## ✅ 核心功能

- **主实验因子挖掘**：由 LLM 提出假设与因子表达式，并进行小规模回测评分
- **因子库输出**：生成 `all_factors_library*.json` 供后续组合使用
- **独立回测**：从因子库自由组合因子，进行完整回测评估
- **质量门控**：一致性检验、冗余度检验、复杂度检验可自由开关

---

## 🧩 项目结构（核心部分）

```
alphaagent/                      # 核心因子挖掘模块
backtest_v2/                     # 独立回测模块
运行实验.sh                      # 主实验入口脚本
alphaagent/app/qlib_rd_loop/     # 主流程与配置
```

---

## ⚡ 快速开始

### 1. 环境准备

```bash
conda create -n quantaalpha python=3.10
conda activate quantaalpha
pip install -e .
pip install -r requirements.txt
```

### 2. 配置 `.env`

复制模板并填入 API Key：

```bash
cp .env.example .env
```

`.env` 中常用配置：

```
OPENAI_BASE_URL=...
OPENAI_API_KEY=...
REASONING_MODEL=...
CHAT_MODEL=...
USE_LOCAL=True
```

### 3. Qlib 数据准备

必须先准备 Qlib 数据目录，否则会报错：

```bash
python -c "import qlib; qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')"
```

如果没有数据，请参考 Qlib 官方数据准备流程。

---

## 🚀 主实验：因子挖掘

入口脚本：`运行实验.sh`

```bash
bash 运行实验.sh "价量因子挖掘"
```

常用环境变量：

```
MODEL_PRESET=deepseek
CONFIG_PATH=alphaagent/app/qlib_rd_loop/run_config.yaml
STEP_N=50
EXPERIMENT_ID=exp_demo
```

输出：

- 因子库：`all_factors_library*.json`
- 缓存：由 `WORKSPACE_PATH` / `PICKLE_CACHE_FOLDER_PATH_STR` 控制

---

## 📊 独立回测（组合因子）

入口脚本：`backtest_v2/run_backtest.py`

### 使用自定义因子库：

```bash
python backtest_v2/run_backtest.py -c backtest_v2/config.yaml \
  --factor-source custom \
  --factor-json /path/to/factors.json
```

### 组合官方因子 + 自定义因子：

```bash
python backtest_v2/run_backtest.py -c backtest_v2/config.yaml \
  --factor-source combined \
  --factor-json /path/to/factors.json
```

回测缓存路径由 `backtest_v2/config.yaml` 中 `llm.cache_dir` 控制。

---

## 🧪 质量门控（可选）

配置文件：`alphaagent/app/qlib_rd_loop/run_config.yaml`

```yaml
quality_gate:
  consistency_enabled: false
  complexity_enabled: true
  redundancy_enabled: true
  consistency_strict_mode: false
  max_correction_attempts: 3
```

说明：

- 一致性检验：使用 LLM 核对“假设-描述-公式-表达式”
- 复杂度检验：限制表达式过长/过度参数化
- 冗余度检验：避免与已有因子高度重复

---

## ❓ 常见问题

- **Qlib 报错找不到数据**：确认 `provider_uri` 指向正确数据目录  
- **因子无法解析**：检查表达式函数是否在 parser 支持范围内  
- **缓存读取失败**：确保缓存目录存在并可写  

---

## 📜 License

MIT
