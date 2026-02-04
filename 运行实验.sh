#!/bin/bash
# AlphaAgent 实验运行脚本
#
# 用法：
#   bash 运行实验.sh "初始方向"                      # 输出到 all_factors_library.json
#   bash 运行实验.sh "初始方向" "后缀"               # 输出到 all_factors_library_后缀.json
#
# 示例：
#   bash 运行实验.sh "价量因子挖掘"                  # → all_factors_library.json
#   bash 运行实验.sh "价量因子挖掘" "QA_exp1"        # → all_factors_library_QA_exp1.json
#
# 指定模型运行：
#   MODEL_PRESET=gemini bash 运行实验.sh "方向"      # 使用 Gemini (默认)
#   MODEL_PRESET=deepseek bash 运行实验.sh "方向"    # 使用 DeepSeek V3.2 (OpenRouter)
#   MODEL_PRESET=deepseek_aliyun bash 运行实验.sh "方向"  # 使用 DeepSeek V3.2 (阿里云 DashScope)
#   MODEL_PRESET=claude bash 运行实验.sh "方向"      # 使用 Claude Sonnet 4.5
#   MODEL_PRESET=gpt bash 运行实验.sh "方向"         # 使用 GPT-5.2
#   MODEL_PRESET=qwen bash 运行实验.sh "方向"        # 使用 Qwen3-235B (阿里云 DashScope)
#
# 或直接指定模型名称：
#   REASONING_MODEL=deepseek/deepseek-v3.2 CHAT_MODEL=deepseek/deepseek-v3.2 bash 运行实验.sh "方向"
#
# 并行运行多个实验：
#   # 实验1 - 使用 Gemini
#   MODEL_PRESET=gemini EXPERIMENT_ID=exp1 bash 运行实验.sh "方向1" "exp1"
#   # 实验2 - 使用 DeepSeek (在另一个终端)
#   MODEL_PRESET=deepseek EXPERIMENT_ID=exp2 bash 运行实验.sh "方向2" "exp2"

cd /home/tjxy/quantagent

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 检查 alphaagent 是否可用
if ! command -v alphaagent &> /dev/null; then
    echo "❌ 错误: alphaagent 命令未找到"
    echo "请先安装 AlphaAgent:"
    echo "  cd AlphaAgent && pip install -e ."
    exit 1
fi

echo "✅ 虚拟环境已激活"
echo "📦 Python: $(python --version)"
echo "📍 AlphaAgent: $(which alphaagent)"
echo ""

# 进入 AlphaAgent 目录
cd AlphaAgent

# =============================================================================
# 模型预设配置
# =============================================================================
# 可通过 MODEL_PRESET 环境变量快速切换模型
# 支持的预设: gemini (默认), deepseek, deepseek_aliyun, claude, gpt, qwen
# 也可直接通过 REASONING_MODEL 和 CHAT_MODEL 环境变量覆盖
# =============================================================================
MODEL_PRESET=${MODEL_PRESET:-""}

if [ -n "${MODEL_PRESET}" ]; then
    case "${MODEL_PRESET}" in
        gemini)
            export REASONING_MODEL="google/gemini-3-pro-preview"
            export CHAT_MODEL="google/gemini-3-pro-preview"
            echo "🤖 模型预设: Gemini 3 Pro Preview"
            ;;
        deepseek)
            export REASONING_MODEL="deepseek/deepseek-v3.2"
            export CHAT_MODEL="deepseek/deepseek-v3.2"
            echo "🤖 模型预设: DeepSeek V3.2 (OpenRouter)"
            ;;
        deepseek_aliyun)
            # 使用阿里云 DashScope API 调用 DeepSeek V3.2
            export REASONING_MODEL="deepseek-v3.2"
            export CHAT_MODEL="deepseek-v3.2"
            export OPENAI_API_KEY="${DASHSCOPE_API_KEY:-}"
            export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
            echo "🤖 模型预设: DeepSeek V3.2 (阿里云 DashScope)"
            ;;
        claude)
            export REASONING_MODEL="anthropic/claude-sonnet-4.5"
            export CHAT_MODEL="anthropic/claude-sonnet-4.5"
            echo "🤖 模型预设: Claude Sonnet 4.5"
            ;;
        gpt)
            export REASONING_MODEL="openai/gpt-5.2"
            export CHAT_MODEL="openai/gpt-5.2"
            echo "🤖 模型预设: GPT-5.2"
            ;;
        qwen)
            # 使用 DashScope API (instruct 版本支持 JSON 模式)
            export REASONING_MODEL="qwen3-235b-a22b-instruct-2507"
            export CHAT_MODEL="qwen3-235b-a22b-instruct-2507"
            export OPENAI_API_KEY="${DASHSCOPE_API_KEY:-}"
            export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
            echo "🤖 模型预设: Qwen3-235B Instruct (DashScope)"
            ;;
        *)
            echo "⚠️ 未知的模型预设: ${MODEL_PRESET}"
            echo "   支持的预设: gemini, deepseek, deepseek_aliyun, claude, gpt, qwen"
            echo "   将使用 .env 文件中的默认配置"
            ;;
    esac
fi

# 显示当前使用的模型
if [ -n "${REASONING_MODEL}" ]; then
    echo "   推理模型: ${REASONING_MODEL}"
fi
if [ -n "${CHAT_MODEL}" ]; then
    echo "   对话模型: ${CHAT_MODEL}"
fi
echo ""

# 运行实验
# 默认从配置文件读取参数：alphaagent/app/qlib_rd_loop/run_config.yaml
CONFIG_PATH=${CONFIG_PATH:-"alphaagent/app/qlib_rd_loop/run_config.yaml"}
STEP_N=${STEP_N:-""}

# 实验隔离配置 - 每次实验自动生成独立的工作空间和缓存目录
# 可以通过 EXPERIMENT_ID 环境变量手动指定，否则自动生成时间戳ID
# 设置 EXPERIMENT_ID=shared 可以使用共享的默认目录（向后兼容）
if [ -z "${EXPERIMENT_ID}" ]; then
    # 自动生成基于时间戳的实验ID: exp_YYYYMMDD_HHMMSS
    EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
fi
# 导出 EXPERIMENT_ID 供 Python 子进程使用（用于因子缓存路径记录）
export EXPERIMENT_ID

if [ "${EXPERIMENT_ID}" != "shared" ]; then
    export WORKSPACE_PATH="/mnt/DATA/quantagent/AlphaAgent/RD-Agent_workspace_${EXPERIMENT_ID}"
    export PICKLE_CACHE_FOLDER_PATH_STR="/mnt/DATA/quantagent/AlphaAgent/pickle_cache_${EXPERIMENT_ID}"
    echo "🔀 实验隔离模式: EXPERIMENT_ID=${EXPERIMENT_ID}"
    echo "   工作空间: ${WORKSPACE_PATH}"
    echo "   缓存目录: ${PICKLE_CACHE_FOLDER_PATH_STR}"
    # 自动创建目录
    mkdir -p "${WORKSPACE_PATH}"
    mkdir -p "${PICKLE_CACHE_FOLDER_PATH_STR}"
else
    echo "📁 使用共享目录模式 (EXPERIMENT_ID=shared)"
fi

# 解析参数
DIRECTION="$1"
LIBRARY_SUFFIX="$2"

# 设置因子库输出路径（通过环境变量传递）
if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
    LIBRARY_FILE="all_factors_library_${LIBRARY_SUFFIX}.json"
else
    export FACTOR_LIBRARY_SUFFIX=""
    LIBRARY_FILE="all_factors_library.json"
fi

# 回测配置说明
# 数据时间范围: 2016-01-01 ~ 2025-12-31
# 训练集: 2016-01-01 ~ 2020-12-31
# 验证集: 2021-01-01 ~ 2021-12-31
# 测试集: 2022-01-01 ~ 2025-12-31
# 回测时间: 2022-01-01 ~ 2025-12-31 (在测试集上进行回测)
# 配置文件位置:
#   - alphaagent/scenarios/qlib/experiment/factor_template/conf.yaml
#   - alphaagent/scenarios/qlib/experiment/factor_template/conf_cn_combined_kdd_ver.yaml

echo "🚀 开始运行实验..."
echo "📄 配置文件: ${CONFIG_PATH}"
echo "📂 因子库输出: ${LIBRARY_FILE}"
echo "📅 回测时间: 2022-01-01 ~ 2025-12-31"
echo "----------------------------------------"
if [ -n "${STEP_N}" ]; then
  alphaagent mine --direction "${DIRECTION}" --step_n "${STEP_N}" --config_path "${CONFIG_PATH}"
else
  alphaagent mine --direction "${DIRECTION}" --config_path "${CONFIG_PATH}"
fi

