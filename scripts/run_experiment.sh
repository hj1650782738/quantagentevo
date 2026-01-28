#!/bin/bash
# ============================================================================
# QuantaAlpha - 因子挖掘实验运行脚本
# ============================================================================
#
# 用法：
#   bash scripts/run_experiment.sh "初始方向"              # 使用默认配置
#   bash scripts/run_experiment.sh "初始方向" "后缀"       # 指定输出后缀
#
# 示例：
#   bash scripts/run_experiment.sh "价量因子挖掘"
#   bash scripts/run_experiment.sh "价量因子挖掘" "exp1"
#
# 模型配置（通过环境变量）：
#   MODEL_PRESET=gemini bash scripts/run_experiment.sh "方向"
#   MODEL_PRESET=deepseek bash scripts/run_experiment.sh "方向"
#   MODEL_PRESET=claude bash scripts/run_experiment.sh "方向"
#
# 支持的模型预设: gemini, deepseek, deepseek_aliyun, claude, gpt, qwen
#
# ============================================================================

# 获取脚本所在目录的父目录作为项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "❌ 错误: .env 文件不存在"
    echo "请从 .env.example 复制并配置您的 API 密钥:"
    echo "  cp .env.example .env"
    exit 1
fi

# 加载环境变量
set -a
source .env
set +a

# 检查 quantaalpha 是否可用
if ! command -v quantaalpha &> /dev/null; then
    echo "❌ 错误: quantaalpha 命令未找到"
    echo "请先安装 QuantaAlpha:"
    echo "  pip install -e ."
    exit 1
fi

echo "============================================"
echo "  QuantaAlpha - 因子挖掘实验"
echo "============================================"
echo "📦 Python: $(python --version)"
echo "📍 QuantaAlpha: $(which quantaalpha)"
echo ""

# =============================================================================
# 模型预设配置
# =============================================================================
MODEL_PRESET=${MODEL_PRESET:-""}

if [ -n "${MODEL_PRESET}" ]; then
    case "${MODEL_PRESET}" in
        gemini)
            export REASONING_MODEL="google/gemini-3-flash-preview"
            export CHAT_MODEL="google/gemini-3-flash-preview"
            echo "🤖 模型: Gemini 3 Flash Preview"
            ;;
        deepseek)
            export REASONING_MODEL="deepseek/deepseek-v3.2"
            export CHAT_MODEL="deepseek/deepseek-v3.2"
            echo "🤖 模型: DeepSeek V3.2 (OpenRouter)"
            ;;
        deepseek_aliyun)
            export REASONING_MODEL="deepseek-v3.2"
            export CHAT_MODEL="deepseek-v3.2"
            export OPENAI_API_KEY="${DASHSCOPE_API_KEY}"
            export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
            echo "🤖 模型: DeepSeek V3.2 (阿里云)"
            ;;
        claude)
            export REASONING_MODEL="anthropic/claude-sonnet-4.5"
            export CHAT_MODEL="anthropic/claude-sonnet-4.5"
            echo "🤖 模型: Claude Sonnet 4.5"
            ;;
        gpt)
            export REASONING_MODEL="openai/gpt-5.2"
            export CHAT_MODEL="openai/gpt-5.2"
            echo "🤖 模型: GPT-5.2"
            ;;
        qwen)
            export REASONING_MODEL="qwen3-235b-a22b-instruct-2507"
            export CHAT_MODEL="qwen3-235b-a22b-instruct-2507"
            export OPENAI_API_KEY="${DASHSCOPE_API_KEY}"
            export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
            echo "🤖 模型: Qwen3-235B"
            ;;
        *)
            echo "⚠️ 未知的模型预设: ${MODEL_PRESET}"
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

# =============================================================================
# 配置路径
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"configs/run_config.yaml"}
export CONFIG_PATH

# 实验隔离
if [ -z "${EXPERIMENT_ID}" ]; then
    EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
fi

if [ "${EXPERIMENT_ID}" != "shared" ]; then
    export WORKSPACE_PATH="${PROJECT_ROOT}/data/workspace_${EXPERIMENT_ID}"
    export PICKLE_CACHE_FOLDER_PATH_STR="${PROJECT_ROOT}/data/cache_${EXPERIMENT_ID}"
    echo "🔀 实验ID: ${EXPERIMENT_ID}"
    mkdir -p "${WORKSPACE_PATH}" "${PICKLE_CACHE_FOLDER_PATH_STR}"
fi

# 解析参数
DIRECTION="$1"
LIBRARY_SUFFIX="$2"

if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
fi

echo "============================================"
echo "🚀 开始运行实验..."
echo "📄 配置: ${CONFIG_PATH}"
echo "📂 输出: data/factors/"
echo "============================================"

if [ -n "${STEP_N}" ]; then
    quantaalpha mine --direction "${DIRECTION}" --step_n "${STEP_N}" --config_path "${CONFIG_PATH}"
else
    quantaalpha mine --direction "${DIRECTION}" --config_path "${CONFIG_PATH}"
fi
