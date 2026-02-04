"""
执行层 - 封装实际的命令执行逻辑

【重要】如果需要修改后端功能，只需修改此文件：
- 修改命令参数：修改 build_mining_command() 或 build_backtest_command()
- 修改环境变量：修改 build_env()
- 添加新功能：添加新的 build_xxx_command() 函数

只要不改变函数签名和返回值结构，前端就不会受影响。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# 项目根目录
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_env(
    api_key: str = "",
    api_url: str = "",
    model: str = "",
    qlib_data_path: str = "",
    extra_env: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    构建子进程环境变量
    
    【修改指南】如需添加新的环境变量，在此函数中添加
    """
    env = os.environ.copy()
    
    # API 配置
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    if api_url:
        env["OPENAI_BASE_URL"] = api_url
    if model:
        env["REASONING_MODEL"] = model
        env["CHAT_MODEL"] = model
    
    # Qlib 数据路径
    if qlib_data_path:
        env["QLIB_PROVIDER_URI"] = qlib_data_path
    
    # 额外环境变量
    if extra_env:
        env.update(extra_env)
    
    return env


def build_mining_command(
    direction: str,
    num_directions: int = 10,
    max_rounds: int = 11,
    library_suffix: str = "",
    config_path: str = ""
) -> Tuple[List[str], Dict[str, str]]:
    """
    构建因子挖掘命令
    
    【修改指南】如需修改挖掘命令的参数，修改此函数
    
    Returns:
        (command_list, extra_env_dict)
    """
    # 默认配置文件路径
    if not config_path:
        config_path = str(REPO_ROOT / "alphaagent" / "app" / "qlib_rd_loop" / "run_config.yaml")
    
    cmd = [
        "alphaagent", "mine",
        "--direction", direction,
        "--config_path", config_path,
    ]
    
    # 额外的环境变量用于覆盖配置
    extra_env = {
        "PLANNING_NUM_DIRECTIONS": str(num_directions),
        "EVOLUTION_MAX_ROUNDS": str(max_rounds),
    }
    
    if library_suffix:
        extra_env["FACTOR_LIBRARY_SUFFIX"] = library_suffix
    
    return cmd, extra_env


def build_backtest_command(
    config_path: str,
    factor_source: str = "custom",
    factor_json_paths: Optional[List[str]] = None,
    experiment_name: str = "",
    dry_run: bool = False,
    verbose: bool = False
) -> List[str]:
    """
    构建回测命令
    
    【修改指南】如需修改回测命令的参数，修改此函数
    """
    cmd = [
        "python",
        str(REPO_ROOT / "backtest_v2" / "run_backtest.py"),
        "-c", config_path,
        "--factor-source", factor_source,
    ]
    
    if factor_json_paths:
        for path in factor_json_paths:
            cmd.extend(["--factor-json", path])
    
    if experiment_name:
        cmd.extend(["--experiment", experiment_name])
    
    if dry_run:
        cmd.append("--dry-run")
    
    if verbose:
        cmd.append("--verbose")
    
    return cmd


def start_process(
    cmd: List[str],
    env: Dict[str, str],
    cwd: Optional[str] = None
) -> subprocess.Popen:
    """
    启动子进程
    
    【修改指南】如需修改进程启动方式（如日志重定向），修改此函数
    """
    if cwd is None:
        cwd = str(REPO_ROOT)
    
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # 行缓冲
    )


def validate_qlib_data_path(path: str) -> Tuple[bool, str]:
    """
    验证 Qlib 数据路径是否有效
    
    【修改指南】如需修改验证逻辑，修改此函数
    """
    if not path:
        return False, "数据路径不能为空"
    
    p = Path(path)
    if not p.exists():
        return False, f"路径不存在: {path}"
    
    if not p.is_dir():
        return False, f"路径不是目录: {path}"
    
    # 检查是否包含 Qlib 数据的典型文件/目录
    # 可根据实际情况调整检查逻辑
    expected_items = ["calendars", "instruments", "features"]
    found_items = [item for item in expected_items if (p / item).exists()]
    
    if len(found_items) == 0:
        return False, f"路径不像是 Qlib 数据目录（缺少 {expected_items}）"
    
    return True, "验证通过"


def validate_config_file(path: str) -> Tuple[bool, str]:
    """
    验证配置文件是否有效
    """
    if not path:
        return False, "配置文件路径不能为空"
    
    p = Path(path)
    if not p.exists():
        return False, f"配置文件不存在: {path}"
    
    if p.suffix.lower() not in [".yaml", ".yml"]:
        return False, "配置文件必须是 YAML 格式"
    
    return True, "验证通过"


def validate_factor_json(path: str) -> Tuple[bool, str]:
    """
    验证因子库 JSON 文件是否有效
    """
    if not path:
        return False, "因子库文件路径不能为空"
    
    p = Path(path)
    if not p.exists():
        return False, f"因子库文件不存在: {path}"
    
    if p.suffix.lower() != ".json":
        return False, "因子库文件必须是 JSON 格式"
    
    return True, "验证通过"


def get_default_config() -> Dict:
    """
    获取默认配置（供前端显示）
    
    【修改指南】如需修改默认值，修改此函数
    """
    return {
        "mining": {
            "num_directions": 10,
            "max_rounds": 11,
            "default_config_path": str(REPO_ROOT / "alphaagent" / "app" / "qlib_rd_loop" / "run_config.yaml"),
        },
        "backtest": {
            "factor_sources": ["alpha158", "alpha158_20", "alpha360", "custom", "combined"],
            "default_config_path": str(REPO_ROOT / "backtest_v2" / "config.yaml"),
        },
        "api": {
            "default_model": "deepseek/deepseek-v3.2",
            "default_api_url": "https://openrouter.ai/api/v1",
        }
    }
