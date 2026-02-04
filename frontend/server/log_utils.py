"""
日志工具

处理日志输出和打包
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import List


def make_zip_from_lines(lines: List[str], filename: str = "output.log") -> bytes:
    """
    将输出行打包成 ZIP 文件
    
    Args:
        lines: 输出行列表
        filename: ZIP 内的文件名
    
    Returns:
        ZIP 文件的字节内容
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        content = "\n".join(lines)
        zf.writestr(filename, content.encode("utf-8"))
    return buffer.getvalue()


def make_zip_from_dir(log_dir: Path, out_path: Path) -> Path:
    """
    将目录打包成 ZIP 文件
    
    Args:
        log_dir: 日志目录
        out_path: 输出 ZIP 文件路径
    
    Returns:
        输出文件路径
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in log_dir.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(log_dir)
            zf.write(p, arcname=str(rel))
    return out_path


def infer_stage_from_output(lines: List[str], status: str) -> dict:
    """
    从输出行推断当前阶段
    
    【修改指南】如需修改阶段识别逻辑，修改此函数
    """
    if status == "done":
        return {"name": "完成", "progress": 1.0, "detail": "运行完成"}
    if status == "failed":
        return {"name": "失败", "progress": 1.0, "detail": "运行失败"}
    if status == "starting":
        return {"name": "启动中", "progress": 0.05, "detail": "正在启动..."}
    
    # 从输出行推断阶段
    stage_keywords = [
        ("进化轮次", "进化中", 0.6),
        ("变异", "变异阶段", 0.5),
        ("交叉", "交叉阶段", 0.7),
        ("回测", "回测中", 0.8),
        ("因子构建", "构建因子", 0.4),
        ("假设生成", "生成假设", 0.3),
        ("规划", "规划阶段", 0.2),
        ("初始化", "初始化", 0.1),
    ]
    
    # 从后往前扫描输出
    for line in reversed(lines[-50:]):
        for keyword, stage_name, progress in stage_keywords:
            if keyword in line:
                return {
                    "name": stage_name,
                    "progress": progress,
                    "detail": line.strip()[:100]
                }
    
    return {"name": "运行中", "progress": 0.5, "detail": "处理中..."}
