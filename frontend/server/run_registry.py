"""
运行任务注册管理

管理所有正在运行和已完成的任务
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunInfo:
    """运行任务信息"""
    run_id: str
    task_type: str  # "mining" 或 "backtest"
    pid: int
    popen: object
    started_at: float
    updated_at: float
    status: str = "starting"  # starting, running, done, failed
    exit_code: Optional[int] = None
    config: Dict = field(default_factory=dict)
    output_lines: deque = field(default_factory=lambda: deque(maxlen=500))


class RunRegistry:
    """任务注册表"""
    
    def __init__(self):
        self._runs: Dict[str, RunInfo] = {}
        self._lock = threading.Lock()
    
    def create(
        self,
        run_id: str,
        task_type: str,
        popen,
        config: Dict
    ) -> RunInfo:
        """创建新的运行任务"""
        now = time.time()
        info = RunInfo(
            run_id=run_id,
            task_type=task_type,
            pid=popen.pid,
            popen=popen,
            started_at=now,
            updated_at=now,
            config=config,
        )
        with self._lock:
            self._runs[run_id] = info
        return info
    
    def get(self, run_id: str) -> Optional[RunInfo]:
        """获取任务信息"""
        with self._lock:
            return self._runs.get(run_id)
    
    def list_runs(self, task_type: Optional[str] = None) -> List[RunInfo]:
        """列出所有任务"""
        with self._lock:
            runs = list(self._runs.values())
        if task_type:
            runs = [r for r in runs if r.task_type == task_type]
        return runs
    
    def refresh_status(self, info: RunInfo) -> None:
        """刷新任务状态"""
        exit_code = info.popen.poll()
        info.exit_code = exit_code
        if exit_code is None:
            info.status = "running"
        else:
            info.status = "done" if exit_code == 0 else "failed"
        info.updated_at = time.time()
    
    def append_output(self, info: RunInfo, line: str) -> None:
        """追加输出行"""
        info.output_lines.append(line)
        info.updated_at = time.time()
    
    def get_output(self, info: RunInfo, last_n: int = 100) -> List[str]:
        """获取最近的输出行"""
        return list(info.output_lines)[-last_n:]
    
    def delete(self, run_id: str) -> bool:
        """删除任务记录"""
        with self._lock:
            if run_id in self._runs:
                del self._runs[run_id]
                return True
            return False
