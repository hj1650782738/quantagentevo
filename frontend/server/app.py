"""
QuantaAlpha 前端服务器

【API 契约】以下 API 接口保持稳定，前端依赖这些接口：
- GET  /api/health          健康检查
- GET  /api/config          获取默认配置
- POST /api/runs            启动新任务
- GET  /api/runs/<run_id>   获取任务状态
- GET  /api/runs/<run_id>/logs.zip  下载日志

【修改指南】
- 修改后端逻辑：编辑 executor.py
- 添加新 API：在此文件添加新路由（不影响现有前端）
- 修改现有 API 返回格式：需同步修改前端
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# 添加 server 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from executor import (
    build_env,
    build_mining_command,
    build_backtest_command,
    start_process,
    validate_qlib_data_path,
    validate_config_file,
    validate_factor_json,
    get_default_config,
    REPO_ROOT,
)
from run_registry import RunRegistry, RunInfo
from log_utils import make_zip_from_lines, infer_stage_from_output

# 静态文件目录
WEB_ROOT = Path(__file__).parent.parent / "web"

# 全局任务注册表
registry = RunRegistry()


def _json_response(handler: BaseHTTPRequestHandler, data: dict, status: int = 200):
    """发送 JSON 响应"""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict | None:
    """读取请求体中的 JSON"""
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _output_reader(info: RunInfo):
    """后台线程：读取进程输出"""
    try:
        for line in iter(info.popen.stdout.readline, ""):
            if not line:
                break
            registry.append_output(info, line.rstrip())
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""
    
    def log_message(self, format, *args):
        # 静默 HTTP 日志（可按需开启）
        pass
    
    def do_OPTIONS(self):
        """处理 CORS 预检请求"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        # API 路由
        if path == "/api/health":
            return _json_response(self, {"ok": True, "status": "healthy"})
        
        if path == "/api/config":
            return _json_response(self, {"ok": True, "config": get_default_config()})
        
        if path.startswith("/api/runs/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 3:
                run_id = parts[2]
                if len(parts) >= 4 and parts[3] == "logs.zip":
                    return self._handle_logs_download(run_id)
                return self._handle_status(run_id)
        
        if path == "/api/runs":
            return self._handle_list_runs()
        
        # 静态文件
        if path == "/":
            path = "/index.html"
        
        file_path = (WEB_ROOT / path.lstrip("/")).resolve()
        
        # 安全检查：确保在 WEB_ROOT 内
        if not str(file_path).startswith(str(WEB_ROOT.resolve())):
            return _json_response(self, {"ok": False, "error": "invalid path"}, 400)
        
        if not file_path.exists() or file_path.is_dir():
            self.send_response(404)
            self.end_headers()
            return
        
        # 确定 Content-Type
        content_type = "text/plain"
        suffix = file_path.suffix.lower()
        if suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif suffix in (".png", ".jpg", ".jpeg", ".svg", ".ico"):
            content_type = f"image/{suffix.lstrip('.')}"
        
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            return self._handle_start_run()
        return _json_response(self, {"ok": False, "error": "not found"}, 404)
    
    def _handle_start_run(self):
        """启动新任务"""
        payload = _read_json(self)
        if not payload:
            return _json_response(self, {"ok": False, "error": "invalid JSON"}, 400)
        
        task_type = payload.get("type", "mining")
        
        if task_type == "mining":
            return self._start_mining(payload)
        elif task_type == "backtest":
            return self._start_backtest(payload)
        else:
            return _json_response(self, {"ok": False, "error": f"unknown task type: {task_type}"}, 400)
    
    def _start_mining(self, payload: dict):
        """启动因子挖掘任务"""
        config = payload.get("config", {})
        
        # 必需参数
        direction = config.get("direction", "").strip()
        if not direction:
            return _json_response(self, {"ok": False, "error": "研究方向不能为空"}, 400)
        
        qlib_data_path = config.get("qlib_data_path", "").strip()
        valid, msg = validate_qlib_data_path(qlib_data_path)
        if not valid:
            return _json_response(self, {"ok": False, "error": msg}, 400)
        
        # API 配置
        api_config = config.get("api", {})
        api_key = api_config.get("api_key", "")
        api_url = api_config.get("api_url", "")
        model = api_config.get("model", "")
        
        if not api_key:
            return _json_response(self, {"ok": False, "error": "API Key 不能为空"}, 400)
        
        # 可选参数
        num_directions = config.get("num_directions", 10)
        max_rounds = config.get("max_rounds", 11)
        library_suffix = config.get("library_suffix", "")
        
        # 构建命令和环境变量
        cmd, extra_env = build_mining_command(
            direction=direction,
            num_directions=num_directions,
            max_rounds=max_rounds,
            library_suffix=library_suffix,
        )
        
        env = build_env(
            api_key=api_key,
            api_url=api_url,
            model=model,
            qlib_data_path=qlib_data_path,
            extra_env=extra_env,
        )
        
        # 启动进程
        try:
            popen = start_process(cmd, env)
        except Exception as e:
            return _json_response(self, {"ok": False, "error": str(e)}, 500)
        
        # 注册任务
        run_id = f"mining_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        info = registry.create(run_id, "mining", popen, {
            "direction": direction,
            "num_directions": num_directions,
            "max_rounds": max_rounds,
        })
        
        # 启动输出读取线程
        threading.Thread(target=_output_reader, args=(info,), daemon=True).start()
        
        return _json_response(self, {
            "ok": True,
            "run_id": run_id,
            "pid": popen.pid,
        })
    
    def _start_backtest(self, payload: dict):
        """启动回测任务"""
        config = payload.get("config", {})
        
        # 必需参数：配置文件
        config_path = config.get("config_path", "").strip()
        valid, msg = validate_config_file(config_path)
        if not valid:
            return _json_response(self, {"ok": False, "error": msg}, 400)
        
        # 因子源
        factor_source = config.get("factor_source", "custom")
        factor_json_paths = config.get("factor_json_paths", [])
        
        if factor_source in ["custom", "combined"] and not factor_json_paths:
            return _json_response(self, {
                "ok": False,
                "error": "使用 custom/combined 因子源时必须指定因子库文件"
            }, 400)
        
        # 验证因子库文件
        for path in factor_json_paths:
            valid, msg = validate_factor_json(path)
            if not valid:
                return _json_response(self, {"ok": False, "error": msg}, 400)
        
        # API 配置（回测可能需要 LLM 计算某些因子）
        api_config = config.get("api", {})
        api_key = api_config.get("api_key", "")
        api_url = api_config.get("api_url", "")
        model = api_config.get("model", "")
        qlib_data_path = config.get("qlib_data_path", "")
        
        # 可选参数
        experiment_name = config.get("experiment_name", "")
        dry_run = config.get("dry_run", False)
        verbose = config.get("verbose", False)
        
        # 构建命令
        cmd = build_backtest_command(
            config_path=config_path,
            factor_source=factor_source,
            factor_json_paths=factor_json_paths,
            experiment_name=experiment_name,
            dry_run=dry_run,
            verbose=verbose,
        )
        
        env = build_env(
            api_key=api_key,
            api_url=api_url,
            model=model,
            qlib_data_path=qlib_data_path,
        )
        
        # 启动进程
        try:
            popen = start_process(cmd, env)
        except Exception as e:
            return _json_response(self, {"ok": False, "error": str(e)}, 500)
        
        # 注册任务
        run_id = f"backtest_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        info = registry.create(run_id, "backtest", popen, {
            "config_path": config_path,
            "factor_source": factor_source,
        })
        
        # 启动输出读取线程
        threading.Thread(target=_output_reader, args=(info,), daemon=True).start()
        
        return _json_response(self, {
            "ok": True,
            "run_id": run_id,
            "pid": popen.pid,
        })
    
    def _handle_status(self, run_id: str):
        """获取任务状态"""
        info = registry.get(run_id)
        if not info:
            return _json_response(self, {"ok": False, "error": "任务不存在"}, 404)
        
        registry.refresh_status(info)
        output_lines = registry.get_output(info, last_n=100)
        stage = infer_stage_from_output(output_lines, info.status)
        
        return _json_response(self, {
            "ok": True,
            "run_id": info.run_id,
            "task_type": info.task_type,
            "pid": info.pid,
            "status": info.status,
            "exit_code": info.exit_code,
            "started_at": info.started_at,
            "updated_at": info.updated_at,
            "config": info.config,
            "stage": stage,
            "output": output_lines[-20:],  # 最近 20 行
        })
    
    def _handle_list_runs(self):
        """列出所有任务"""
        runs = registry.list_runs()
        result = []
        for info in runs:
            registry.refresh_status(info)
            result.append({
                "run_id": info.run_id,
                "task_type": info.task_type,
                "status": info.status,
                "started_at": info.started_at,
            })
        return _json_response(self, {"ok": True, "runs": result})
    
    def _handle_logs_download(self, run_id: str):
        """下载任务日志"""
        info = registry.get(run_id)
        if not info:
            return _json_response(self, {"ok": False, "error": "任务不存在"}, 404)
        
        output_lines = list(info.output_lines)
        zip_data = make_zip_from_lines(output_lines, f"{run_id}.log")
        
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f"attachment; filename={run_id}_logs.zip")
        self.send_header("Content-Length", str(len(zip_data)))
        self.end_headers()
        self.wfile.write(zip_data)


def main():
    parser = argparse.ArgumentParser(description="QuantaAlpha 前端服务器")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址")
    parser.add_argument("--port", type=int, default=8080, help="端口号")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           QuantaAlpha Studio                                 ║
║                                                              ║
║   前端服务已启动: http://{args.host}:{args.port}/                   ║
║                                                              ║
║   按 Ctrl+C 停止服务                                          ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
