# QuantaAlpha 前端界面

轻量级 Web UI，用于运行因子挖掘实验和回测任务。

> **注意**：该前端用于演示和复现实验。用户需要自行提供 Qlib 兼容格式的数据。

## 🚀 快速开始

### 1. 启动服务器

在项目根目录下运行：

```bash
python frontend/server/app.py --host 127.0.0.1 --port 8080
```

如需远程访问（如服务器部署）：

```bash
python frontend/server/app.py --host 0.0.0.0 --port 8080
```

### 2. 访问界面

打开浏览器访问：

```
http://127.0.0.1:8080/
```

## ✨ 功能特性

### 因子挖掘

- 输入研究方向，启动 LLM 驱动的因子挖掘
- 配置并行方向数和进化轮次
- 实时查看运行状态和输出
- 支持多种 LLM API（OpenRouter、DashScope 等）

### 回测

- 支持多种因子源（alpha158、alpha360、自定义）
- 上传自定义因子库 JSON 文件
- 使用用户提供的回测配置

## 📦 用户需要提供

1. **Qlib 兼容格式的数据**
   - 包含 `calendars/`、`instruments/`、`features/` 等目录
   - 数据路径在界面中填写

2. **API 配置**
   - API Key（如 OpenRouter、DashScope 等）
   - API URL（可选，默认 OpenRouter）
   - 模型名称（如 `deepseek/deepseek-v3.2`）

3. **回测配置文件**（仅回测功能）
   - YAML 格式的配置文件
   - 自定义因子库 JSON 文件（如使用 custom/combined 因子源）

## 🔒 安全说明

- **API Key 安全**：API Key 仅注入到子进程环境变量中，运行结束后销毁
- **不持久化**：任何敏感信息都不会保存到磁盘
- **本地绑定**：默认绑定 `127.0.0.1`，仅本机可访问

## 🛠️ 架构说明

```
frontend/
├── server/
│   ├── app.py           # API 层（稳定接口）
│   ├── executor.py      # 执行层（后端逻辑，可修改）
│   ├── run_registry.py  # 任务注册管理
│   └── log_utils.py     # 日志工具
└── web/
    ├── index.html       # 主页面
    ├── app.js           # 前端逻辑
    └── styles.css       # 样式
```

## 🔧 如何修改后端功能

本前端采用**松耦合架构**，后端修改不会导致前端崩溃。

### 修改规则

| 修改场景 | 修改位置 | 前端影响 |
|----------|----------|----------|
| 添加新的命令参数 | `executor.py` | 无影响 |
| 修改命令执行方式 | `executor.py` | 无影响 |
| 修改环境变量注入 | `executor.py` → `build_env()` | 无影响 |
| 修改验证逻辑 | `executor.py` → `validate_*()` | 无影响 |
| 添加新的 API 端点 | `app.py` 新增路由 | 无影响 |
| 修改现有 API 返回格式 | `app.py` | **需同步修改前端** |

### 常见修改示例

#### 1. 添加新的环境变量

编辑 `executor.py` 的 `build_env()` 函数：

```python
def build_env(...):
    env = os.environ.copy()
    # ... 现有代码 ...
    
    # 添加新的环境变量
    if some_new_config:
        env["NEW_ENV_VAR"] = some_new_config
    
    return env
```

#### 2. 修改挖掘命令参数

编辑 `executor.py` 的 `build_mining_command()` 函数：

```python
def build_mining_command(...):
    cmd = [
        "alphaagent", "mine",
        "--direction", direction,
        "--config_path", config_path,
        # 添加新参数
        "--new_param", new_value,
    ]
    return cmd, extra_env
```

#### 3. 添加新的 API 端点

编辑 `app.py`：

```python
def do_GET(self):
    # ... 现有路由 ...
    
    if path == "/api/new_endpoint":
        return self._handle_new_endpoint()
    
    # ... 静态文件 ...

def _handle_new_endpoint(self):
    # 实现新端点
    return _json_response(self, {"ok": True, "data": ...})
```

#### 4. 修改阶段识别逻辑

编辑 `log_utils.py` 的 `infer_stage_from_output()` 函数：

```python
def infer_stage_from_output(lines, status):
    # 添加新的关键词识别
    stage_keywords = [
        ("新阶段关键词", "新阶段名称", 0.55),
        # ... 其他关键词 ...
    ]
    # ...
```

## ❓ 常见问题

**Q: 页面显示"空闲"但任务已启动？**

A: 检查终端中 `app.py` 是否有报错。确保 `alphaagent` 命令可用（已安装项目依赖）。

**Q: 如何查看完整日志？**

A: 点击"下载日志"按钮下载 ZIP 文件，或直接在终端运行命令查看详细输出。

**Q: API 调用失败？**

A: 检查 API Key 是否正确，API URL 是否可访问。部分模型可能需要特定的 API URL。

## 📝 API 参考

### POST /api/runs

启动新任务。

**请求体（因子挖掘）**：
```json
{
  "type": "mining",
  "config": {
    "direction": "价量因子挖掘",
    "qlib_data_path": "/path/to/qlib_data",
    "api": {
      "api_key": "sk-xxx",
      "api_url": "https://openrouter.ai/api/v1",
      "model": "deepseek/deepseek-v3.2"
    },
    "num_directions": 10,
    "max_rounds": 11,
    "library_suffix": "exp1"
  }
}
```

**请求体（回测）**：
```json
{
  "type": "backtest",
  "config": {
    "config_path": "/path/to/config.yaml",
    "qlib_data_path": "/path/to/qlib_data",
    "factor_source": "custom",
    "factor_json_paths": ["/path/to/factors.json"],
    "api": { ... },
    "dry_run": false,
    "verbose": false,
    "experiment_name": "exp1"
  }
}
```

**响应**：
```json
{
  "ok": true,
  "run_id": "mining_1234567890_abc123",
  "pid": 12345
}
```

### GET /api/runs/<run_id>

获取任务状态。

**响应**：
```json
{
  "ok": true,
  "run_id": "mining_1234567890_abc123",
  "task_type": "mining",
  "status": "running",
  "stage": {
    "name": "进化中",
    "progress": 0.6,
    "detail": "第 3 轮进化..."
  },
  "output": ["line1", "line2", ...]
}
```

### GET /api/runs/<run_id>/logs.zip

下载任务日志（ZIP 格式）。
