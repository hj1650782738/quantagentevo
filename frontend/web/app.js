/**
 * QuantaAlpha Studio 前端逻辑
 * 
 * 【API 契约依赖】本文件依赖以下 API 接口：
 * - POST /api/runs       启动任务
 * - GET  /api/runs/<id>  获取状态
 * - GET  /api/runs/<id>/logs.zip  下载日志
 * 
 * 如果后端修改了这些接口的返回格式，需要同步修改此文件。
 */

// DOM 元素
const statusBadge = document.getElementById("statusBadge");
const stageName = document.getElementById("stageName");
const stageDetail = document.getElementById("stageDetail");
const progressBar = document.getElementById("progressBar");
const runIdEl = document.getElementById("runId");
const runStatusEl = document.getElementById("runStatus");
const outputContent = document.getElementById("outputContent");
const configSummary = document.getElementById("configSummary");

// 因子挖掘相关
const miningBtn = document.getElementById("miningBtn");
const downloadMiningBtn = document.getElementById("downloadMiningBtn");
const directionInput = document.getElementById("direction");
const qlibDataPathInput = document.getElementById("qlibDataPath");
const apiKeyInput = document.getElementById("apiKey");
const apiUrlInput = document.getElementById("apiUrl");
const modelInput = document.getElementById("model");
const numDirectionsInput = document.getElementById("numDirections");
const maxRoundsInput = document.getElementById("maxRounds");
const librarySuffixInput = document.getElementById("librarySuffix");

// 回测相关
const backtestBtn = document.getElementById("backtestBtn");
const downloadBacktestBtn = document.getElementById("downloadBacktestBtn");
const backtestConfigInput = document.getElementById("backtestConfig");
const backtestQlibPathInput = document.getElementById("backtestQlibPath");
const factorSourceSelect = document.getElementById("factorSource");
const factorJsonPathsInput = document.getElementById("factorJsonPaths");
const backtestApiKeyInput = document.getElementById("backtestApiKey");
const backtestApiUrlInput = document.getElementById("backtestApiUrl");
const backtestModelInput = document.getElementById("backtestModel");
const dryRunInput = document.getElementById("dryRun");
const verboseInput = document.getElementById("verbose");
const experimentNameInput = document.getElementById("experimentName");

// Tab 切换
const tabs = document.querySelectorAll(".tab");
const tabPanels = document.querySelectorAll(".tab-panel");

// 状态
let currentRunId = null;
let currentTaskType = null;
let pollTimer = null;

// ============================================================
// Tab 切换
// ============================================================
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    const targetTab = tab.dataset.tab;
    
    tabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    
    tabPanels.forEach(panel => {
      panel.classList.toggle("active", panel.id === `${targetTab}Panel`);
    });
  });
});

// ============================================================
// UI 更新函数
// ============================================================
function setBadge(status) {
  statusBadge.className = "badge";
  switch (status) {
    case "running":
    case "starting":
      statusBadge.classList.add("badge-running");
      statusBadge.textContent = "运行中";
      break;
    case "done":
      statusBadge.classList.add("badge-done");
      statusBadge.textContent = "完成";
      break;
    case "failed":
      statusBadge.classList.add("badge-failed");
      statusBadge.textContent = "失败";
      break;
    default:
      statusBadge.classList.add("badge-idle");
      statusBadge.textContent = "空闲";
  }
}

function setStage(stage) {
  stageName.textContent = stage?.name || "空闲";
  stageDetail.textContent = stage?.detail || "等待启动...";
  const progress = Math.round((stage?.progress || 0) * 100);
  progressBar.style.width = `${progress}%`;
}

function setOutput(lines) {
  if (lines && lines.length > 0) {
    outputContent.textContent = lines.join("\n");
    // 自动滚动到底部
    outputContent.parentElement.scrollTop = outputContent.parentElement.scrollHeight;
  } else {
    outputContent.textContent = "等待输出...";
  }
}

function setConfigSummary(config) {
  if (!config) {
    configSummary.textContent = "-";
    return;
  }
  const lines = Object.entries(config).map(([k, v]) => `${k}: ${v}`);
  configSummary.textContent = lines.join("\n");
}

function showError(message) {
  alert(`错误: ${message}`);
}

// ============================================================
// API 调用
// ============================================================
async function startMining() {
  const direction = directionInput.value.trim();
  const qlibDataPath = qlibDataPathInput.value.trim();
  const apiKey = apiKeyInput.value.trim();
  const apiUrl = apiUrlInput.value.trim();
  const model = modelInput.value.trim();
  const numDirections = parseInt(numDirectionsInput.value) || 10;
  const maxRounds = parseInt(maxRoundsInput.value) || 11;
  const librarySuffix = librarySuffixInput.value.trim();

  // 验证
  if (!direction) {
    return showError("请输入研究方向");
  }
  if (!qlibDataPath) {
    return showError("请输入 Qlib 数据路径");
  }
  if (!apiKey) {
    return showError("请输入 API Key");
  }

  miningBtn.disabled = true;
  downloadMiningBtn.disabled = true;

  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "mining",
        config: {
          direction,
          qlib_data_path: qlibDataPath,
          api: { api_key: apiKey, api_url: apiUrl, model },
          num_directions: numDirections,
          max_rounds: maxRounds,
          library_suffix: librarySuffix,
        }
      })
    });

    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "启动失败");
    }

    currentRunId = data.run_id;
    currentTaskType = "mining";
    runIdEl.textContent = currentRunId;
    
    setBadge("running");
    setStage({ name: "启动中", progress: 0.05, detail: "正在启动因子挖掘..." });
    setConfigSummary({ 方向: direction, 并行数: numDirections, 轮次: maxRounds });
    
    poll();
  } catch (err) {
    showError(err.message);
    miningBtn.disabled = false;
  }
}

async function startBacktest() {
  const configPath = backtestConfigInput.value.trim();
  const qlibDataPath = backtestQlibPathInput.value.trim();
  const factorSource = factorSourceSelect.value;
  const factorJsonPaths = factorJsonPathsInput.value
    .split("\n")
    .map(s => s.trim())
    .filter(s => s.length > 0);
  const apiKey = backtestApiKeyInput.value.trim();
  const apiUrl = backtestApiUrlInput.value.trim();
  const model = backtestModelInput.value.trim();
  const dryRun = dryRunInput.checked;
  const verbose = verboseInput.checked;
  const experimentName = experimentNameInput.value.trim();

  // 验证
  if (!configPath) {
    return showError("请输入回测配置文件路径");
  }
  if ((factorSource === "custom" || factorSource === "combined") && factorJsonPaths.length === 0) {
    return showError("使用 custom/combined 因子源时必须指定因子库文件");
  }

  backtestBtn.disabled = true;
  downloadBacktestBtn.disabled = true;

  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "backtest",
        config: {
          config_path: configPath,
          qlib_data_path: qlibDataPath,
          factor_source: factorSource,
          factor_json_paths: factorJsonPaths,
          api: { api_key: apiKey, api_url: apiUrl, model },
          dry_run: dryRun,
          verbose,
          experiment_name: experimentName,
        }
      })
    });

    const data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "启动失败");
    }

    currentRunId = data.run_id;
    currentTaskType = "backtest";
    runIdEl.textContent = currentRunId;
    
    setBadge("running");
    setStage({ name: "启动中", progress: 0.05, detail: "正在启动回测..." });
    setConfigSummary({ 配置文件: configPath, 因子源: factorSource });
    
    poll();
  } catch (err) {
    showError(err.message);
    backtestBtn.disabled = false;
  }
}

async function poll() {
  if (!currentRunId) return;
  clearTimeout(pollTimer);

  try {
    const res = await fetch(`/api/runs/${currentRunId}`);
    const data = await res.json();

    if (!data.ok) {
      throw new Error(data.error || "获取状态失败");
    }

    setBadge(data.status);
    setStage(data.stage);
    setOutput(data.output);
    runStatusEl.textContent = data.status;

    // 启用下载按钮
    if (currentTaskType === "mining") {
      downloadMiningBtn.disabled = false;
    } else {
      downloadBacktestBtn.disabled = false;
    }

    // 继续轮询或结束
    if (data.status === "running" || data.status === "starting") {
      pollTimer = setTimeout(poll, 1500);
    } else {
      // 任务结束
      if (currentTaskType === "mining") {
        miningBtn.disabled = false;
      } else {
        backtestBtn.disabled = false;
      }
    }
  } catch (err) {
    stageDetail.textContent = `错误: ${err.message}`;
    pollTimer = setTimeout(poll, 2000);
  }
}

function downloadLogs() {
  if (!currentRunId) return;
  window.location.href = `/api/runs/${currentRunId}/logs.zip`;
}

// ============================================================
// 事件绑定
// ============================================================
miningBtn.addEventListener("click", startMining);
backtestBtn.addEventListener("click", startBacktest);
downloadMiningBtn.addEventListener("click", downloadLogs);
downloadBacktestBtn.addEventListener("click", downloadLogs);

// ============================================================
// 初始化
// ============================================================
setBadge("idle");
setStage({ name: "空闲", progress: 0, detail: "等待启动..." });
