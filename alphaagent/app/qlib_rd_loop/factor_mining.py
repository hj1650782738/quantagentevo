"""
Factor workflow with session control and evolution support.

Supports three round phases:
- Original: Initial exploration in each direction
- Mutation: Orthogonal exploration from parent trajectories
- Crossover: Hybrid strategies from multiple parents
"""

from typing import Any
from pathlib import Path
import fire
import signal
import sys
import threading
from multiprocessing import Process
from functools import wraps
import time
import ctypes
import os
from alphaagent.app.qlib_rd_loop.conf import ALPHA_AGENT_FACTOR_PROP_SETTING
from alphaagent.app.qlib_rd_loop.planning import generate_parallel_directions
from alphaagent.app.qlib_rd_loop.planning import load_run_config
from alphaagent.components.workflow.alphaagent_loop import AlphaAgentLoop
from alphaagent.components.evolution import (
    EvolutionController, 
    EvolutionConfig,
    StrategyTrajectory,
    RoundPhase,
)
from alphaagent.core.exception import FactorEmptyError
from alphaagent.log import logger
from alphaagent.log.time import measure_time
from alphaagent.oai.llm_conf import LLM_SETTINGS




def force_timeout():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 优先选择timeout参数
            seconds = LLM_SETTINGS.factor_mining_timeout
            def handle_timeout(signum, frame):
                logger.error(f"强制终止程序执行，已超过{seconds}秒")
                sys.exit(1)

            # 设置信号处理器
            signal.signal(signal.SIGALRM, handle_timeout)
            # 设置闹钟
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # 取消闹钟
                signal.alarm(0)
            return result
        return wrapper
    return decorator


def _run_branch(
    direction: str | None,
    step_n: int,
    use_local: bool,
    idx: int,
    log_root: str,
    log_prefix: str,
):
    if log_root:
        branch_name = f"{log_prefix}_{idx:02d}"
        branch_log = Path(log_root) / branch_name
        branch_log.mkdir(parents=True, exist_ok=True)
        logger.set_trace_path(branch_log)
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=None,
        use_local=use_local,
    )
    model_loop.user_initial_direction = direction
    model_loop.run(step_n=step_n, stop_event=None)


def _run_evolution_task(
    task: dict[str, Any],
    directions: list[str],
    step_n: int,
    use_local: bool,
    user_direction: str | None,
    log_root: str,
    stop_event: threading.Event | None,
) -> dict[str, Any]:
    """
    运行单个进化任务（一轮小流程）。
    
    Args:
        task: 进化任务描述
        directions: 原始方向列表
        step_n: 每轮步数
        use_local: 是否使用本地回测
        user_direction: 用户初始方向
        log_root: 日志根目录
        stop_event: 停止事件
        
    Returns:
        包含轨迹数据的字典
    """
    phase = task["phase"]
    direction_id = task["direction_id"]
    strategy_suffix = task.get("strategy_suffix", "")
    round_idx = task["round_idx"]
    parent_trajectories = task.get("parent_trajectories", [])
    
    # 根据阶段确定方向
    if phase == RoundPhase.ORIGINAL:
        direction = directions[direction_id] if direction_id < len(directions) else None
    elif phase == RoundPhase.MUTATION:
        # 变异轮使用原始方向，但附加策略后缀
        direction = directions[direction_id] if direction_id < len(directions) else None
    else:  # CROSSOVER
        # 交叉轮使用混合方向
        direction = None
    
    # 生成轨迹ID
    trajectory_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase)
    parent_ids = [p.trajectory_id for p in parent_trajectories]
    
    # 设置日志目录
    if log_root:
        branch_name = f"{phase.value}_{round_idx:02d}_{direction_id:02d}"
        branch_log = Path(log_root) / branch_name
        branch_log.mkdir(parents=True, exist_ok=True)
        logger.set_trace_path(branch_log)
    
    logger.info(f"开始进化任务: phase={phase.value}, round={round_idx}, direction={direction_id}")
    
    # 创建并运行循环
    model_loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=stop_event,
        use_local=use_local,
        strategy_suffix=strategy_suffix,
        evolution_phase=phase.value,
        trajectory_id=trajectory_id,
        parent_trajectory_ids=parent_ids,
        direction_id=direction_id,
    )
    model_loop.user_initial_direction = user_direction
    
    # 运行一轮小流程（5步）
    model_loop.run(step_n=step_n, stop_event=stop_event)
    
    # 获取轨迹数据
    traj_data = model_loop._get_trajectory_data()
    traj_data["task"] = task
    
    return traj_data


def run_evolution_loop(
    initial_direction: str | None,
    evolution_cfg: dict[str, Any],
    exec_cfg: dict[str, Any],
    planning_cfg: dict[str, Any],
    stop_event: threading.Event | None = None,
):
    """
    运行进化循环：原始轮 → 变异轮 → 交叉轮 → 变异轮 → ...
    
    Args:
        initial_direction: 用户初始方向
        evolution_cfg: 进化配置
        exec_cfg: 执行配置
        planning_cfg: 规划配置
        stop_event: 停止事件
    """
    # 解析配置
    num_directions = int(planning_cfg.get("num_directions", 2))
    max_rounds = int(evolution_cfg.get("max_rounds", 10))
    crossover_size = int(evolution_cfg.get("crossover_size", 2))
    crossover_n = int(evolution_cfg.get("crossover_n", 3))
    steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
    use_local = bool(exec_cfg.get("use_local", True))
    log_root = exec_cfg.get("branch_log_root") or "log"
    
    # 生成初始方向
    prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
    prompt_path = Path(__file__).parent / str(prompt_file)
    
    if initial_direction:
        directions = generate_parallel_directions(
            initial_direction=initial_direction,
            n=num_directions,
            prompt_file=prompt_path,
            max_attempts=int(planning_cfg.get("max_attempts", 5)),
            use_llm=bool(planning_cfg.get("use_llm", True)),
            allow_fallback=bool(planning_cfg.get("allow_fallback", True)),
        )
    else:
        directions = [None] * num_directions
    
    logger.info(f"生成了 {len(directions)} 个探索方向")
    for i, d in enumerate(directions):
        logger.info(f"  方向 {i}: {d}")
    
    # 创建进化控制器
    pool_save_path = Path(log_root) / "trajectory_pool.json"
    mutation_prompt_path = Path(__file__).parent / "evolution_prompts.yaml"
    
    config = EvolutionConfig(
        num_directions=len(directions),
        steps_per_loop=steps_per_loop,
        max_rounds=max_rounds,
        crossover_size=crossover_size,
        crossover_n=crossover_n,
        prefer_diverse_crossover=True,
        pool_save_path=str(pool_save_path),
        mutation_prompt_path=str(mutation_prompt_path) if mutation_prompt_path.exists() else None,
        crossover_prompt_path=str(mutation_prompt_path) if mutation_prompt_path.exists() else None,
    )
    
    controller = EvolutionController(config)
    
    # 进化主循环
    logger.info("="*60)
    logger.info("开始进化循环")
    logger.info(f"配置: directions={len(directions)}, max_rounds={max_rounds}, "
               f"crossover_size={crossover_size}, crossover_n={crossover_n}")
    logger.info("="*60)
    
    while not controller.is_complete():
        if stop_event and stop_event.is_set():
            logger.info("收到停止信号，终止进化循环")
            break
        
        # 获取下一个任务
        task = controller.get_next_task()
        if task is None:
            logger.info("进化完成：没有更多任务")
            break
        
        logger.info(f"执行任务: phase={task['phase'].value}, round={task['round_idx']}, "
                   f"direction={task['direction_id']}")
        
        try:
            # 运行任务
            traj_data = _run_evolution_task(
                task=task,
                directions=directions,
                step_n=steps_per_loop,
                use_local=use_local,
                user_direction=initial_direction,
                log_root=log_root,
                stop_event=stop_event,
            )
            
            # 创建轨迹并报告完成
            trajectory = controller.create_trajectory_from_loop_result(
                task=task,
                hypothesis=traj_data.get("hypothesis"),
                experiment=traj_data.get("experiment"),
                feedback=traj_data.get("feedback"),
            )
            
            controller.report_task_complete(task, trajectory)
            
            logger.info(f"任务完成: trajectory_id={trajectory.trajectory_id}, "
                       f"RankIC={trajectory.get_primary_metric()}")
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 继续下一个任务
            continue
    
    # 保存最终状态
    state_path = Path(log_root) / "evolution_state.json"
    controller.save_state(state_path)
    
    # 输出最佳结果
    best_trajs = controller.get_best_trajectories(top_n=5)
    logger.info("="*60)
    logger.info(f"进化完成！最佳轨迹 (Top {len(best_trajs)}):")
    for i, t in enumerate(best_trajs):
        logger.info(f"  {i+1}. {t.trajectory_id}: phase={t.phase.value}, "
                   f"RankIC={t.get_primary_metric():.4f}")
    logger.info(f"轨迹池统计: {controller.pool.get_statistics()}")
    logger.info("="*60)


@force_timeout()
def main(path=None, step_n=100, direction=None, stop_event=None, config_path=None, evolution_mode=None):
    """
    Autonomous alpha factor mining with optional evolution support.

    Args:
        path: 会话路径（用于恢复）
        step_n: 步骤数，默认100（20个循环 * 5个步骤/循环）
        direction: 初始方向
        stop_event: 停止事件
        config_path: 运行配置文件路径
        evolution_mode: 是否启用进化模式（None=使用配置，True/False=覆盖配置）

    进化模式流程：
        原始轮 → 变异轮 → 交叉轮 → 变异轮 → 交叉轮 → ...

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor_alphaagent.py $LOG_PATH/__session__/1/0_propose  --step_n 1  --potential_direction "[Initial Direction (Optional)]"

    """
    try:
        config_default = Path(__file__).parent / "run_config.yaml"
        config_file = Path(config_path) if config_path else config_default
        run_cfg = load_run_config(config_file)
        planning_cfg = (run_cfg.get("planning") or {}) if isinstance(run_cfg, dict) else {}
        exec_cfg = (run_cfg.get("execution") or {}) if isinstance(run_cfg, dict) else {}
        evolution_cfg = (run_cfg.get("evolution") or {}) if isinstance(run_cfg, dict) else {}

        # 确定是否使用进化模式
        if evolution_mode is not None:
            use_evolution = evolution_mode
        else:
            use_evolution = bool(evolution_cfg.get("enabled", False))

        if step_n is None or step_n == 100:
            if exec_cfg.get("step_n") is not None:
                step_n = exec_cfg.get("step_n")
            else:
                max_loops = int(exec_cfg.get("max_loops", 10))
                steps_per_loop = int(exec_cfg.get("steps_per_loop", 5))
                step_n = max_loops * steps_per_loop

        use_local = os.getenv("USE_LOCAL", "True").lower()
        use_local = True if use_local in ["true", "1"] else False
        if exec_cfg.get("use_local") is not None:
            use_local = bool(exec_cfg.get("use_local"))
        exec_cfg["use_local"] = use_local
        
        logger.info(f"Use {'Local' if use_local else 'Docker container'} to execute factor backtest")
        
        # 进化模式
        if use_evolution and path is None:
            logger.info("="*60)
            logger.info("启用进化模式: 原始轮 → 变异轮 → 交叉轮 循环")
            logger.info("="*60)
            
            run_evolution_loop(
                initial_direction=direction,
                evolution_cfg=evolution_cfg,
                exec_cfg=exec_cfg,
                planning_cfg=planning_cfg,
                stop_event=stop_event,
            )
        
        # 传统模式（无进化）
        elif path is None:
            planning_enabled = bool(planning_cfg.get("enabled", False))
            n_dirs = int(planning_cfg.get("num_directions", 1))
            max_attempts = int(planning_cfg.get("max_attempts", 5))
            use_llm = bool(planning_cfg.get("use_llm", True))
            allow_fallback = bool(planning_cfg.get("allow_fallback", True))
            prompt_file = planning_cfg.get("prompt_file") or "planning_prompts.yaml"
            prompt_path = Path(__file__).parent / str(prompt_file)
            if planning_enabled and direction:
                directions = generate_parallel_directions(
                    initial_direction=direction,
                    n=n_dirs,
                    prompt_file=prompt_path,
                    max_attempts=max_attempts,
                    use_llm=use_llm,
                    allow_fallback=allow_fallback,
                )
            else:
                directions = [direction] if direction else [None]

            log_root = exec_cfg.get("branch_log_root") or "log"
            log_prefix = exec_cfg.get("branch_log_prefix") or "branch"
            use_branch_logs = planning_enabled and len(directions) > 1
            parallel_execution = bool(exec_cfg.get("parallel_execution", False))

            if parallel_execution and len(directions) > 1:
                procs: list[Process] = []
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}")
                    p = Process(
                        target=_run_branch,
                        args=(dir_text, step_n, use_local, idx, log_root if use_branch_logs else "", log_prefix),
                    )
                    p.start()
                    procs.append(p)
                for p in procs:
                    p.join()
            else:
                for idx, dir_text in enumerate(directions, start=1):
                    if dir_text:
                        logger.info(f"[Planning] Branch {idx}/{len(directions)} direction: {dir_text}")
                    if use_branch_logs:
                        branch_name = f"{log_prefix}_{idx:02d}"
                        branch_log = Path(log_root) / branch_name
                        branch_log.mkdir(parents=True, exist_ok=True)
                        logger.set_trace_path(branch_log)
                    model_loop = AlphaAgentLoop(
                        ALPHA_AGENT_FACTOR_PROP_SETTING,
                        potential_direction=dir_text,
                        stop_event=stop_event,
                        use_local=use_local,
                    )
                    model_loop.user_initial_direction = direction
                    model_loop.run(step_n=step_n, stop_event=stop_event)
        else:
            model_loop = AlphaAgentLoop.load(path, use_local=use_local)
            model_loop.run(step_n=step_n, stop_event=stop_event)
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise
    finally:
        logger.info("程序执行完成或被终止")

if __name__ == "__main__":
    fire.Fire(main)
