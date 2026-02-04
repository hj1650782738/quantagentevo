#!/usr/bin/env python3
"""
展示所有生成的因子（表格格式）
使用方法: python3 show_all_factors.py
"""

import pickle
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd

def list_all_experiments():
    """列出所有可用的实验"""
    # 搜索新旧两个日志目录
    log_dirs = [
        Path("/mnt/DATA/quantagent/AlphaAgent/log"),  # 新路径
        Path("/home/tjxy/quantagent/AlphaAgent/log"),  # 旧路径（兼容历史数据）
    ]
    exps = []
    for log_dir in log_dirs:
        if log_dir.exists():
            exps.extend([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("2026-")])
    exps = sorted(exps, key=lambda x: x.name, reverse=True)
    
    if not exps:
        print("❌ 未找到实验目录")
        return []
    
    print(f"\n{'='*150}")
    print(f"{'所有可用实验列表':^150}")
    print(f"{'='*150}\n")
    print(f"共找到 {len(exps)} 个实验\n")
    
    # 打印表头
    header = f"{'序号':<6} | {'实验ID':<40} | {'创建时间':<20} | {'有因子':<8} | {'有SOTA':<8} | {'初始方向':<50}"
    print(header)
    print("-" * 150)
    
    exp_list = []
    for i, exp_dir in enumerate(exps, 1):
        exp_id = exp_dir.name
        mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # 检查是否有因子
        factor_dir = exp_dir / "r" / "experiment generation"
        has_factors = "✅" if factor_dir.exists() and list(factor_dir.rglob("*.pkl")) else "❌"
        
        # 检查是否有SOTA因子和初始方向
        has_sota = "❌"
        initial_direction = "未提供"
        session_dir = exp_dir / "__session__"
        if session_dir.exists():
            session_files = sorted(session_dir.glob("*/*_*"), 
                                  key=lambda f: (int(f.parent.name), int(f.name.split("_")[0])))
            if session_files:
                try:
                    session = pickle.load(open(session_files[0], 'rb'))  # 使用第一个session文件获取初始方向
                    
                    # 提取初始方向
                    if hasattr(session, 'hypothesis_generator'):
                        hg = session.hypothesis_generator
                        if hasattr(hg, 'potential_direction') and hg.potential_direction:
                            initial_direction = hg.potential_direction
                    
                    # 检查SOTA（使用最新的session文件）
                    if len(session_files) > 0:
                        latest_session = pickle.load(open(session_files[-1], 'rb'))
                        if hasattr(latest_session, 'trace') and hasattr(latest_session.trace, 'get_sota_hypothesis_and_experiment'):
                            sota_hyp, sota_exp = latest_session.trace.get_sota_hypothesis_and_experiment()
                            if sota_hyp and sota_exp:
                                has_sota = "✅"
                except Exception as e:
                    # 如果读取失败，继续处理下一个实验
                    pass
        
        exp_list.append({
            '序号': i,
            '实验ID': exp_id,
            '创建时间': mtime,
            '有因子': has_factors,
            '有SOTA': has_sota,
            '初始方向': initial_direction
        })
        
        # 如果实验ID太长，截断显示
        display_id = exp_id
        if len(display_id) > 38:
            display_id = display_id[:35] + "..."
        
        # 如果初始方向太长，截断显示
        display_direction = initial_direction
        if len(display_direction) > 48:
            display_direction = display_direction[:45] + "..."
        
        row = f"{i:<6} | {display_id:<40} | {mtime:<20} | {has_factors:<8} | {has_sota:<8} | {display_direction:<50}"
        print(row)
    
    print(f"\n{'='*150}\n")
    print("💡 使用方法:")
    print("   python3 show_all_factors.py --exp <实验ID> [其他选项]")
    print("   例如: python3 show_all_factors.py --exp 2026-01-04_11-39-17-817865 --sota")
    print()
    
    return exp_list

def get_latest_experiment(exp_id=None):
    """获取实验目录
    
    Args:
        exp_id (str, optional): 实验ID（如 "2026-01-04_11-39-17-817865"），如果指定则返回该实验，否则返回最新的实验
    
    Returns:
        Path: 实验目录路径
    """
    # 搜索新旧两个日志目录
    log_dirs = [
        Path("/mnt/DATA/quantagent/AlphaAgent/log"),  # 新路径
        Path("/home/tjxy/quantagent/AlphaAgent/log"),  # 旧路径（兼容历史数据）
    ]
    
    # 如果指定了实验ID，在所有日志目录中查找
    if exp_id:
        for log_dir in log_dirs:
        exp_dir = log_dir / exp_id
        if exp_dir.exists() and exp_dir.is_dir():
            return exp_dir
            print(f"❌ 实验目录不存在: {exp_id}")
        print(f"   已搜索路径:")
        for log_dir in log_dirs:
            print(f"     - {log_dir / exp_id}")
            print(f"\n💡 提示: 使用 --list 参数查看所有可用实验")
            sys.exit(1)
    
    # 否则返回最新的实验目录（从所有日志目录中查找）
    exps = []
    for log_dir in log_dirs:
        if log_dir.exists():
            exps.extend([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("2026-")])
    exps = sorted(exps, key=lambda x: x.name, reverse=True)
    
    if not exps:
        print("❌ 未找到实验目录")
        sys.exit(1)
    
    # 优先返回包含因子的实验目录
    for exp_dir in exps:
        factor_dir = exp_dir / "r" / "experiment generation"
        if factor_dir.exists():
            pkl_files = list(factor_dir.rglob("*.pkl"))
            if pkl_files:
                return exp_dir
    
    # 如果没有找到包含因子的，返回最新的
    return exps[0]

def extract_all_factors(exp_dir):
    """从实验目录中提取所有因子"""
    factor_dir = exp_dir / "r" / "experiment generation"
    
    # 检查目录是否存在
    if not factor_dir.exists():
        print(f"❌ 因子目录不存在: {factor_dir}")
        return []
    
    # 查找所有包含因子的子目录
    subdirs = [d for d in factor_dir.iterdir() if d.is_dir()]
    if not subdirs:
        # 如果没有子目录，直接在factor_dir中查找pkl文件
        pkl_files = sorted(factor_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime)
        if not pkl_files:
            print(f"⚠️  未找到因子文件: {factor_dir}")
            return []
        subdirs = [factor_dir]
    
    all_factors = []
    for subdir in subdirs:
        pkl_files = sorted(subdir.glob("*.pkl"), key=lambda x: x.stat().st_mtime)
        for pkl_file in pkl_files:
            try:
                data = pickle.load(open(pkl_file, 'rb'))
                if isinstance(data, list):
                    for idx, factor_task in enumerate(data):
                        factor_info = {
                            '序号': len(all_factors) + 1,
                            '因子名称': factor_task.factor_name,
                            '文件': pkl_file.name,
                            '生成时间': datetime.fromtimestamp(pkl_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            '批次': pkl_file.stem,
                            '因子对象': factor_task  # 保存完整的因子对象以便后续提取详细信息
                        }
                        
                        # 提取表达式（优先使用 factor_expression）
                        if hasattr(factor_task, 'factor_expression') and factor_task.factor_expression:
                            factor_info['表达式'] = factor_task.factor_expression
                        elif hasattr(factor_task, 'expression') and factor_task.expression:
                            factor_info['表达式'] = factor_task.expression
                        
                        # 提取因子描述
                        if hasattr(factor_task, 'factor_description') and factor_task.factor_description:
                            factor_info['描述'] = factor_task.factor_description
                        
                        # 提取因子公式（LaTeX格式）
                        if hasattr(factor_task, 'factor_formulation') and factor_task.factor_formulation:
                            factor_info['公式'] = factor_task.factor_formulation
                        
                        # 提取变量说明
                        if hasattr(factor_task, 'variables') and factor_task.variables:
                            factor_info['变量'] = factor_task.variables
                        
                        all_factors.append(factor_info)
            except Exception as e:
                print(f"⚠️  读取文件 {pkl_file} 时出错: {e}", file=sys.stderr)
    
    return all_factors

def print_factors_table(factors):
    """以表格格式打印因子"""
    if not factors:
        print("❌ 未找到任何因子")
        return
    
    print(f"\n{'='*120}")
    print(f"{'所有生成的因子列表':^120}")
    print(f"{'='*120}\n")
    print(f"共找到 {len(factors)} 个因子\n")
    
    # 打印表头
    header = f"{'序号':<6} | {'因子名称':<45} | {'生成时间':<20} | {'文件':<30}"
    print(header)
    print("-" * 120)
    
    # 打印每一行
    for factor in factors:
        name = factor['因子名称']
        if len(name) > 43:
            name = name[:40] + "..."
        
        file_name = factor['文件']
        if len(file_name) > 28:
            file_name = file_name[:25] + "..."
        
        row = f"{factor['序号']:<6} | {name:<45} | {factor['生成时间']:<20} | {file_name:<30}"
        print(row)
    
    print(f"\n{'='*120}\n")
    
    # 询问是否显示详细表达式
    print("💡 提示: 要查看因子的详细表达式，可以使用以下命令:")
    print("   python3 show_all_factors.py --detail")
    print("   或")
    print("   python3 show_all_factors.py --detail --name <因子名称>")

def print_factors_table_with_expression(factors, filter_name=None):
    """以表格格式打印因子（包含表达式和详细信息）"""
    if not factors:
        print("❌ 未找到任何因子")
        return
    
    if filter_name:
        factors = [f for f in factors if filter_name.lower() in f['因子名称'].lower()]
        if not factors:
            print(f"❌ 未找到包含 '{filter_name}' 的因子")
            return
    
    print(f"\n{'='*150}")
    print(f"{'所有生成的因子列表（含详细信息）':^150}")
    print(f"{'='*150}\n")
    print(f"共显示 {len(factors)} 个因子\n")
    
    for factor in factors:
        print(f"{'='*150}")
        print(f"序号: {factor['序号']}")
        print(f"因子名称: {factor['因子名称']}")
        print(f"生成时间: {factor['生成时间']}")
        print(f"文件: {factor['文件']}")
        print()
        
        # 显示因子描述
        if '描述' in factor and factor['描述']:
            print(f"📝 因子描述:")
            desc = factor['描述']
            # 如果描述太长，适当换行
            if len(desc) > 120:
                words = desc.split()
                line = ""
                for word in words:
                    if len(line + word) > 120:
                        print(f"   {line}")
                        line = word + " "
                    else:
                        line += word + " "
                if line:
                    print(f"   {line}")
            else:
                print(f"   {desc}")
            print()
        
        # 显示因子公式（LaTeX格式）
        if '公式' in factor and factor['公式']:
            print(f"📐 因子公式（LaTeX）:")
            formula = factor['公式']
            # 如果公式太长，适当换行
            if len(formula) > 120:
                # 尝试在 \\ 处换行
                parts = formula.split('\\\\')
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        print(f"   {part}\\\\")
                    else:
                        print(f"   {part}")
            else:
                print(f"   {formula}")
            print()
        
        # 显示因子表达式
        if '表达式' in factor and factor['表达式']:
            print(f"💻 因子表达式:")
            expr = factor['表达式']
            # 如果表达式太长，适当换行
            if len(expr) > 120:
                # 在括号或逗号后适当换行
                import re
                # 在较长的函数调用后换行
                parts = re.split(r'([(),])', expr)
                line = ""
                for part in parts:
                    if len(line + part) > 120 and line:
                        print(f"   {line}")
                        line = part
                    else:
                        line += part
                if line:
                    print(f"   {line}")
            else:
                print(f"   {expr}")
            print()
        
        # 显示变量说明
        if '变量' in factor and factor['变量']:
            print(f"📚 变量说明:")
            variables = factor['变量']
            for var_name, var_desc in variables.items():
                print(f"   {var_name}: {var_desc}")
            print()
        
        print()

def load_session(exp_dir):
    """加载session对象"""
    session_dir = exp_dir / "__session__"
    if not session_dir.exists():
        return None
    
    # 查找最新的session文件
    session_files = sorted(session_dir.glob("*/*_*"), 
                          key=lambda f: (int(f.parent.name), int(f.name.split("_")[0])))
    if not session_files:
        return None
    
    try:
        latest_session = session_files[-1]
        with latest_session.open("rb") as f:
            session = pickle.load(f)
        return session
    except Exception as e:
        print(f"⚠️  加载session失败: {e}", file=sys.stderr)
        return None

def show_memory_bank(exp_dir):
    """显示记忆库（KnowledgeBase）"""
    session = load_session(exp_dir)
    if session is None:
        print("❌ 无法加载session")
        return
    
    if hasattr(session, 'trace') and hasattr(session.trace, 'knowledge_base'):
        kb = session.trace.knowledge_base
        if kb:
            print(f"\n{'='*150}")
            print(f"{'记忆库（KnowledgeBase）':^150}")
            print(f"{'='*150}\n")
            
            # 显示knowledge_base的所有属性
            kb_attrs = {k: v for k, v in kb.__dict__.items() if k != 'path'}
            if kb_attrs:
                for key, value in kb_attrs.items():
                    print(f"📚 {key}:")
                    if isinstance(value, (list, dict)):
                        print(f"   {type(value).__name__}，包含 {len(value)} 项")
                        if isinstance(value, list) and len(value) > 0:
                            print(f"   示例: {str(value[0])[:200]}...")
                    else:
                        value_str = str(value)
                        if len(value_str) > 200:
                            value_str = value_str[:200] + "..."
                        print(f"   {value_str}")
                    print()
            else:
                print("   记忆库为空")
        else:
            print("⚠️  记忆库未初始化")
    else:
        print("⚠️  未找到记忆库")

def show_hypotheses(exp_dir):
    """显示所有假设"""
    hyp_dir = exp_dir / "r" / "hypothesis generation"
    if not hyp_dir.exists():
        print("❌ 假设目录不存在")
        return
    
    hyp_files = sorted(hyp_dir.rglob("*.pkl"), key=lambda x: x.stat().st_mtime)
    if not hyp_files:
        print("⚠️  未找到假设文件")
        return
    
    print(f"\n{'='*150}")
    print(f"{'所有假设列表':^150}")
    print(f"{'='*150}\n")
    print(f"共找到 {len(hyp_files)} 个假设\n")
    
    for i, hyp_file in enumerate(hyp_files, 1):
        try:
            data = pickle.load(open(hyp_file, 'rb'))
            # 假设可能是Hypothesis对象或包含hypothesis属性的对象
            if hasattr(data, 'hypothesis'):
                hypothesis = data.hypothesis
            elif hasattr(data, '__class__') and 'Hypothesis' in str(type(data)):
                hypothesis = data
            else:
                hypothesis = str(data)
            
            print(f"{'='*150}")
            print(f"假设 {i}")
            print(f"文件: {hyp_file.name}")
            print(f"时间: {datetime.fromtimestamp(hyp_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\n假设内容:")
            hyp_str = str(hypothesis)
            if len(hyp_str) > 500:
                hyp_str = hyp_str[:500] + "..."
            print(f"   {hyp_str}")
            print()
        except Exception as e:
            print(f"⚠️  读取假设文件 {hyp_file} 时出错: {e}", file=sys.stderr)

def show_feedbacks(exp_dir):
    """显示所有反馈"""
    fb_dir = exp_dir / "ef" / "feedback"
    if not fb_dir.exists():
        print("❌ 反馈目录不存在")
        return
    
    fb_files = sorted(fb_dir.rglob("*.pkl"), key=lambda x: x.stat().st_mtime)
    if not fb_files:
        print("⚠️  未找到反馈文件")
        return
    
    print(f"\n{'='*150}")
    print(f"{'所有反馈列表':^150}")
    print(f"{'='*150}\n")
    print(f"共找到 {len(fb_files)} 个反馈\n")
    
    for i, fb_file in enumerate(fb_files, 1):
        try:
            feedback = pickle.load(open(fb_file, 'rb'))
            
            print(f"{'='*150}")
            print(f"反馈 {i}")
            print(f"文件: {fb_file.name}")
            print(f"时间: {datetime.fromtimestamp(fb_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            if hasattr(feedback, 'observations'):
                obs = str(feedback.observations)
                if len(obs) > 300:
                    obs = obs[:300] + "..."
                print(f"📊 观察结果:")
                print(f"   {obs}")
                print()
            
            if hasattr(feedback, 'hypothesis_evaluation'):
                eval_str = str(feedback.hypothesis_evaluation)
                if len(eval_str) > 300:
                    eval_str = eval_str[:300] + "..."
                print(f"📈 假设评估:")
                print(f"   {eval_str}")
                print()
            
            if hasattr(feedback, 'new_hypothesis'):
                new_hyp = str(feedback.new_hypothesis)
                if len(new_hyp) > 300:
                    new_hyp = new_hyp[:300] + "..."
                print(f"💡 新假设:")
                print(f"   {new_hyp}")
                print()
            
            if hasattr(feedback, 'decision'):
                decision_str = "✅ 成功" if feedback.decision else "❌ 失败"
                print(f"🎯 决策: {decision_str}")
                print()
            
            if hasattr(feedback, 'reason'):
                reason = str(feedback.reason)
                if len(reason) > 300:
                    reason = reason[:300] + "..."
                print(f"📝 原因:")
                print(f"   {reason}")
                print()
            
            print()
        except Exception as e:
            print(f"⚠️  读取反馈文件 {fb_file} 时出错: {e}", file=sys.stderr)

def get_all_workspace_dirs():
    """
    动态发现所有工作空间目录
    支持 QuantaAlpha_workspace 和 QuantaAlpha_workspace_{EXPERIMENT_ID} 格式
    """
    workspace_base_dirs = [
        Path("/mnt/DATA/quantagent/QuantaAlpha"),  # 新路径基础目录
        Path("/home/tjxy/quantagent/AlphaAgent/git_ignore_folder"),  # 旧路径基础目录
    ]
    
    workspace_dirs = []
    for base_dir in workspace_base_dirs:
        if not base_dir.exists():
            continue
        # 查找所有匹配 QuantaAlpha_workspace* 的目录
        for ws_dir in base_dir.iterdir():
            if ws_dir.is_dir() and ws_dir.name.startswith("QuantaAlpha_workspace"):
                workspace_dirs.append(ws_dir)
    
    return workspace_dirs


def show_backtest_results(exp_dir):
    """显示回测结果"""
    # 动态发现所有工作空间目录（支持 EXPERIMENT_ID 隔离）
    workspace_dirs = get_all_workspace_dirs()
    
    if not workspace_dirs:
        print("❌ 回测工作空间目录不存在")
        return
    
    # 查找所有有回测结果的工作空间（从所有 workspace 目录中搜索）
    workspaces = []
    for workspace_dir in workspace_dirs:
        for ws_dir in workspace_dir.iterdir():
            if ws_dir.is_dir():
                csv_file = ws_dir / "qlib_res.csv"
                if csv_file.exists():
                    workspaces.append((ws_dir, csv_file.stat().st_mtime))
    
    if not workspaces:
        print("⚠️  未找到回测结果")
        return
    
    # 按时间排序
    workspaces.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*150}")
    print(f"{'回测结果列表':^150}")
    print(f"{'='*150}\n")
    print(f"共找到 {len(workspaces)} 个回测结果\n")
    
    for i, (ws_dir, mtime) in enumerate(workspaces[:10], 1):  # 只显示最新10个
        csv_file = ws_dir / "qlib_res.csv"
        print(f"{'='*150}")
        print(f"回测结果 {i}")
        print(f"工作空间ID: {ws_dir.name}")
        print(f"时间: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            df = pd.read_csv(csv_file, index_col=0)
            print("📊 回测指标:")
            # 显示关键指标
            key_metrics = ['Rank IC', 'IC', 'ICIR', 'Rank ICIR', 
                          '1day.excess_return_without_cost.annualized_return',
                          '1day.excess_return_without_cost.information_ratio',
                          '1day.excess_return_without_cost.max_drawdown']
            for metric in key_metrics:
                if metric in df.index:
                    value = df.loc[metric, '0']
                    print(f"   {metric}: {value}")
            print()
        except Exception as e:
            print(f"⚠️  读取回测结果失败: {e}", file=sys.stderr)
    
    if len(workspaces) > 10:
        print(f"\n... 还有 {len(workspaces) - 10} 个回测结果未显示")

def show_experiment_history(exp_dir):
    """显示实验历史（假设-回测-反馈的完整流程）"""
    session = load_session(exp_dir)
    if session is None:
        print("❌ 无法加载session，无法显示完整历史")
        return
    
    if not (hasattr(session, 'trace') and hasattr(session.trace, 'hist')):
        print("⚠️  未找到实验历史")
        return
    
    hist = session.trace.hist
    if not hist:
        print("⚠️  实验历史为空")
        return
    
    print(f"\n{'='*150}")
    print(f"{'实验历史（假设-回测-反馈）':^150}")
    print(f"{'='*150}\n")
    print(f"共 {len(hist)} 轮实验\n")
    
    for i, (hypothesis, experiment, feedback) in enumerate(hist, 1):
        print(f"{'='*150}")
        print(f"轮次 {i}")
        print()
        
        # 显示假设
        print(f"💡 假设:")
        hyp_str = str(hypothesis)
        if len(hyp_str) > 400:
            hyp_str = hyp_str[:400] + "..."
        print(f"   {hyp_str}")
        print()
        
        # 显示回测结果
        if hasattr(experiment, 'running_info') and hasattr(experiment.running_info, 'result'):
            result = experiment.running_info.result
            if result is not None:
                print(f"📊 回测结果:")
                if isinstance(result, pd.DataFrame):
                    # 显示关键指标
                    key_metrics = ['Rank IC', 'IC', 'ICIR', 'Rank ICIR']
                    for metric in key_metrics:
                        if metric in result.index:
                            value = result.loc[metric].iloc[0] if len(result.columns) > 0 else result.loc[metric]
                            print(f"   {metric}: {value}")
                else:
                    print(f"   {str(result)[:300]}")
                print()
        
        # 显示反馈
        print(f"📝 反馈:")
        if hasattr(feedback, 'observations'):
            obs = str(feedback.observations)
            if len(obs) > 300:
                obs = obs[:300] + "..."
            print(f"   观察: {obs}")
        
        if hasattr(feedback, 'decision'):
            decision_str = "✅ 成功" if feedback.decision else "❌ 失败"
            print(f"   决策: {decision_str}")
        
        if hasattr(feedback, 'hypothesis_evaluation'):
            eval_str = str(feedback.hypothesis_evaluation)
            if len(eval_str) > 300:
                eval_str = eval_str[:300] + "..."
            print(f"   评估: {eval_str}")
        print()
        print()

def get_experiment_result(experiment):
    """从experiment对象中提取回测结果"""
    # 尝试多种方式获取结果
    if hasattr(experiment, 'result') and experiment.result is not None:
        return experiment.result
    if hasattr(experiment, 'running_info') and hasattr(experiment.running_info, 'result'):
        return experiment.running_info.result
    return None

def extract_ic_from_result(result):
    """从回测结果中提取IC相关指标"""
    ic_info = {}
    if result is None:
        return ic_info
    
    # 处理Series类型
    if isinstance(result, pd.Series):
        # 尝试不同的索引名称
        for idx_name in ['Rank IC', 'rank_ic', 'RankIC']:
            if idx_name in result.index:
                value = result.loc[idx_name]
                ic_info['Rank IC'] = float(value) if pd.notna(value) else None
                break
        
        for idx_name in ['IC', 'ic']:
            if idx_name in result.index:
                value = result.loc[idx_name]
                ic_info['IC'] = float(value) if pd.notna(value) else None
                break
        
        for idx_name in ['ICIR', 'icir']:
            if idx_name in result.index:
                value = result.loc[idx_name]
                ic_info['ICIR'] = float(value) if pd.notna(value) else None
                break
        
        for idx_name in ['Rank ICIR', 'rank_icir', 'RankICIR']:
            if idx_name in result.index:
                value = result.loc[idx_name]
                ic_info['Rank ICIR'] = float(value) if pd.notna(value) else None
                break
    
    # 处理DataFrame类型
    elif isinstance(result, pd.DataFrame):
        # 尝试不同的索引名称
        for idx_name in ['Rank IC', 'rank_ic', 'RankIC']:
            if idx_name in result.index:
                value = result.loc[idx_name].iloc[0] if len(result.columns) > 0 else result.loc[idx_name]
                ic_info['Rank IC'] = float(value) if pd.notna(value) else None
                break
        
        for idx_name in ['IC', 'ic']:
            if idx_name in result.index:
                value = result.loc[idx_name].iloc[0] if len(result.columns) > 0 else result.loc[idx_name]
                ic_info['IC'] = float(value) if pd.notna(value) else None
                break
        
        for idx_name in ['ICIR', 'icir']:
            if idx_name in result.index:
                value = result.loc[idx_name].iloc[0] if len(result.columns) > 0 else result.loc[idx_name]
                ic_info['ICIR'] = float(value) if pd.notna(value) else None
                break
        
        for idx_name in ['Rank ICIR', 'rank_icir', 'RankICIR']:
            if idx_name in result.index:
                value = result.loc[idx_name].iloc[0] if len(result.columns) > 0 else result.loc[idx_name]
                ic_info['Rank ICIR'] = float(value) if pd.notna(value) else None
                break
    
    return ic_info

def judge_factor_quality(rank_ic, icir=None, max_correlation=None, factor_workspace_path=None):
    """
    判断因子质量（新标准）
    
    新标准:
    - RankIC > 0.01（降低要求，从0.015改为0.01）
    - 与Alpha158因子库的最大相关性 < 0.7
    
    旧标准（已废弃）:
    - RankIC > 0.02 且 ICIR > 0.3
    
    Args:
        rank_ic: RankIC值
        icir: ICIR值（保留兼容性，但不再使用）
        max_correlation: 与Alpha158的最大相关性（如果已计算）
        factor_workspace_path: 因子工作空间路径（用于计算相关性）
        
    Returns:
        质量等级字符串
    """
    if rank_ic is None:
        return "Unknown"
    
    try:
        rank_ic_val = float(rank_ic)
        
        # 新标准：RankIC > 0.01 且与Alpha158相关性 < 0.7
        if max_correlation is not None:
            corr_val = abs(float(max_correlation))
            if rank_ic_val > 0.01 and corr_val < 0.7:
                return "High-Quality"
            elif rank_ic_val > 0:
                return "Valid"
            else:
                return "Poor"
        else:
            # 如果相关性未提供，只根据RankIC判断（降级处理）
            # 尝试从factor_quality_evaluator计算相关性
            try:
                from factor_quality_evaluator import judge_factor_quality_new
                quality, info = judge_factor_quality_new(
                    rank_ic, 
                    max_correlation=None,
                    factor_workspace_path=factor_workspace_path
                )
                return quality
            except:
                # 如果无法计算相关性，只根据RankIC判断
                if rank_ic_val > 0.01:
                    return "Valid"  # 降级为Valid，因为无法验证相关性
                elif rank_ic_val > 0:
                    return "Valid"
                else:
                    return "Poor"
    except:
        return "Unknown"

def display_sota_factors(exp_dir):
    """显示SOTA因子"""
    session = load_session(exp_dir)
    if session is None:
        print("❌ 无法加载session")
        return
    
    if not (hasattr(session, 'trace') and hasattr(session.trace, 'get_sota_hypothesis_and_experiment')):
        print("⚠️  无法获取SOTA信息")
        return
    
    sota_hyp, sota_exp = session.trace.get_sota_hypothesis_and_experiment()
    if not (sota_hyp and sota_exp):
        print("⚠️  未找到SOTA因子")
        return
    
    print(f"\n{'='*150}")
    print(f"{'SOTA因子列表':^150}")
    print(f"{'='*150}\n")
    
    # 获取SOTA实验的回测结果
    sota_result = get_experiment_result(sota_exp)
    ic_info = extract_ic_from_result(sota_result)
    
    # 显示SOTA假设
    print(f"💡 SOTA假设:")
    hyp_str = str(sota_hyp)
    if len(hyp_str) > 400:
        hyp_str = hyp_str[:400] + "..."
    print(f"   {hyp_str}")
    print()
    
    # 显示SOTA因子
    if hasattr(sota_exp, 'sub_tasks'):
        print(f"📊 SOTA因子（共 {len(sota_exp.sub_tasks)} 个）:\n")
        for i, task in enumerate(sota_exp.sub_tasks, 1):
            print(f"{'='*150}")
            print(f"SOTA因子 {i}")
            print()
            
            if hasattr(task, 'factor_name'):
                print(f"因子名称: {task.factor_name}")
            
            if hasattr(task, 'factor_expression'):
                expr = task.factor_expression
                if len(expr) > 120:
                    expr = expr[:117] + "..."
                print(f"表达式: {expr}")
            
            if hasattr(task, 'factor_description'):
                desc = task.factor_description
                if len(desc) > 300:
                    desc = desc[:297] + "..."
                print(f"描述: {desc}")
            
            # 显示IC信息（如果可用）
            if ic_info:
                print(f"\n📈 回测指标:")
                for key, value in ic_info.items():
                    if value is not None:
                        print(f"   {key}: {value:.6f}")
            
            print()
    else:
        print("⚠️  SOTA实验中没有因子任务")

def display_factor_ic(exp_dir):
    """显示所有因子的IC信息"""
    session = load_session(exp_dir)
    if session is None:
        print("❌ 无法加载session")
        return
    
    if not (hasattr(session, 'trace') and hasattr(session.trace, 'hist')):
        print("⚠️  未找到实验历史")
        return
    
    hist = session.trace.hist
    if not hist:
        print("⚠️  实验历史为空")
        return
    
    print(f"\n{'='*150}")
    print(f"{'因子IC信息汇总':^150}")
    print(f"{'='*150}\n")
    
    all_factors_with_ic = []
    
    for i, (hypothesis, experiment, feedback) in enumerate(hist, 1):
        result = get_experiment_result(experiment)
        ic_info = extract_ic_from_result(result)
        
        if hasattr(experiment, 'sub_tasks'):
            for task in experiment.sub_tasks:
                if hasattr(task, 'factor_name'):
                    factor_info = {
                        '轮次': i,
                        '因子名称': task.factor_name,
                        'Rank IC': ic_info.get('Rank IC'),
                        'IC': ic_info.get('IC'),
                        'ICIR': ic_info.get('ICIR'),
                        'Rank ICIR': ic_info.get('Rank ICIR'),
                    }
                    if hasattr(task, 'factor_expression'):
                        factor_info['表达式'] = task.factor_expression
                    all_factors_with_ic.append(factor_info)
    
    if not all_factors_with_ic:
        print("⚠️  未找到因子IC信息")
        return
    
    print(f"共找到 {len(all_factors_with_ic)} 个因子的IC信息\n")
    
    # 打印表头
    header = f"{'轮次':<6} | {'因子名称':<50} | {'Rank IC':<12} | {'IC':<12} | {'ICIR':<12} | {'Rank ICIR':<12}"
    print(header)
    print("-" * 150)
    
    # 打印每一行
    for factor in all_factors_with_ic:
        name = factor['因子名称']
        if len(name) > 48:
            name = name[:45] + "..."
        
        rank_ic = f"{factor['Rank IC']:.6f}" if factor['Rank IC'] is not None else "N/A"
        ic = f"{factor['IC']:.6f}" if factor['IC'] is not None else "N/A"
        icir = f"{factor['ICIR']:.6f}" if factor['ICIR'] is not None else "N/A"
        rank_icir = f"{factor['Rank ICIR']:.6f}" if factor['Rank ICIR'] is not None else "N/A"
        
        row = f"{factor['轮次']:<6} | {name:<50} | {rank_ic:<12} | {ic:<12} | {icir:<12} | {rank_icir:<12}"
        print(row)
    
    print(f"\n{'='*150}\n")

def display_factor_quality(exp_dir):
    """显示因子质量分类"""
    session = load_session(exp_dir)
    if session is None:
        print("❌ 无法加载session")
        return
    
    if not (hasattr(session, 'trace') and hasattr(session.trace, 'hist')):
        print("⚠️  未找到实验历史")
        return
    
    hist = session.trace.hist
    if not hist:
        print("⚠️  实验历史为空")
        return
    
    print(f"\n{'='*150}")
    print(f"{'因子质量分类汇总':^150}")
    print(f"{'='*150}\n")
    
    all_factors = []
    sota_factors = []
    
    for i, (hypothesis, experiment, feedback) in enumerate(hist, 1):
        result = get_experiment_result(experiment)
        ic_info = extract_ic_from_result(result)
        
        rank_ic = ic_info.get('Rank IC')
        icir = ic_info.get('ICIR')
        quality = judge_factor_quality(rank_ic, icir)
        
        if hasattr(experiment, 'sub_tasks'):
            for task in experiment.sub_tasks:
                if hasattr(task, 'factor_name'):
                    factor_info = {
                        '轮次': i,
                        '因子名称': task.factor_name,
                        '质量': quality,
                        '是否SOTA': feedback.decision if hasattr(feedback, 'decision') else False,
                        'Rank IC': rank_ic,
                        'ICIR': icir,
                    }
                    if hasattr(task, 'factor_expression'):
                        factor_info['表达式'] = task.factor_expression
                    all_factors.append(factor_info)
                    
                    if factor_info['是否SOTA']:
                        sota_factors.append(factor_info)
    
    if not all_factors:
        print("⚠️  未找到因子信息")
        return
    
    # 按质量分类
    high_quality = [f for f in all_factors if f['质量'] == 'High-Quality']
    valid = [f for f in all_factors if f['质量'] == 'Valid']
    poor = [f for f in all_factors if f['质量'] == 'Poor']
    unknown = [f for f in all_factors if f['质量'] == 'Unknown']
    
    # 显示统计信息
    print(f"📊 统计信息:")
    print(f"   总因子数: {len(all_factors)}")
    print(f"   SOTA因子数: {len(sota_factors)}")
    print(f"   High-Quality因子数: {len(high_quality)}")
    print(f"   Valid因子数: {len(valid)}")
    print(f"   Poor因子数: {len(poor)}")
    print(f"   Unknown因子数: {len(unknown)}")
    print()
    
    # 显示SOTA因子
    if sota_factors:
        print(f"{'='*150}")
        print(f"🏆 SOTA因子列表（共 {len(sota_factors)} 个）")
        print(f"{'='*150}\n")
        for i, factor in enumerate(sota_factors, 1):
            print(f"{i}. {factor['因子名称']}")
            print(f"   轮次: {factor['轮次']}")
            if factor['Rank IC'] is not None:
                print(f"   Rank IC: {factor['Rank IC']:.6f}")
            if factor['ICIR'] is not None:
                print(f"   ICIR: {factor['ICIR']:.6f}")
            print(f"   质量: {factor['质量']}")
            print()
    
    # 显示High-Quality因子
    if high_quality:
        print(f"{'='*150}")
        print(f"⭐ High-Quality因子列表（共 {len(high_quality)} 个）")
        print(f"{'='*150}\n")
        for i, factor in enumerate(high_quality, 1):
            print(f"{i}. {factor['因子名称']}")
            print(f"   轮次: {factor['轮次']}")
            if factor['Rank IC'] is not None:
                print(f"   Rank IC: {factor['Rank IC']:.6f}")
            if factor['ICIR'] is not None:
                print(f"   ICIR: {factor['ICIR']:.6f}")
            print()
    
    # 显示所有因子质量表格
    print(f"{'='*150}")
    print(f"📋 所有因子质量分类")
    print(f"{'='*150}\n")
    
    # 打印表头
    header = f"{'轮次':<6} | {'因子名称':<50} | {'质量':<15} | {'SOTA':<6} | {'Rank IC':<12} | {'ICIR':<12}"
    print(header)
    print("-" * 150)
    
    # 打印每一行
    for factor in all_factors:
        name = factor['因子名称']
        if len(name) > 48:
            name = name[:45] + "..."
        
        quality = factor['质量']
        sota = "✅" if factor['是否SOTA'] else "❌"
        rank_ic = f"{factor['Rank IC']:.6f}" if factor['Rank IC'] is not None else "N/A"
        icir = f"{factor['ICIR']:.6f}" if factor['ICIR'] is not None else "N/A"
        
        row = f"{factor['轮次']:<6} | {name:<50} | {quality:<15} | {sota:<6} | {rank_ic:<12} | {icir:<12}"
        print(row)
    
    print(f"\n{'='*150}\n")
    
    # 质量判断标准说明
    print("💡 质量判断标准:")
    print("   High-Quality: Rank IC > 0.01 且与Alpha158相关性 < 0.7")
    print("   Valid: Rank IC > 0")
    print("   Poor: Rank IC <= 0")
    print()
    print("   注意: 新标准已更新，降低RankIC要求并加入相关性检查")
    print("   Unknown: 缺少IC或ICIR数据")
    print()

def main():
    """主函数"""
    import argparse

    #     # 1. 先列出所有实验，找到想查看的实验ID
    # python3 show_all_factors.py --list

    # # 2. 使用实验ID查看特定实验的结果
    # python3 show_all_factors.py --exp 2026-01-04_11-39-17-817865 --sota

    # # 3. 如果不指定--exp，默认查看最新实验
    # python3 show_all_factors.py --sota

    
    parser = argparse.ArgumentParser(description='展示所有生成的因子及相关信息')
    parser.add_argument('--detail', action='store_true', help='显示详细表达式')
    parser.add_argument('--name', type=str, help='按因子名称过滤')
    parser.add_argument('--memory', action='store_true', help='显示记忆库')
    parser.add_argument('--hypotheses', action='store_true', help='显示所有假设')
    parser.add_argument('--feedbacks', action='store_true', help='显示所有反馈')
    parser.add_argument('--backtests', action='store_true', help='显示回测结果')
    parser.add_argument('--history', action='store_true', help='显示实验历史（假设-回测-反馈）')
    parser.add_argument('--sota', action='store_true', help='显示SOTA因子')
    parser.add_argument('--ic', action='store_true', help='显示因子IC信息')
    parser.add_argument('--quality', action='store_true', help='显示因子质量分类')
    parser.add_argument('--exp', '--experiment', type=str, dest='exp_id', help='指定实验ID（如: 2026-01-04_11-39-17-817865）')
    parser.add_argument('--list', action='store_true', help='列出所有可用实验')
    parser.add_argument('--all', action='store_true', help='显示所有信息')
    args = parser.parse_args()
    
    # 如果指定了--list，只列出实验列表
    if args.list:
        list_all_experiments()
        return
    
    # 获取实验目录（如果指定了实验ID则使用指定的，否则使用最新的）
    exp_dir = get_latest_experiment(args.exp_id)
    print(f"📁 实验目录: {exp_dir.name}")
    
    # 如果指定了--all，显示所有信息
    if args.all:
        args.memory = True
        args.hypotheses = True
        args.feedbacks = True
        args.backtests = True
        args.history = True
        args.sota = True
        args.ic = True
        args.quality = True
    
    # 显示记忆库
    if args.memory:
        show_memory_bank(exp_dir)
    
    # 显示假设
    if args.hypotheses:
        show_hypotheses(exp_dir)
    
    # 显示反馈
    if args.feedbacks:
        show_feedbacks(exp_dir)
    
    # 显示回测结果
    if args.backtests:
        show_backtest_results(exp_dir)
    
    # 显示实验历史
    if args.history:
        show_experiment_history(exp_dir)
    
    # 显示SOTA因子
    if args.sota:
        display_sota_factors(exp_dir)
    
    # 显示因子IC信息
    if args.ic:
        display_factor_ic(exp_dir)
    
    # 显示因子质量分类
    if args.quality:
        display_factor_quality(exp_dir)
    
    # 如果没有指定任何特殊选项，显示因子
    if not any([args.memory, args.hypotheses, args.feedbacks, args.backtests, args.history, args.sota, args.ic, args.quality]):
        # 提取因子
        factors = extract_all_factors(exp_dir)
        
        # 打印表格
        if args.detail:
            print_factors_table_with_expression(factors, args.name)
        else:
            print_factors_table(factors)

if __name__ == "__main__":
    main()

