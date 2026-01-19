#!/usr/bin/env python3
"""
下载 S&P 500 数据并转换为 qlib 兼容格式

使用 akshare 的东财接口下载美股数据，处理空缺值、停牌等情况，
并转换为与 cn_data 对齐的 qlib 二进制格式。

使用方法:
    python download_sp500_data.py --start 2016-01-01 --end 2025-12-26

注意事项:
    - S&P 500 成分股会随时间变化，本脚本会下载当前成分股的历史数据
    - 对于已退市的股票，历史数据仍会保留
    - 空缺值使用 NaN 表示，不会用 0 代替
    - 使用前复权数据 (qfq)
"""

import os
import sys
import time
import argparse
import struct
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================
DEFAULT_OUTPUT_DIR = "/home/tjxy/.qlib/qlib_data/us_data"
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2025-12-26"

# qlib 数据字段映射
# cn_data 有: open, high, low, close, volume, amount, adjclose, vwap, factor, change
# akshare 有: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
FIELD_MAPPING = {
    'open': '开盘',
    'high': '最高',
    'low': '最低',
    'close': '收盘',
    'volume': '成交量',
    'amount': '成交额',
}

# 下载配置
MAX_WORKERS = 4  # 并行下载线程数
RETRY_COUNT = 3  # 重试次数
RETRY_DELAY = 5  # 重试延迟（秒）
REQUEST_DELAY = 0.5  # 请求间隔（秒），避免被封


# ============================================================================
# S&P 500 成分股获取
# ============================================================================
HISTORICAL_CONSTITUENTS_FILE = Path(__file__).parent / "data" / "sp500_historical_constituents.csv"
HISTORICAL_CONSTITUENTS_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"


def download_historical_constituents() -> bool:
    """下载历史成分股数据文件"""
    import urllib.request
    
    if HISTORICAL_CONSTITUENTS_FILE.exists():
        logger.info(f"历史成分股文件已存在: {HISTORICAL_CONSTITUENTS_FILE}")
        return True
    
    logger.info("下载 S&P 500 历史成分股数据...")
    try:
        HISTORICAL_CONSTITUENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HISTORICAL_CONSTITUENTS_URL, HISTORICAL_CONSTITUENTS_FILE)
        logger.info(f"下载完成: {HISTORICAL_CONSTITUENTS_FILE}")
        return True
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


def get_sp500_constituents(start_date: str, end_date: str) -> List[str]:
    """
    获取指定时间范围内所有曾经是 S&P 500 成分股的股票
    
    数据来源: GitHub fja05680/sp500 项目
    包含 1996-2026 年间的每日成分股变动记录
    
    Args:
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        
    Returns:
        股票代码列表（去重）
    """
    # 下载历史数据（如果不存在）
    if not download_historical_constituents():
        logger.warning("无法获取历史成分股数据，使用备用列表")
        return get_backup_sp500_list()
    
    logger.info("加载 S&P 500 历史成分股数据...")
    
    try:
        df = pd.read_csv(HISTORICAL_CONSTITUENTS_FILE)
        df['date'] = pd.to_datetime(df['date'])
        
        # 筛选时间范围
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (df['date'] >= start) & (df['date'] <= end)
        df_filtered = df[mask]
        
        if len(df_filtered) == 0:
            logger.warning(f"在 {start_date} ~ {end_date} 范围内没有找到数据")
            return get_backup_sp500_list()
        
        # 收集所有出现过的股票
        all_tickers = set()
        for tickers_str in df_filtered['tickers']:
            tickers = [t.strip() for t in tickers_str.split(',')]
            all_tickers.update(tickers)
        
        # 处理特殊股票代码 (BRK.B -> BRK-B)
        processed_tickers = []
        for ticker in sorted(all_tickers):
            # 跳过明显无效的股票代码
            if ticker.endswith('Q') and len(ticker) > 4:  # 退市股票 (如 AAMRQ)
                continue
            # 处理特殊字符
            ticker = ticker.replace('.', '-')
            processed_tickers.append(ticker)
        
        logger.info(f"在 {start_date} ~ {end_date} 期间找到 {len(processed_tickers)} 只股票")
        return processed_tickers
        
    except Exception as e:
        logger.error(f"加载历史成分股数据失败: {e}")
        return get_backup_sp500_list()


def get_backup_sp500_list() -> List[str]:
    """
    备用的 S&P 500 核心成分股列表（当前市值前100）
    """
    symbols = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK-B', 'UNH',
        'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
        'LLY', 'PEP', 'COST', 'KO', 'AVGO', 'WMT', 'MCD', 'CSCO', 'TMO', 'ACN',
        'ABT', 'DHR', 'VZ', 'CMCSA', 'NEE', 'PM', 'ADBE', 'TXN', 'NKE', 'WFC',
        'BMY', 'RTX', 'UPS', 'HON', 'ORCL', 'QCOM', 'COP', 'T', 'MS', 'SPGI',
        'BA', 'GS', 'IBM', 'CAT', 'LOW', 'AMGN', 'SBUX', 'BLK', 'DE', 'INTU',
        'INTC', 'GILD', 'AMD', 'AXP', 'MDT', 'ELV', 'MDLZ', 'ISRG', 'ADI', 'CVS',
        'PLD', 'REGN', 'LMT', 'TJX', 'BKNG', 'SYK', 'VRTX', 'ADP', 'TMUS', 'MMC',
        'CB', 'CI', 'SCHW', 'ZTS', 'PNC', 'MO', 'SO', 'DUK', 'EOG', 'CME',
        'NOC', 'CL', 'EQIX', 'USB', 'ITW', 'ETN', 'BDX', 'SLB', 'APD', 'AON',
    ]
    logger.info(f"使用备用列表: {len(symbols)} 只股票")
    return symbols


def find_akshare_symbol(ticker: str) -> Optional[str]:
    """
    将标准股票代码转换为 akshare 格式
    
    akshare 格式: 交易所代码.股票代码
    - 105: NASDAQ
    - 106: NYSE
    - 107: AMEX
    
    Args:
        ticker: 标准股票代码，如 AAPL, MSFT
        
    Returns:
        akshare 格式的代码，如 105.AAPL
    """
    import akshare as ak
    
    # 尝试不同的交易所前缀
    exchanges = ['105', '106', '107']  # NASDAQ, NYSE, AMEX
    
    for exchange in exchanges:
        symbol = f"{exchange}.{ticker}"
        try:
            # 尝试获取少量数据来验证代码是否有效
            df = ak.stock_us_hist(
                symbol=symbol,
                period="daily",
                start_date="20240101",
                end_date="20240110",
                adjust="qfq"
            )
            if df is not None and len(df) > 0:
                return symbol
        except Exception:
            continue
    
    return None


# ============================================================================
# 数据下载
# ============================================================================
def download_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq"
) -> Optional[pd.DataFrame]:
    """
    下载单只股票的历史数据
    
    Args:
        symbol: akshare 格式的股票代码，如 105.AAPL
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        adjust: 复权类型，qfq=前复权
        
    Returns:
        DataFrame with OHLCV data, or None if failed
    """
    import akshare as ak
    
    # 转换日期格式
    start = start_date.replace('-', '')
    end = end_date.replace('-', '')
    
    for attempt in range(RETRY_COUNT):
        try:
            df = ak.stock_us_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust
            )
            
            if df is None or len(df) == 0:
                return None
                
            return df
            
        except Exception as e:
            if attempt < RETRY_COUNT - 1:
                logger.warning(f"{symbol} 下载失败 (尝试 {attempt + 1}/{RETRY_COUNT}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"{symbol} 下载失败: {e}")
                return None
    
    return None


def process_stock_data(
    df: pd.DataFrame,
    ticker: str,
    calendar: List[str]
) -> Optional[pd.DataFrame]:
    """
    处理股票数据，对齐到交易日历
    
    Args:
        df: 原始数据 DataFrame
        ticker: 股票代码
        calendar: 交易日历列表
        
    Returns:
        处理后的 DataFrame，按交易日历对齐
    """
    if df is None or len(df) == 0:
        return None
    
    try:
        # 标准化日期列
        if '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
        else:
            logger.warning(f"{ticker}: 未找到日期列")
            return None
        
        df = df.set_index('date')
        
        # 重命名列
        rename_map = {}
        for qlib_col, ak_col in FIELD_MAPPING.items():
            if ak_col in df.columns:
                rename_map[ak_col] = qlib_col
        
        df = df.rename(columns=rename_map)
        
        # 计算额外字段
        # adjclose: 使用前复权收盘价（已经是前复权了）
        df['adjclose'] = df['close']
        
        # vwap: 成交金额 / 成交量
        if 'amount' in df.columns and 'volume' in df.columns:
            df['vwap'] = np.where(
                df['volume'] > 0,
                df['amount'] / df['volume'],
                df['close']
            )
        else:
            df['vwap'] = df['close']
        
        # factor: 复权因子（前复权数据，设为1）
        df['factor'] = 1.0
        
        # change: 涨跌幅
        if '涨跌幅' in df.columns:
            df['change'] = df['涨跌幅'] / 100.0  # 转换为小数
        else:
            df['change'] = df['close'].pct_change()
        
        # 创建完整的交易日历索引
        calendar_dates = pd.to_datetime(calendar)
        full_index = pd.DatetimeIndex(calendar_dates)
        
        # 只保留需要的列
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                        'adjclose', 'vwap', 'factor', 'change']
        df = df[[c for c in cols_to_keep if c in df.columns]]
        
        # 重新索引到完整日历，空缺值为 NaN
        df = df.reindex(full_index)
        
        # 确保数据类型为 float32（qlib 使用的格式）
        for col in df.columns:
            df[col] = df[col].astype(np.float32)
        
        return df
        
    except Exception as e:
        logger.error(f"{ticker} 处理数据失败: {e}")
        return None


# ============================================================================
# qlib 格式转换
# ============================================================================
def save_to_qlib_format(
    df: pd.DataFrame,
    ticker: str,
    output_dir: Path,
    calendar: List[str]
) -> Tuple[str, str]:
    """
    将数据保存为 qlib 二进制格式
    
    qlib bin 文件格式:
    - 第一个 float32: start_index (数据从日历的第几天开始，0-based)
    - 后续 float32: 实际数据值
    
    Args:
        df: 处理后的 DataFrame，索引为日期
        ticker: 股票代码
        output_dir: 输出目录
        calendar: 交易日历列表
        
    Returns:
        (start_date, end_date) 元组
    """
    # 创建股票目录 (qlib 使用小写路径)
    stock_dir = output_dir / "features" / ticker.lower()
    stock_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据的有效日期范围
    valid_mask = ~df['close'].isna()
    if not valid_mask.any():
        return None, None
    
    first_valid = df.index[valid_mask].min()
    last_valid = df.index[valid_mask].max()
    
    start_date = first_valid.strftime('%Y-%m-%d')
    end_date = last_valid.strftime('%Y-%m-%d')
    
    # 找到第一个有效数据在日历中的索引
    calendar_dates = pd.to_datetime(calendar)
    
    # 计算 start_index: 第一个有效数据对应的日历索引
    try:
        start_index = np.where(calendar_dates == first_valid)[0][0]
    except IndexError:
        # 如果找不到精确匹配，找最近的日期
        start_index = np.searchsorted(calendar_dates, first_valid)
    
    # 只保存从第一个有效数据到最后一个有效数据的范围
    # 找到 last_valid 的索引
    try:
        end_index = np.where(calendar_dates == last_valid)[0][0]
    except IndexError:
        end_index = np.searchsorted(calendar_dates, last_valid)
    
    # 提取这个范围内的数据
    data_range = df.loc[calendar_dates[start_index]:calendar_dates[end_index]]
    
    # 保存每个字段为 .day.bin 文件
    for col in df.columns:
        data = data_range[col].values.astype(np.float32)
        
        # qlib 格式：第一个值是 start_index，后面是数据
        bin_path = stock_dir / f"{col}.day.bin"
        
        with open(bin_path, 'wb') as f:
            # 写入 start_index (float32)
            f.write(struct.pack('<f', float(start_index)))
            # 写入数据
            for val in data:
                f.write(struct.pack('<f', val))
    
    return start_date, end_date


def generate_calendar(start_date: str, end_date: str) -> List[str]:
    """
    生成美股交易日历
    
    排除周末和美国主要股市假日：
    - 新年 (1月1日或调休)
    - 马丁路德金日 (1月第三个周一)
    - 总统日 (2月第三个周一)
    - 耶稣受难日 (复活节前的周五)
    - 阵亡将士纪念日 (5月最后一个周一)
    - 六月节 (6月19日或调休)
    - 独立日 (7月4日或调休)
    - 劳动节 (9月第一个周一)
    - 感恩节 (11月第四个周四)
    - 圣诞节 (12月25日或调休)
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 生成所有日期
    all_dates = pd.date_range(start, end, freq='D')
    
    # 排除周末
    trading_days = all_dates[all_dates.dayofweek < 5]
    
    # 计算美国假日
    us_holidays = set()
    
    for year in range(start.year, end.year + 1):
        # 新年
        new_year = pd.Timestamp(year, 1, 1)
        if new_year.dayofweek == 5:  # 周六
            us_holidays.add(pd.Timestamp(year - 1, 12, 31))
        elif new_year.dayofweek == 6:  # 周日
            us_holidays.add(pd.Timestamp(year, 1, 2))
        else:
            us_holidays.add(new_year)
        
        # 马丁路德金日 (1月第三个周一)
        jan_first = pd.Timestamp(year, 1, 1)
        first_monday = jan_first + pd.Timedelta(days=(7 - jan_first.dayofweek) % 7)
        mlk_day = first_monday + pd.Timedelta(days=14)
        us_holidays.add(mlk_day)
        
        # 总统日 (2月第三个周一)
        feb_first = pd.Timestamp(year, 2, 1)
        first_monday = feb_first + pd.Timedelta(days=(7 - feb_first.dayofweek) % 7)
        presidents_day = first_monday + pd.Timedelta(days=14)
        us_holidays.add(presidents_day)
        
        # 阵亡将士纪念日 (5月最后一个周一)
        may_last = pd.Timestamp(year, 5, 31)
        memorial_day = may_last - pd.Timedelta(days=may_last.dayofweek)
        us_holidays.add(memorial_day)
        
        # 六月节 (6月19日)
        juneteenth = pd.Timestamp(year, 6, 19)
        if juneteenth.dayofweek == 5:
            us_holidays.add(pd.Timestamp(year, 6, 18))
        elif juneteenth.dayofweek == 6:
            us_holidays.add(pd.Timestamp(year, 6, 20))
        else:
            us_holidays.add(juneteenth)
        
        # 独立日 (7月4日)
        independence_day = pd.Timestamp(year, 7, 4)
        if independence_day.dayofweek == 5:
            us_holidays.add(pd.Timestamp(year, 7, 3))
        elif independence_day.dayofweek == 6:
            us_holidays.add(pd.Timestamp(year, 7, 5))
        else:
            us_holidays.add(independence_day)
        
        # 劳动节 (9月第一个周一)
        sep_first = pd.Timestamp(year, 9, 1)
        labor_day = sep_first + pd.Timedelta(days=(7 - sep_first.dayofweek) % 7)
        us_holidays.add(labor_day)
        
        # 感恩节 (11月第四个周四)
        nov_first = pd.Timestamp(year, 11, 1)
        first_thursday = nov_first + pd.Timedelta(days=(3 - nov_first.dayofweek) % 7)
        thanksgiving = first_thursday + pd.Timedelta(days=21)
        us_holidays.add(thanksgiving)
        
        # 圣诞节 (12月25日)
        christmas = pd.Timestamp(year, 12, 25)
        if christmas.dayofweek == 5:
            us_holidays.add(pd.Timestamp(year, 12, 24))
        elif christmas.dayofweek == 6:
            us_holidays.add(pd.Timestamp(year, 12, 26))
        else:
            us_holidays.add(christmas)
    
    # 排除假日
    trading_days = trading_days[~trading_days.isin(us_holidays)]
    
    return [d.strftime('%Y-%m-%d') for d in trading_days]


def save_calendar(calendar: List[str], output_dir: Path):
    """保存交易日历"""
    calendar_dir = output_dir / "calendars"
    calendar_dir.mkdir(parents=True, exist_ok=True)
    
    with open(calendar_dir / "day.txt", 'w') as f:
        for date in calendar:
            f.write(f"{date}\n")
    
    logger.info(f"保存交易日历: {len(calendar)} 个交易日")


def save_instruments(
    instruments: Dict[str, Tuple[str, str]],
    output_dir: Path
):
    """
    保存股票池文件
    
    Args:
        instruments: {ticker: (start_date, end_date)} 字典
        output_dir: 输出目录
    """
    inst_dir = output_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 all.txt（所有股票）
    with open(inst_dir / "all.txt", 'w') as f:
        for ticker, (start, end) in sorted(instruments.items()):
            if start and end:
                f.write(f"{ticker}\t{start}\t{end}\n")
    
    # 保存 sp500.txt（S&P 500 成分股）- 与 all.txt 相同
    with open(inst_dir / "sp500.txt", 'w') as f:
        for ticker, (start, end) in sorted(instruments.items()):
            if start and end:
                f.write(f"{ticker}\t{start}\t{end}\n")
    
    logger.info(f"保存股票池: {len(instruments)} 只股票")


def generate_sp500_membership(start_date: str, end_date: str, output_dir: Path):
    """
    生成 S&P 500 成员资格文件
    记录每只股票在 S&P 500 中的确切时间段
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
    """
    if not HISTORICAL_CONSTITUENTS_FILE.exists():
        logger.warning("历史成分股文件不存在，跳过成员资格生成")
        return
    
    logger.info("生成 S&P 500 成员资格记录...")
    
    df = pd.read_csv(HISTORICAL_CONSTITUENTS_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df['date'] >= start) & (df['date'] <= end)
    df_filtered = df[mask].sort_values('date')
    
    if len(df_filtered) == 0:
        return
    
    # 跟踪每只股票的成员资格时间段
    membership = {}  # ticker -> [(start, end), ...]
    current_members = set()
    
    for _, row in df_filtered.iterrows():
        date = row['date']
        tickers = set(t.strip().replace('.', '-') for t in row['tickers'].split(','))
        
        # 新加入的股票
        for ticker in tickers - current_members:
            if ticker not in membership:
                membership[ticker] = []
            membership[ticker].append([date, None])
        
        # 退出的股票
        for ticker in current_members - tickers:
            if ticker in membership and membership[ticker]:
                membership[ticker][-1][1] = date
        
        current_members = tickers
    
    # 关闭仍在列表中的股票的时间段
    last_date = df_filtered['date'].max()
    for ticker in current_members:
        if ticker in membership and membership[ticker] and membership[ticker][-1][1] is None:
            membership[ticker][-1][1] = last_date
    
    # 保存成员资格文件
    inst_dir = output_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    
    with open(inst_dir / "sp500_membership.txt", 'w') as f:
        f.write("# S&P 500 Membership History\n")
        f.write("# Format: ticker\tstart_date\tend_date\n")
        for ticker, periods in sorted(membership.items()):
            for start_dt, end_dt in periods:
                if start_dt and end_dt:
                    f.write(f"{ticker.lower()}\t{start_dt.strftime('%Y-%m-%d')}\t{end_dt.strftime('%Y-%m-%d')}\n")
    
    logger.info(f"保存 S&P 500 成员资格记录: {len(membership)} 只股票")


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="下载 S&P 500 数据并转换为 qlib 格式")
    parser.add_argument('--start', default=DEFAULT_START_DATE, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default=DEFAULT_END_DATE, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='输出目录')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='并行下载线程数')
    parser.add_argument('--symbols', nargs='*', help='指定下载的股票代码（用于测试）')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("S&P 500 数据下载器")
    logger.info(f"日期范围: {args.start} ~ {args.end}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    # 生成交易日历
    calendar = generate_calendar(args.start, args.end)
    save_calendar(calendar, output_dir)
    
    # 获取 S&P 500 成分股
    if args.symbols:
        tickers = args.symbols
        logger.info(f"使用指定的 {len(tickers)} 只股票")
    else:
        tickers = get_sp500_constituents(args.start, args.end)
    
    logger.info(f"准备下载 {len(tickers)} 只股票")
    
    # 查找 akshare 格式的股票代码
    import akshare as ak
    
    symbol_map = {}  # ticker -> akshare_symbol
    failed_symbols = []
    
    logger.info("查找 akshare 股票代码...")
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] 查找 {ticker}...")
        ak_symbol = find_akshare_symbol(ticker)
        if ak_symbol:
            symbol_map[ticker] = ak_symbol
            logger.info(f"  -> {ak_symbol}")
        else:
            failed_symbols.append(ticker)
            logger.warning(f"  -> 未找到")
        time.sleep(REQUEST_DELAY)
    
    logger.info(f"成功找到 {len(symbol_map)} 只股票的代码")
    if failed_symbols:
        logger.warning(f"未找到 {len(failed_symbols)} 只股票: {failed_symbols[:10]}...")
    
    # 下载和处理数据
    instruments = {}
    success_count = 0
    fail_count = 0
    
    for i, (ticker, ak_symbol) in enumerate(symbol_map.items()):
        logger.info(f"[{i+1}/{len(symbol_map)}] 下载 {ticker} ({ak_symbol})...")
        
        # 下载数据
        df = download_stock_data(ak_symbol, args.start, args.end)
        
        if df is None:
            fail_count += 1
            logger.warning(f"  {ticker} 下载失败")
            continue
        
        # 处理数据
        processed_df = process_stock_data(df, ticker, calendar)
        
        if processed_df is None:
            fail_count += 1
            logger.warning(f"  {ticker} 处理失败")
            continue
        
        # 保存为 qlib 格式
        start_date, end_date = save_to_qlib_format(
            processed_df, ticker, output_dir, calendar
        )
        
        if start_date and end_date:
            # qlib 使用小写股票代码
            instruments[ticker.lower()] = (start_date, end_date)
            success_count += 1
            logger.info(f"  {ticker} 完成: {start_date} ~ {end_date}, {len(df)} 条记录")
        else:
            fail_count += 1
            logger.warning(f"  {ticker} 无有效数据")
        
        # 请求延迟
        time.sleep(REQUEST_DELAY)
    
    # 保存股票池
    save_instruments(instruments, output_dir)
    
    # 生成 S&P 500 成员资格记录
    generate_sp500_membership(args.start, args.end, output_dir)
    
    # 汇总
    logger.info("=" * 60)
    logger.info("下载完成!")
    logger.info(f"成功: {success_count} 只股票")
    logger.info(f"失败: {fail_count} 只股票")
    logger.info(f"数据目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

