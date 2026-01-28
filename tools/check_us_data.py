
import qlib
from qlib.data import D
from qlib.config import REG_US
import pandas as pd

# 初始化 Qlib (US 数据)
provider_uri = "/home/tjxy/.qlib/qlib_data/us_data"
qlib.init(provider_uri=provider_uri, region=REG_US)

print(f"Checking data range in: {provider_uri}")

# 获取所有股票代码
instruments = D.instruments("all")
stock_list = D.list_instruments(instruments=instruments, as_list=True)
print(f"Total instruments: {len(stock_list)}")

if not stock_list:
    print("No instruments found!")
    exit(1)

# 获取日历
calendar = D.calendar(start_time='2010-01-01', end_time='2030-12-31')
if len(calendar) > 0:
    print(f"Calendar range: {calendar[0]} to {calendar[-1]}")
    last_date = calendar[-1]
else:
    print("Calendar is empty!")

# 检查部分股票的实际数据
sample_stocks = stock_list[:5]
print(f"Checking sample stocks: {sample_stocks}")

for stock in sample_stocks:
    df = D.features([stock], ['$close'], start_time='2020-01-01', end_time='2030-12-31')
    if not df.empty:
        last_idx = df.index.get_level_values('datetime').max()
        print(f"Stock {stock} last date: {last_idx}")
    else:
        print(f"Stock {stock} has no data since 2020")
