import qlib
from qlib.data import D
import pandas as pd

print("初始化 Qlib...")
qlib.init(provider_uri="/home/tjxy/.qlib/qlib_data/cn_data", region="cn")

print("\n测试不同时间范围的 instruments:")

# 测试1: 默认
result1 = D.instruments("csi300")
print(f"1. D.instruments('csi300'): 类型={type(result1)}, 长度={len(result1) if hasattr(result1, '__len__') else 'N/A'}")
if hasattr(result1, '__iter__') and len(result1) < 10:
    print(f"   内容: {list(result1)}")

# 测试2: 指定时间范围
result2 = D.instruments("csi300", start_time="2024-01-01", end_time="2024-06-30")
print(f"2. D.instruments('csi300', 2024-01-01~2024-06-30): 类型={type(result2)}, 长度={len(result2) if hasattr(result2, '__len__') else 'N/A'}")
if hasattr(result2, '__iter__') and len(result2) < 10:
    print(f"   内容: {list(result2)}")

# 测试3: 更早的时间范围
result3 = D.instruments("csi300", start_time="2020-01-01", end_time="2020-12-31")
print(f"3. D.instruments('csi300', 2020-01-01~2020-12-31): 类型={type(result3)}, 长度={len(result3) if hasattr(result3, '__len__') else 'N/A'}")

# 测试4: 手动读取文件
print("\n手动读取 csi300.txt:")
df = pd.read_csv("/home/tjxy/.qlib/qlib_data/cn_data/instruments/csi300.txt", 
                 sep="\t", header=None, names=["instrument", "start", "end"])
print(f"文件中的股票数: {len(df)}")
print(f"start_time 范围: {df['start'].min()} ~ {df['start'].max()}")
print(f"end_time 范围: {df['end'].min()} ~ {df['end'].max()}")

# 检查在 2024 年有多少只股票
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
valid_2024 = df[(df['start'] <= pd.Timestamp('2024-01-01')) & (df['end'] >= pd.Timestamp('2024-12-31'))]
print(f"在 2024 年有效的股票数: {len(valid_2024)}")
