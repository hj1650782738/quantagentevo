import qlib
from qlib.data import D
import time

print("初始化 Qlib...")
qlib.init(provider_uri="/home/tjxy/.qlib/qlib_data/cn_data", region="cn")

print("获取 csi300 股票列表...")
codes = D.instruments("csi300")
print(f"股票数量: {len(codes) if hasattr(codes, '__len__') else 'unknown'}")

print("测试 D.features (小范围)...")
start = time.time()
fields = ["$close", "$open", "$volume", "$change", "$factor"]

try:
    df = D.features(
        codes,
        fields,
        start_time="2024-01-01",
        end_time="2024-06-30",
        freq="day",
    )
    print(f"✓ 成功! 耗时: {time.time() - start:.2f}秒, 数据形状: {df.shape}")
except Exception as e:
    print(f"✗ 失败: {e}")

print("\n测试 benchmark SH000905...")
start = time.time()
try:
    df_bench = D.features(
        ["SH000905"],
        ["$close"],
        start_time="2024-01-01",
        end_time="2024-06-30",
        freq="day",
    )
    print(f"✓ SH000905 成功! 耗时: {time.time() - start:.2f}秒, 数据形状: {df_bench.shape}")
except Exception as e:
    print(f"✗ SH000905 失败: {e}")

print("\n完成!")
