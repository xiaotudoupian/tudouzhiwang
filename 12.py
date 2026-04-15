import pickle
import numpy as np

# 文件路径
pkl_path = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/go_ic_paper_standard_clean.pkl"

# 加载
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# ===================== 自动识别类型并打印 =====================
print("="*50)
print("📦 文件类型:", type(data))
print("="*50)

# 如果是字典
if isinstance(data, dict):
    print(f"🔑 字典长度: {len(data)}")
    print("\n🔍 前 10 个 key:")
    keys = list(data.keys())[:10]
    for k in keys:
        print(f"  {k}  ->  {data[k]}")

# 如果是列表
elif isinstance(data, list):
    print(f"📄 列表长度: {len(data)}")
    print("\n🔍 前 5 个元素:")
    for i, item in enumerate(data[:5]):
        print(f"  [{i}] {item}")

# 如果是 numpy 数组
elif isinstance(data, np.ndarray):
    print(f"📊 Numpy 数组形状: {data.shape}")
    print(f"📊 数据类型: {data.dtype}")

# 其他
else:
    print("📄 内容:", data)

print("\n✅ 查看完成！")