import numpy as np
OUTPUT_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/TALE/data/me/bp_label_regular_1.npy"
# 生成和GO术语数一致的全0数组
regular_array = np.zeros((5431,), dtype=np.float32)
np.save(OUTPUT_PATH, regular_array)
print("生成完成：", OUTPUT_PATH)