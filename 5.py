import numpy as np
import pickle

# ===================== 配置参数 =====================
GO_ONTOLOGY_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/TALE/data/me/bp_go_1.pickle"
GO2IND_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/TALE/data/me/go2ind_bp.pickle"
# 输出的整数索引版矩阵（覆盖原文件）
OUTPUT_MATRIX_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/TALE/data/me/bp_label_matrix_1_sparse.npy"
NB_CLASSES = 5431

# ===================== 生成整数索引数组 =====================
def rebuild_label_matrix():
    # 1. 加载GO本体和索引映射
    with open(GO_ONTOLOGY_PATH, "rb") as f:
        go_ontology = pickle.load(f)
    with open(GO2IND_PATH, "rb") as f:
        go2ind = pickle.load(f)
    
    # 2. 收集所有有效父节点索引（只保留整数索引）
    valid_indices = []
    for go_term, info in go_ontology.items():
        child_ind = info["ind"]
        # 遍历父节点，收集合法索引
        for father_go in info["father"]:
            if father_go in go2ind:
                father_ind = go2ind[father_go]
                valid_indices.append(father_ind)  # 只存父节点索引（整数）
    
    # 3. 转为int64的二维数组（适配tf.gather的输入格式）
    valid_indices = np.array(valid_indices, dtype=np.int64).reshape(-1, 1)
    # 4. 保存（覆盖原float32矩阵）
    np.save(OUTPUT_MATRIX_PATH, valid_indices)
    
    print("=== 整数索引版label_matrix生成完成 ===")
    print(f"索引数组形状：{valid_indices.shape}")  # 比如(12000, 1)
    print(f"索引范围：{valid_indices.min()} ~ {valid_indices.max()}")  # 0~5430
    print(f"输出路径：{OUTPUT_MATRIX_PATH}")

if __name__ == "__main__":
    rebuild_label_matrix()