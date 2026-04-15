import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 配置过滤参数（核心修改：阈值改为3.0） =====================
# 原文件路径
species_config = {
    "Ath": "/misc/hard_disk/others_res/zhangdy/wangluo/atted_unzip/Ath/atted_network/Ath_coexpr_edgelist.txt",
    "Gma": "/misc/hard_disk/others_res/zhangdy/wangluo/atted_unzip/Gma/atted_network/Gma_coexpr_edgelist.txt",
    "Osa": "/misc/hard_disk/others_res/zhangdy/wangluo/atted_unzip/Osa/atted_network/Osa_coexpr_edgelist.txt",
    "Tae": "/misc/hard_disk/others_res/zhangdy/wangluo/atted_unzip/Tae/atted_network/Tae_coexpr_edgelist.txt",
    "Zma": "/misc/hard_disk/others_res/zhangdy/wangluo/atted_unzip/Zma/atted_network/Zma_coexpr_edgelist.txt"
}

# 过滤后文件保存路径（自动创建）
output_root = "/misc/hard_disk/others_res/zhangdy/wangluo/filtered_edges/"
os.makedirs(output_root, exist_ok=True)

# 核心修改：ATTED推荐阈值，从4.0改为3.0
Z_THRESHOLD = 3.0

# ===================== 2. 分块过滤大文件（逻辑不变） =====================
def filter_z_score(species_name, input_path, output_path):
    print(f"\n开始过滤 {species_name}：保留z-score>{Z_THRESHOLD}的边")
    # 分块读取（每块100万行），避免内存溢出
    chunk_size = 1000000
    filtered_chunks = []
    total_original = 0
    total_filtered = 0

    for chunk in pd.read_csv(input_path, sep="\t", header=None, names=["gene1", "gene2", "z_score"], 
                             chunksize=chunk_size):
        # 强制z-score为数值，过滤NaN
        chunk["z_score"] = pd.to_numeric(chunk["z_score"], errors="coerce")
        chunk = chunk.dropna(subset=["z_score"])
        
        # 统计原始行数
        total_original += len(chunk)
        # 过滤z-score>3的边（核心修改后的条件）
        chunk_filtered = chunk[chunk["z_score"] > Z_THRESHOLD]
        total_filtered += len(chunk_filtered)
        filtered_chunks.append(chunk_filtered)

    # 合并并保存过滤后的文件
    final_df = pd.concat(filtered_chunks, ignore_index=True)
    final_df.to_csv(output_path, sep="\t", header=False, index=False)
    
    # 输出过滤统计
    filter_ratio = round((total_original - total_filtered)/total_original*100, 2)
    print(f"✅ {species_name} 过滤完成：")
    print(f"   - 原始边数：{total_original}")
    print(f"   - 过滤后边数：{total_filtered}")
    print(f"   - 剔除噪声边占比：{filter_ratio}%")
    print(f"   - 保存路径：{output_path}")
    return final_df

# ===================== 3. 批量过滤所有物种（逻辑不变） =====================
if __name__ == "__main__":
    for species, input_path in species_config.items():
        output_path = os.path.join(output_root, f"{species}_filtered_edges.txt")
        filter_z_score(species, input_path, output_path)
    print(f"\n🚀 所有物种过滤完成！过滤后文件保存在：{output_root}")