import os

# ===================== 配置文件路径（直接使用你的路径） =====================
FASTA_C_PATH = r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\total_C_sequences.fasta"
FASTA_F_PATH = r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\total_F_sequences.fasta"

# ===================== 核心统计函数 =====================
def count_fasta_sequences(fasta_path: str) -> int:
    """
    统计FASTA文件中的蛋白质序列数量（以>开头的有效行计数）
    :param fasta_path: FASTA文件路径
    :return: 有效序列数量
    """
    # 检查文件是否存在
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"❌ 文件不存在：{fasta_path}")
    
    seq_count = 0
    with open(fasta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 仅统计以>开头的非空行（有效序列ID行）
            if line.startswith(">") and len(line) > 1:
                seq_count += 1
    
    return seq_count

# ===================== 主流程 =====================
if __name__ == "__main__":
    print("="*60)
    print("📊 FASTA文件蛋白质序列数量统计")
    print("="*60)
    
    # 统计total_C_sequences.fasta
    try:
        c_seq_count = count_fasta_sequences(FASTA_C_PATH)
        print(f"📁 total_C_sequences.fasta | 蛋白质序列数量：{c_seq_count}")
    except Exception as e:
        print(f"❌ 统计total_C_sequences.fasta失败：{str(e)}")
    
    # 统计total_F_sequences.fasta
    try:
        f_seq_count = count_fasta_sequences(FASTA_F_PATH)
        print(f"📁 total_F_sequences.fasta | 蛋白质序列数量：{f_seq_count}")
    except Exception as e:
        print(f"❌ 统计total_F_sequences.fasta失败：{str(e)}")
    
    # 总统计（可选）
    if 'c_seq_count' in locals() and 'f_seq_count' in locals():
        total = c_seq_count + f_seq_count
        print("-"*60)
        print(f"📈 总计 | 两个文件共 {total} 条蛋白质序列")
    print("="*60)