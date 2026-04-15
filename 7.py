import obonet
import pandas as pd
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# ===================== 核心配置 =====================
ALLOWED_RELATIONS = {"is_a", "part_of"}  # 只保留这两种关系的父术语
OBO_PATH = r"C:\Users\dell\Desktop\go.obo"  # 请替换为你的go.obo实际路径

# ===================== 你的GAF文件配置 =====================
FILES_CONFIG = {
    "total_C": {
        "input": r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\total_C_filtered.gaf",
        "output": r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\total_C_filtered_with_ancestors.gaf"
    },
    "total_F": {
        "input": r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\total_F_filtered.gaf",
        "output": r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\total_F_filtered_with_ancestors.gaf"
    }
}

# ===================== 加载 GO 本体 =====================
def load_go(obo_path):
    print("[1/4] 加载 go.obo 文件...")
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"GO本体文件不存在：{obo_path}，请检查路径！")
    
    G = obonet.read_obo(obo_path)
    go2ns = {}
    for node, attr in G.nodes(data=True):
        if "namespace" in attr:
            go2ns[node] = attr["namespace"]
    
    print(f"✅ GO本体加载完成：")
    print(f"  - 总GO术语数：{len(G.nodes())}")
    print(f"  - 有命名空间的GO术语数：{len(go2ns)}")
    return G, go2ns

# ===================== 核心：获取单个GO的所有祖先（仅is_a/part_of，同命名空间） =====================
def get_all_ancestors(go_id, G, go2ns):
    """
    获取单个GO术语的所有祖先（父术语）
    :param go_id: 目标GO术语（如GO:0015297）
    :param G: GO本体有向图
    :param go2ns: GO→命名空间映射
    :return: 祖先GO集合（不含自身）
    """
    # 过滤无效GO
    if go_id not in G or go_id not in go2ns:
        return set()
    
    target_ns = go2ns[go_id]  # 目标GO的命名空间
    visited = set()  # 存储所有祖先
    stack = [go_id]   # 深度优先遍历栈
    
    while stack:
        current_go = stack.pop()
        # 遍历当前GO的出边（指向父节点）
        for _, parent_go, rel_type in G.out_edges(current_go, keys=True):
            # 只保留is_a/part_of关系
            if rel_type not in ALLOWED_RELATIONS:
                continue
            # 只保留同命名空间的父节点
            if parent_go not in go2ns or go2ns[parent_go] != target_ns:
                continue
            # 去重并继续遍历父节点的父节点
            if parent_go not in visited:
                visited.add(parent_go)
                stack.append(parent_go)
    return visited

# ===================== 辅助函数：解析/统计GO字符串 =====================
def parse_go_terms(go_str):
    """解析GO字符串为列表（处理空值、多余引号/空格）"""
    if pd.isna(go_str) or go_str.strip() == "":
        return []
    # 清理格式：去引号、去空格、按逗号分割
    clean_str = go_str.strip().replace('"', '').replace("'", "")
    go_list = [g.strip() for g in clean_str.split(",") if g.strip()]
    return go_list

def count_go_terms(go_str):
    """统计GO字符串中的术语数量"""
    return len(parse_go_terms(go_str))

# ===================== 核心：处理单行GO字符串（补全父术语） =====================
def process_go_string(go_str, G, go2ns):
    """
    给单行GO字符串补全所有父术语
    :param go_str: 原始GO字符串（如"GO:0004364,GO:0016740"）
    :return: 补全后的GO字符串（排序后）
    """
    # 解析原始GO列表
    raw_go_list = parse_go_terms(go_str)
    if not raw_go_list:
        return ""
    
    # 收集原始GO + 所有祖先GO
    all_go_set = set(raw_go_list)
    for go_id in raw_go_list:
        ancestors = get_all_ancestors(go_id, G, go2ns)
        all_go_set.update(ancestors)  # 合并祖先
    
    # 排序并格式化输出（保持一致性）
    sorted_go_list = sorted(all_go_set)
    return ",".join(sorted_go_list)

# ===================== 处理单个GAF文件 =====================
def process_single_gaf(file_type, input_path, output_path, G, go2ns):
    print(f"\n[处理 {file_type} 文件]")
    # 检查输入文件
    if not os.path.exists(input_path):
        print(f"❌ 错误：文件 {input_path} 不存在，跳过！")
        return None
    
    # 读取GAF文件（制表符分隔，表头为Protein_ID\tGO_Terms）
    df = pd.read_csv(input_path, sep="\t", header=0, names=["Protein_ID", "GO_Terms"])
    print(f"✅ 读取成功：{len(df)} 行数据")
    
    # 1. 统计原始GO数量
    df["原始GO数量"] = df["GO_Terms"].apply(count_go_terms)
    
    # 2. 批量补全父术语（带进度条）
    print(f"📌 批量补全GO父术语...")
    tqdm.pandas(desc=f"Processing {file_type}")
    df["GO_Terms"] = df["GO_Terms"].progress_apply(lambda x: process_go_string(x, G, go2ns))
    
    # 3. 统计补全后数量
    df["补全后GO数量"] = df["GO_Terms"].apply(count_go_terms)
    df["新增GO数量"] = df["补全后GO数量"] - df["原始GO数量"]
    
    # 4. 保存结果（制表符分隔，保留统计列）
    output_cols = ["Protein_ID", "GO_Terms", "原始GO数量", "补全后GO数量", "新增GO数量"]
    df[output_cols].to_csv(output_path, sep="\t", index=False)
    print(f"✅ 保存完成：{output_path}")
    
    # 5. 输出该文件的统计摘要
    print(f"\n[{file_type} 统计摘要]")
    print(f"  - 总蛋白数：{len(df)}")
    print(f"  - 平均原始GO数量：{df['原始GO数量'].mean():.2f}")
    print(f"  - 平均补全后GO数量：{df['补全后GO数量'].mean():.2f}")
    print(f"  - 平均新增GO数量：{df['新增GO数量'].mean():.2f}")
    print(f"  - 最大新增数量：{df['新增GO数量'].max()}")
    print(f"  - 新增数量为0的样本数：{len(df[df['新增GO数量'] == 0])}")
    
    return df

# ===================== 主函数 =====================
def main():
    try:
        # 1. 加载GO本体
        G, go2ns = load_go(OBO_PATH)
        
        # 2. 批量处理所有GAF文件
        all_stats = {}
        print("\n[2/4] 开始处理GAF文件...")
        for file_type, config in FILES_CONFIG.items():
            df = process_single_gaf(
                file_type=file_type,
                input_path=config["input"],
                output_path=config["output"],
                G=G,
                go2ns=go2ns
            )
            if df is not None:
                all_stats[file_type] = {
                    "总蛋白数": len(df),
                    "平均原始GO数": round(df["原始GO数量"].mean(), 2),
                    "平均补全后GO数": round(df["补全后GO数量"].mean(), 2),
                    "平均新增GO数": round(df["新增GO数量"].mean(), 2)
                }
        
        # 3. 输出全局汇总
        print("\n" + "="*60)
        print("[全局统计汇总（GO父术语补全）]")
        print("="*60)
        for file_type, stats in all_stats.items():
            print(f"\n{file_type.upper()}：")
            for k, v in stats.items():
                print(f"  - {k}：{v}")
        
        print("\n🎉 所有文件处理完成！")
        print("\n📁 输出文件列表：")
        for file_type, config in FILES_CONFIG.items():
            print(f"  - {file_type}：{config['output']}")
    
    except Exception as e:
        print(f"\n❌ 程序执行出错：{str(e)}")

if __name__ == "__main__":
    main()