import os
from collections import defaultdict

# ===================== 配置文件路径（已替换为指定的新路径） =====================
# GO OBO文件路径
OBO_FILE = "/misc/hard_disk/others_res/zhangdy/go.obo"

# 定义需要处理的3个文件：输入路径 → 输出路径（在原目录下生成带父术语的新文件）
INPUT_OUTPUT_MAP = {
    # 训练集（指定路径）
    "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits/BP/merged_species_dataset/merged_train_annotations.tsv":
    "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits/BP/merged_species_dataset/merged_train_annotations_with_parents.tsv",
    
    # 验证集（指定路径）
    "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits/BP/merged_species_dataset/merged_val_annotations.tsv":
    "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits/BP/merged_species_dataset/merged_val_annotations_with_parents.tsv",
    
    # 测试集（指定路径）
    "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits/BP/merged_species_dataset/merged_test_annotations.tsv":
    "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits/BP/merged_species_dataset/merged_test_annotations_with_parents.tsv"
}

# ===================== 步骤1：解析go.obo，构建GO祖先字典 =====================
def parse_go_obo(obo_path):
    """
    解析go.obo，返回：
    1. go_ancestors: key=GO ID, value=该GO的所有祖先GO集合（is_a + part_of，仅限BP）
    2. go_namespace: key=GO ID, value=命名空间（BP/CC/MF）
    """
    # 先存储直接父节点（is_a + part_of 关系）
    go_direct_parents = defaultdict(set)
    go_namespace = dict()
    current_go = None
    current_ns = None

    print(f"正在解析OBO文件：{obo_path}")
    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 开始新的Term块
            if line == "[Term]":
                current_go = None
                current_ns = None
            
            # 提取GO ID（核心标识）
            elif line.startswith("id: "):
                current_go = line.split("id: ")[1].split("!")[0].strip()
            
            # 提取命名空间（仅处理biological_process）
            elif line.startswith("namespace: "):
                current_ns = line.split("namespace: ")[1].strip()
                if current_go:
                    go_namespace[current_go] = current_ns
            
            # 提取is_a关系（直接父节点，仅限BP）
            elif line.startswith("is_a: ") and current_ns == "biological_process":
                parent_go = line.split("is_a: ")[1].split("!")[0].strip()
                go_direct_parents[current_go].add(parent_go)
            
            # 提取part_of关系（直接父节点，仅限BP）
            elif line.startswith("relationship: part_of ") and current_ns == "biological_process":
                parent_go = line.split("relationship: part_of ")[1].split("!")[0].strip()
                go_direct_parents[current_go].add(parent_go)

    # 递归获取所有祖先GO（多层级，直到根节点）
    go_ancestors = defaultdict(set)
    def get_all_ancestors(go_id):
        """递归获取一个GO术语的所有祖先（含多层）"""
        if go_id in go_ancestors:
            return go_ancestors[go_id]
        
        # 先获取直接父节点
        direct_parents = go_direct_parents.get(go_id, set())
        all_ancestors = direct_parents.copy()
        
        # 递归获取父节点的祖先
        for parent in direct_parents:
            all_ancestors.update(get_all_ancestors(parent))
        
        go_ancestors[go_id] = all_ancestors
        return all_ancestors

    # 为所有BP类型的GO计算完整祖先链
    bp_go_count = 0
    for go_id in go_namespace:
        if go_namespace[go_id] == "biological_process":
            get_all_ancestors(go_id)
            bp_go_count += 1

    print(f"OBO解析完成：")
    print(f"  - 总BP类GO术语数：{bp_go_count}")
    print(f"  - 含祖先关系的GO术语数：{len(go_ancestors)}")
    return go_ancestors, go_namespace

# ===================== 步骤2：处理单个TSV文件，补全GO父术语 =====================
def process_tsv(input_tsv, output_tsv, go_ancestors, go_namespace):
    """
    处理单个TSV注释文件：
    1. 读取每行蛋白的GO术语
    2. 补全每个GO的所有BP类祖先术语
    3. 去重并生成新的注释行
    4. 输出到指定路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_tsv):
        print(f"❌ 错误：输入文件不存在 → {input_tsv}")
        return
    
    print(f"\n开始处理文件：{input_tsv}")
    processed_count = 0  # 成功处理的行数
    skipped_count = 0    # 跳过的异常行数

    with open(input_tsv, "r", encoding="utf-8") as in_f, open(output_tsv, "w", encoding="utf-8") as out_f:
        # 读取并写入表头（保持原表头不变）
        header = in_f.readline().strip()
        out_f.write(f"{header}\n")

        # 逐行处理蛋白注释
        for line_num, line in enumerate(in_f, start=2):  # 行号从2开始（表头为1）
            line = line.strip()
            if not line:
                skipped_count += 1
                continue

            # 拆分列（适配 "protein_id\tgo_terms\tcount" 格式）
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"⚠️  警告：第{line_num}行格式异常（列数≠3），跳过 → {line}")
                skipped_count += 1
                continue

            # 提取列数据
            protein_id = parts[0].strip()
            go_terms_str = parts[1].strip()
            original_count = parts[2].strip()

            # 拆分原始GO列表（过滤空值）
            original_go_list = [go.strip() for go in go_terms_str.split(",") if go.strip()]
            # 存储补全后的所有GO（去重集合）
            complete_go_set = set(original_go_list)

            # 为每个GO补全BP类祖先
            for go_id in original_go_list:
                # 仅处理BP命名空间的GO（避免混入CC/MF）
                if go_namespace.get(go_id) != "biological_process":
                    continue
                # 添加该GO的所有祖先（多层级）
                complete_go_set.update(go_ancestors.get(go_id, set()))

            # 转换为有序列表（排序后更易读），重新生成注释字符串和计数
            complete_go_list = sorted(list(complete_go_set))
            new_go_terms = ",".join(complete_go_list)
            new_go_count = str(len(complete_go_list))

            # 写入输出文件
            out_f.write(f"{protein_id}\t{new_go_terms}\t{new_go_count}\n")
            processed_count += 1

    # 输出处理统计
    print(f"✅ 文件处理完成：")
    print(f"  - 成功处理行数：{processed_count}")
    print(f"  - 跳过异常行数：{skipped_count}")
    print(f"  - 输出文件路径：{output_tsv}")

# ===================== 主流程：解析OBO → 批量处理3个文件 =====================
if __name__ == "__main__":
    # 第一步：解析GO OBO文件（只需解析一次，复用给所有TSV）
    try:
        go_ancestors, go_namespace = parse_go_obo(OBO_FILE)
    except Exception as e:
        print(f"❌ OBO文件解析失败：{str(e)}")
        exit(1)

    # 第二步：批量处理每个TSV文件
    print("\n" + "="*80)
    print("开始批量处理注释文件（补全GO父术语）")
    print("="*80)
    
    for input_tsv, output_tsv in INPUT_OUTPUT_MAP.items():
        process_tsv(input_tsv, output_tsv, go_ancestors, go_namespace)

    # 第三步：输出最终结果汇总
    print("\n" + "="*80)
    print("✅ 所有文件处理完成！")
    print("📁 输出文件列表：")
    for output_tsv in INPUT_OUTPUT_MAP.values():
        if os.path.exists(output_tsv):
            file_size = os.path.getsize(output_tsv) / 1024  # 转换为KB
            print(f"  - {output_tsv} (大小：{file_size:.2f} KB)")
        else:
            print(f"  - ❌ {output_tsv}（生成失败）")
    print("="*80)