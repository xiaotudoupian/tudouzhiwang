import csv
import pickle
import re
from collections import defaultdict

# ===================== 配置参数（仅保留验证集）=====================
# 输入文件路径：只保留val.csv（原TEST_CSV，已删除TRAIN_CSV）
VAL_CSV = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/数据/7_go_terms_matched_val.csv"
FASTA_FILE = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/数据/fasta_final_filtered.fasta"

# 输出路径（TALE的data/me目录）
OUTPUT_DIR = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/TALE/data/me/"
ONTOLOGY = "mf"  # 按你的需求改：mf/bp/cc

# ===================== 第一步：解析FASTA文件，建立ID→序列的映射 =====================
def parse_fasta(fasta_path):
    """解析FASTA文件，返回{蛋白ID: 氨基酸序列}的字典"""
    fasta_dict = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 以>开头的是ID行
            if line.startswith(">"):
                # 保存上一个ID的序列
                if current_id:
                    fasta_dict[current_id] = "".join(current_seq)
                # 提取ID（取>后的第一个字段）
                current_id = line.lstrip(">").split()[0]
                current_seq = []
            else:
                # 拼接序列
                current_seq.append(line)
        # 保存最后一个ID的序列
        if current_id:
            fasta_dict[current_id] = "".join(current_seq)
    
    print(f"FASTA文件解析完成，共{len(fasta_dict)}条序列")
    return fasta_dict

# ===================== 第二步：解析CSV文件，生成TALE格式数据 =====================
def process_csv(csv_path, fasta_dict):
    """
    解析CSV文件（仅处理val.csv），生成TALE要求的seq_list和label_list
    :param csv_path: val.csv路径
    :param fasta_dict: 蛋白ID→序列的字典
    :return: seq_list, label_list (TALE格式), 所有GO术语集合
    """
    seq_list = []   # TALE的seq文件（列表+字典）
    label_list = [] # TALE的label文件（列表+GO索引，先存GO术语）
    go_terms_all = set()  # 收集所有GO术语，用于后续索引映射
    
    # 读取CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            protein_id = row["Source_ID"].strip()
            go_terms_str = row["GO_Terms"].strip().strip('"')  # 去掉GO术语的双引号
            go_terms = [go.strip() for go in go_terms_str.split(",") if go.strip()]
            
            # 1. 匹配FASTA序列（跳过无序列的ID）
            if protein_id not in fasta_dict:
                print(f"警告：{protein_id} 在FASTA中无序列，跳过")
                continue
            
            # 2. 构建TALE的seq字典
            seq_dict = {
                "ID": protein_id,
                "ac": "0",          # 无登录号填0
                "date": "0",        # 无日期填0
                "seq": fasta_dict[protein_id],
                "GO": go_terms
            }
            seq_list.append(seq_dict)
            
            # 3. 保存GO术语（后续转索引）
            label_list.append(go_terms)
            go_terms_all.update(go_terms)
            
            # 进度提示
            if (idx + 1) % 100 == 0:
                print(f"已处理{idx+1}条数据")
    
    print(f"CSV解析完成：{csv_path} → 有效数据{len(seq_list)}条")
    return seq_list, label_list, go_terms_all

# ===================== 第三步：生成GO术语→索引的映射 =====================
def build_go_index(all_go_terms):
    """为所有GO术语生成唯一索引（从0开始）"""
    go_list = sorted(list(all_go_terms))
    go2ind = {go: idx for idx, go in enumerate(go_list)}
    # 保存GO索引映射（方便后续查看）
    with open(f"{OUTPUT_DIR}/go2ind_{ONTOLOGY}.pickle", "wb") as f:
        pickle.dump(go2ind, f)
    print(f"GO索引映射完成：共{len(go2ind)}个GO术语")
    return go2ind

# ===================== 第四步：将GO术语转为索引，生成label文件 =====================
def convert_go_to_ind(label_list, go2ind):
    """将label_list中的GO术语转为索引"""
    label_ind_list = []
    for go_terms in label_list:
        inds = [go2ind[go] for go in go_terms if go in go2ind]
        label_ind_list.append(inds)
    return label_ind_list

# ===================== 主函数（仅处理val.csv）=====================
if __name__ == "__main__":
    # 1. 解析FASTA文件
    fasta_dict = parse_fasta(FASTA_FILE)
    
    # 2. 处理val.csv（核心：仅保留这一步，删除训练集处理）
    val_seq_list, val_label_go, val_go_all = process_csv(VAL_CSV, fasta_dict)
    
    # 3. 生成GO索引映射（仅基于val.csv的GO术语）
    go2ind = build_go_index(val_go_all)
    
    # 4. 将GO术语转为索引
    val_label_ind = convert_go_to_ind(val_label_go, go2ind)
    
    # 5. 保存TALE格式的验证集文件（命名为test_seq_xxx/test_label_xxx，适配TALE原代码）
    with open(f"{OUTPUT_DIR}/test_seq_{ONTOLOGY}", "wb") as f:
        pickle.dump(val_seq_list, f)
    with open(f"{OUTPUT_DIR}/test_label_{ONTOLOGY}", "wb") as f:
        pickle.dump(val_label_ind, f)
    
    # 6. 输出统计信息（仅展示val.csv相关）
    print("\n=== 最终生成文件统计 ===")
    print(f"验证集seq条数：{len(val_seq_list)}")
    print(f"验证集label条数：{len(val_label_ind)}")
    print(f"GO术语总数：{len(go2ind)}")
    print(f"输出路径：{OUTPUT_DIR}")
    print(f"生成文件：test_seq_{ONTOLOGY}、test_label_{ONTOLOGY}、go2ind_{ONTOLOGY}.pickle")
    print("\n文件生成完成！可直接作为TALE的验证集使用。")
