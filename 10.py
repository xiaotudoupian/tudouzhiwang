import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import random

# 设置随机种子确保可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 定义文件路径
pt_file = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/CC数据/CC.pt"
csv_file = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/CC数据/验证集_output/results_20260310-221532.csv"
gaf_file = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/CC数据/total_C_filtered_with_ancestors.gaf"
fasta_file = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/CC数据/total_C_sequences.fasta"

# 输出目录
output_dir = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/CC数据/split_data"
os.makedirs(output_dir, exist_ok=True)

def load_gaf_data():
    """加载GAF文件并构建GO术语与蛋白ID的映射关系"""
    # 自定义GAF列名（适配实际文件格式）
    gaf_columns = [
        "Protein_ID", "GO_Terms", "原始GO数量", "补全后GO数量", "新增GO数量"
    ]
    # 读取GAF文件（跳过警告行，设置正确分隔符）
    gaf_data = pd.read_csv(gaf_file, sep="\t", names=gaf_columns)
    
    # 构建GO术语到蛋白ID的映射
    go_to_proteins = defaultdict(list)
    protein_to_go = defaultdict(list)
    
    for _, row in gaf_data.iterrows():
        protein_id = row["Protein_ID"]
        go_terms = row["GO_Terms"].split(",") if pd.notna(row["GO_Terms"]) else []
        
        for go_id in go_terms:
            if go_id.startswith("GO:"):  # 过滤有效GO术语
                go_to_proteins[go_id].append(protein_id)
                protein_to_go[protein_id].append(go_id)
    
    # 去重
    for go_id in go_to_proteins:
        go_to_proteins[go_id] = list(set(go_to_proteins[go_id]))
    for protein_id in protein_to_go:
        protein_to_go[protein_id] = list(set(protein_to_go[protein_id]))
    
    print(f"GAF文件统计:")
    print(f"  - 总蛋白数: {len(gaf_data)}")
    print(f"  - 总GO术语数: {len(go_to_proteins)}")
    print(f"  - 前5个GO术语: {list(go_to_proteins.keys())[:5]}")
    
    return gaf_data, go_to_proteins, protein_to_go

def load_fasta_sequences():
    """加载FASTA文件，返回蛋白ID到序列的映射"""
    seq_dict = {}
    with open(fasta_file, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    seq_dict[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]  # 取第一个字段作为Protein_ID
                current_seq = []
            else:
                current_seq.append(line)
        # 处理最后一个序列
        if current_id is not None:
            seq_dict[current_id] = "".join(current_seq)
    print(f"FASTA文件统计: 总序列数 = {len(seq_dict)}")
    return seq_dict

def load_pt_data():
    """加载PT文件并返回蛋白ID和特征的映射"""
    pt_data = torch.load(pt_file, map_location='cpu')
    protein_ids = pt_data['protein_ids']
    cls_features = pt_data['cls_features']
    
    # 构建蛋白ID到特征的映射
    protein_to_feature = {}
    for idx, pid in enumerate(protein_ids):
        protein_to_feature[pid] = cls_features[idx]
    
    print(f"PT文件统计:")
    print(f"  - 蛋白ID数量: {len(protein_ids)}")
    print(f"  - 特征维度: {cls_features.shape}")
    print(f"  - 模型名称: {pt_data['model_name']}")
    
    return pt_data, protein_to_feature

def split_dataset():
    """划分数据集，确保所有GO术语都出现在训练集"""
    # 1. 加载所有数据
    print("="*60)
    print("开始加载数据...")
    gaf_data, go_to_proteins, protein_to_go = load_gaf_data()
    fasta_seqs = load_fasta_sequences()
    csv_data = pd.read_csv(csv_file)
    pt_data, protein_to_feature = load_pt_data()
    
    # 获取所有唯一的蛋白ID（以GAF文件为基准）
    all_proteins = list(gaf_data["Protein_ID"].unique())
    print(f"\n总唯一蛋白ID数量: {len(all_proteins)}")
    
    # 2. 核心步骤：确保所有GO术语都在训练集中
    print("\n" + "="*60)
    print("确保所有GO术语都包含在训练集中...")
    train_proteins = set()
    
    # 为每个GO术语至少选择一个蛋白放入训练集
    for go_id, proteins in go_to_proteins.items():
        if proteins:  # 确保有对应的蛋白
            selected_protein = random.choice(proteins)
            train_proteins.add(selected_protein)
    
    print(f"为覆盖所有GO术语选择的训练集蛋白数: {len(train_proteins)}")
    
    # 3. 划分剩余数据，保持8:1:1比例
    print("\n" + "="*60)
    print("划分剩余数据...")
    remaining_proteins = [p for p in all_proteins if p not in train_proteins]
    total_proteins = len(all_proteins)
    
    # 计算目标训练集大小 (80%)
    target_train_size = int(total_proteins * 0.8)
    additional_train_needed = target_train_size - len(train_proteins)
    
    # 补充训练集
    if additional_train_needed > 0 and remaining_proteins:
        # 从剩余蛋白中随机选择补充到训练集
        additional_train = random.sample(
            remaining_proteins, 
            min(additional_train_needed, len(remaining_proteins))
        )
        train_proteins.update(additional_train)
        remaining_proteins = [p for p in remaining_proteins if p not in additional_train]
    
    # 将剩余数据按1:1划分验证集和测试集
    val_size = int(len(remaining_proteins) * 0.5)
    val_proteins = set(random.sample(remaining_proteins, val_size))
    test_proteins = set([p for p in remaining_proteins if p not in val_proteins])
    
    # 转换为列表便于处理
    train_list = list(train_proteins)
    val_list = list(val_proteins)
    test_list = list(test_proteins)
    
    # 4. 验证GO术语覆盖
    print("\n" + "="*60)
    print("验证GO术语覆盖情况...")
    train_go_terms = set()
    for pid in train_list:
        if pid in protein_to_go:
            train_go_terms.update(protein_to_go[pid])
    
    all_go_terms = set(go_to_proteins.keys())
    missing_go = all_go_terms - train_go_terms
    
    if len(missing_go) == 0:
        print("✅ 验证通过：所有GO术语都在训练集中")
    else:
        print(f"❌ 验证失败：{len(missing_go)}个GO术语不在训练集中")
        print(f"缺失的GO术语: {list(missing_go)}")
        return
    
    # 5. 输出划分统计
    print("\n" + "="*60)
    print("数据集划分统计:")
    print(f"训练集: {len(train_list)} ({len(train_list)/total_proteins*100:.1f}%)")
    print(f"验证集: {len(val_list)} ({len(val_list)/total_proteins*100:.1f}%)")
    print(f"测试集: {len(test_list)} ({len(test_list)/total_proteins*100:.1f}%)")
    print(f"总计: {len(train_list)+len(val_list)+len(test_list)}")
    
    # 6. 保存划分结果
    print("\n" + "="*60)
    print("保存划分结果...")
    
    # 6.1 保存划分信息
    split_info = pd.DataFrame({
        "Protein_ID": train_list + val_list + test_list,
        "Dataset": ["train"]*len(train_list) + ["val"]*len(val_list) + ["test"]*len(test_list)
    })
    split_info.to_csv(os.path.join(output_dir, "dataset_split.csv"), index=False)
    
    # 6.2 拆分GAF文件
    train_gaf = gaf_data[gaf_data["Protein_ID"].isin(train_list)]
    val_gaf = gaf_data[gaf_data["Protein_ID"].isin(val_list)]
    test_gaf = gaf_data[gaf_data["Protein_ID"].isin(test_list)]
    
    train_gaf.to_csv(os.path.join(output_dir, "train.gaf"), sep="\t", index=False, header=False)
    val_gaf.to_csv(os.path.join(output_dir, "val.gaf"), sep="\t", index=False, header=False)
    test_gaf.to_csv(os.path.join(output_dir, "test.gaf"), sep="\t", index=False, header=False)
    
    # 6.3 拆分FASTA文件
    def write_fasta(protein_list, output_file):
        with open(output_file, 'w') as f:
            for pid in protein_list:
                if pid in fasta_seqs:
                    f.write(f">{pid}\n{fasta_seqs[pid]}\n")
    
    write_fasta(train_list, os.path.join(output_dir, "train.fasta"))
    write_fasta(val_list, os.path.join(output_dir, "val.fasta"))
    write_fasta(test_list, os.path.join(output_dir, "test.fasta"))
    
    # 6.4 拆分CSV文件
    train_csv = csv_data[csv_data["Protein_ID"].isin(train_list)]
    val_csv = csv_data[csv_data["Protein_ID"].isin(val_list)]
    test_csv = csv_data[csv_data["Protein_ID"].isin(test_list)]
    
    train_csv.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_csv.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_csv.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # 6.5 拆分PT文件
    def split_pt(protein_list, output_file):
        # 找到对应蛋白ID的索引
        indices = [i for i, pid in enumerate(pt_data['protein_ids']) if pid in protein_list]
        
        # 构建新的PT字典
        new_pt = {
            'protein_ids': [pt_data['protein_ids'][i] for i in indices],
            'cls_features': pt_data['cls_features'][indices],
            'model_name': pt_data['model_name'],
            'target_layer': pt_data['target_layer'],
            'max_sequence_length': pt_data['max_sequence_length']
        }
        
        torch.save(new_pt, output_file)
    
    split_pt(train_list, os.path.join(output_dir, "train.pt"))
    split_pt(val_list, os.path.join(output_dir, "val.pt"))
    split_pt(test_list, os.path.join(output_dir, "test.pt"))
    
    # 7. 最终验证
    print("\n" + "="*60)
    print("文件保存验证:")
    output_files = os.listdir(output_dir)
    for f in output_files:
        f_path = os.path.join(output_dir, f)
        size = os.path.getsize(f_path) / 1024 / 1024  # MB
        print(f"  - {f}: {size:.2f} MB")
    
    print("\n✅ 数据集划分完成！")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    split_dataset()