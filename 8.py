import os
import obonet
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

# ===================== 配置参数（修改为你的表格文件路径） =====================
# GO OBO文件路径
OBO_FILE = r"C:\Users\dell\Desktop\go.obo"
# CC/MF子本体的自定义表格文件（替换成你实际的文件路径！）
CC_TABLE_FILE = r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\total_C_filtered_with_ancestors.gaf"  # 你的CC表格
MF_TABLE_FILE = r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\total_F_filtered_with_ancestors.gaf"  # 你的MF表格

# IC值结果输出路径
OUTPUT_CC_IC = r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\CC_GO_IC_values_actual.csv"
OUTPUT_MF_IC = r"C:\Users\dell\Desktop\GAF_filtered_by_fasta\MF_GO_IC_values_actual.csv"

# ===================== 核心函数 =====================
def parse_go_obo(obo_file):
    """解析GO OBO文件，构建DAG图（和之前一致，无需修改）"""
    print("📌 开始解析GO OBO文件...")
    go_dag = obonet.read_obo(obo_file)
    go_dag = go_dag.to_directed()
    
    go_namespace = {}
    go_parents = defaultdict(list)
    for node_id, node_attr in go_dag.nodes(data=True):
        if not node_id.startswith("GO:"):
            continue
        go_namespace[node_id] = node_attr.get('namespace', '')
        for pred, _, edge_attr in go_dag.in_edges(node_id, data=True):
            if edge_attr.get('type') == 'is_a' or edge_attr.get('relationship') == 'is_a':
                go_parents[node_id].append(pred)
    
    print(f"✅ OBO解析完成 | 总GO数: {len(go_namespace)} | 子本体分布: {pd.Series(go_namespace.values()).value_counts().to_dict()}")
    return go_dag, go_namespace, go_parents

def parse_custom_table(table_file, target_namespace, go_namespace):
    """
    解析你的自定义表格（Protein_ID+GO_Terms），返回：
    - go_annotations: 字典，{GO_ID: 注释的蛋白ID集合}（仅目标子本体）
    - total_proteins: 总注释蛋白数（去重）
    """
    print(f"\n📌 开始解析自定义表格: {os.path.basename(table_file)}")
    # 读取表格（兼容CSV/TSV，自动识别分隔符）
    df = pd.read_csv(
        table_file, 
        sep=None,  # 自动识别分隔符（逗号/制表符）
        engine='python',
        dtype=str, 
        na_filter=False,
        encoding='utf-8'
    )
    
    # 检查必要列是否存在
    required_cols = ['Protein_ID', 'GO_Terms']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"表格缺少必要列！需要包含：{required_cols}，当前列：{df.columns.tolist()}")
    
    # 初始化GO注释统计
    go_annotations = defaultdict(set)
    all_proteins = set()
    
    # 遍历每一行，拆分GO_Terms并统计
    for _, row in df.iterrows():
        protein_id = row['Protein_ID'].strip()
        go_terms_str = row['GO_Terms'].strip()
        
        # 跳过空蛋白ID或空GO_Terms
        if not protein_id or not go_terms_str:
            continue
        
        # 拆分GO术语（按逗号分隔）
        go_terms = [go.strip() for go in go_terms_str.split(',') if go.strip().startswith('GO:')]
        
        # 筛选目标子本体的GO
        for go_id in go_terms:
            if go_namespace.get(go_id, '') == target_namespace:
                go_annotations[go_id].add(protein_id)
                all_proteins.add(protein_id)
    
    # 总注释蛋白数
    total_proteins = len(all_proteins)
    
    print(f"✅ 表格解析完成 | 目标子本体: {target_namespace}")
    print(f"   - 有效GO注释数: {len(go_annotations)} | 总注释蛋白数: {total_proteins}")
    
    # 边界提示：如果无有效注释
    if len(go_annotations) == 0:
        print("⚠️ 警告：未解析到该子本体的有效GO注释！请检查表格中的GO_Terms是否正确。")
    
    return go_annotations, total_proteins

def propagate_annotations(go_annotations, go_parents):
    """注释向上传播（和之前一致）"""
    print("\n📌 开始向上传播GO注释（符合GO DAG继承规则）...")
    go_total_annotations = defaultdict(set)
    
    # 初始化原始注释
    for go_id, proteins in go_annotations.items():
        go_total_annotations[go_id].update(proteins)
    
    # 循环传播注释
    changed = True
    iteration = 0
    while changed and iteration < 100:
        changed = False
        iteration += 1
        current_annot = dict(go_total_annotations)
        
        for go_id in current_annot:
            for parent_go in go_parents.get(go_id, []):
                original_count = len(go_total_annotations[parent_go])
                go_total_annotations[parent_go].update(current_annot[go_id])
                new_count = len(go_total_annotations[parent_go])
                if new_count > original_count:
                    changed = True
    
    # 仅保留有注释的GO
    go_total_annotations = {
        go_id: len(proteins) 
        for go_id, proteins in go_total_annotations.items() 
        if len(proteins) > 0
    }
    
    print(f"✅ 注释传播完成 | 传播后有注释的GO总数: {len(go_total_annotations)}")
    return go_total_annotations

def calculate_go_ic(go_total_annotations, total_proteins, go_namespace, target_namespace):
    """
    计算IC值（增加空数据防护）
    """
    print("\n📌 开始计算GO IC值（仅保留你数据中实际出现的GO）...")
    ic_results = []
    
    # 边界处理：无注释时直接返回空DataFrame
    if total_proteins == 0 or len(go_total_annotations) == 0:
        print("⚠️ 警告：无有效注释数据，无法计算IC值！")
        return pd.DataFrame(columns=['GO_ID', 'namespace', 'annotation_count', 'frequency', 'ic_value'])
    
    # 遍历有注释的GO，计算IC
    for go_id, annot_count in go_total_annotations.items():
        if go_namespace.get(go_id, "") != target_namespace:
            continue
        
        # 计算频率（避免除以0）
        frequency = annot_count / total_proteins
        frequency = max(frequency, 1e-10)
        
        # 计算IC值
        ic_value = -np.log2(frequency)
        
        ic_results.append({
            'GO_ID': go_id,
            'namespace': go_namespace[go_id],
            'annotation_count': annot_count,
            'frequency': round(frequency, 6),
            'ic_value': round(ic_value, 6)
        })
    
    # 转换为DataFrame并排序
    ic_df = pd.DataFrame(ic_results)
    if not ic_df.empty:
        ic_df = ic_df.sort_values('ic_value', ascending=False).reset_index(drop=True)
        print(f"✅ IC值计算完成 | 目标子本体有效GO数: {len(ic_df)}")
        print(f"   - IC值范围: {ic_df['ic_value'].min():.4f} ~ {ic_df['ic_value'].max():.4f}")
    else:
        print("⚠️ 警告：未计算出任何IC值！")
    
    return ic_df

# ===================== 主流程 =====================
def main():
    # 1. 解析OBO文件
    go_dag, go_namespace, go_parents = parse_go_obo(OBO_FILE)
    
    # 2. 计算CC子本体IC值
    print("\n" + "="*60 + " 计算CC子本体IC值 " + "="*60)
    cc_annotations, cc_total_proteins = parse_custom_table(
        CC_TABLE_FILE, 
        target_namespace='cellular_component',
        go_namespace=go_namespace
    )
    cc_total_annot = propagate_annotations(cc_annotations, go_parents)
    cc_ic_df = calculate_go_ic(cc_total_annot, cc_total_proteins, go_namespace, 'cellular_component')
    # 保存CC结果
    cc_ic_df.to_csv(OUTPUT_CC_IC, index=False, encoding='utf-8-sig')
    print(f"\n📤 CC IC结果已保存至: {OUTPUT_CC_IC}")
    
    # 3. 计算MF子本体IC值
    print("\n" + "="*60 + " 计算MF子本体IC值 " + "="*60)
    mf_annotations, mf_total_proteins = parse_custom_table(
        MF_TABLE_FILE, 
        target_namespace='molecular_function',
        go_namespace=go_namespace
    )
    mf_total_annot = propagate_annotations(mf_annotations, go_parents)
    mf_ic_df = calculate_go_ic(mf_total_annot, mf_total_proteins, go_namespace, 'molecular_function')
    # 保存MF结果
    mf_ic_df.to_csv(OUTPUT_MF_IC, index=False, encoding='utf-8-sig')
    print(f"\n📤 MF IC结果已保存至: {OUTPUT_MF_IC}")
    
    # 4. 最终统计
    print("\n" + "="*60 + " 最终统计结果 " + "="*60)
    # CC统计
    if not cc_ic_df.empty:
        print(f"CC子本体 | 实际GO数: {len(cc_ic_df)} | 平均IC值: {cc_ic_df['ic_value'].mean():.4f} | 中位数IC: {cc_ic_df['ic_value'].median():.4f}")
    else:
        print("CC子本体 | 无有效IC数据")
    # MF统计
    if not mf_ic_df.empty:
        print(f"MF子本体 | 实际GO数: {len(mf_ic_df)} | 平均IC值: {mf_ic_df['ic_value'].mean():.4f} | 中位数IC: {mf_ic_df['ic_value'].median():.4f}")
    else:
        print("MF子本体 | 无有效IC数据")
    
    print("\n🎉 程序运行完成！结果文件已保存到桌面。")

if __name__ == "__main__":
    main()