import os
import obonet
from collections import defaultdict, Counter

# ===================== 全局配置（无需修改） =====================
# 文件路径
OBO_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/数据/go.obo"
FILE_8_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/数据/8_go_terms_matched_train.csv"
FILE_7_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/数据/9_go_terms_matched_train.csv"

# 允许的层级关系（True Path Rule）
ALLOWED_RELATIONS = {"is_a", "part_of"}

# ===================== 第一步：解析GO本体，构建递归父术语映射 =====================
def build_go_ancestor_mapping(obo_path):
    """
    解析GO本体，构建：
    1. go2ancestors：{GO术语: 所有递归父术语集合（含自身）}
    2. go2direct_parents：{GO术语: 直接父术语集合}
    """
    print(f"📌 开始解析GO本体：{obo_path}")
    # 解析OBO（保留所有术语，包括过时的）
    go_graph = obonet.read_obo(obo_path, ignore_obsolete=False)
    print(f"✅ OBO解析完成 | 总节点数：{len(go_graph.nodes())} | 总边数：{len(go_graph.edges())}")
    
    # 1. 构建直接父术语映射
    go2direct_parents = defaultdict(set)
    for child, parent, rel_type in go_graph.edges(keys=True):
        if rel_type in ALLOWED_RELATIONS:
            go2direct_parents[child].add(parent)
    
    # 2. 递归获取所有祖先（含自身）
    go2ancestors = {}
    def get_all_ancestors(go_term):
        """递归获取GO术语的所有祖先（含自身）"""
        if go_term in go2ancestors:
            return go2ancestors[go_term]
        # 初始化为自身
        ancestors = {go_term}
        # 遍历直接父节点，递归获取祖先
        for parent in go2direct_parents.get(go_term, set()):
            ancestors.update(get_all_ancestors(parent))
        go2ancestors[go_term] = ancestors
        return ancestors
    
    # 为所有GO术语预计算祖先
    for go_term in go_graph.nodes():
        get_all_ancestors(go_term)
    
    print(f"✅ 祖先映射构建完成 | 覆盖GO术语数：{len(go2ancestors)}")
    return go2ancestors, go2direct_parents, go_graph.nodes()

# ===================== 第二步：读取CSV文件，解析GO列 =====================
def read_go_csv(file_path):
    """
    读取CSV文件，返回：
    1. sample_go_dict：{Source_ID: GO术语集合}
    2. all_go_set：文件中所有唯一GO术语集合
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 文件不存在：{file_path}")
    
    sample_go_dict = {}
    all_go_set = set()
    line_num = 0
    
    print(f"\n📌 开始读取文件：{os.path.basename(file_path)}")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        # 跳过表头
        header = f.readline().strip()
        assert "Source_ID" in header and "GO_Terms" in header, "文件格式错误"
        
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            
            # 拆分Source_ID和GO_Terms（处理GO中含逗号的情况）
            parts = line.split(",", 1)  # 只拆分第一个逗号
            if len(parts) < 2:
                source_id = parts[0].strip() if parts else f"未知行_{line_num}"
                go_terms = []
            else:
                source_id, go_str = parts
                source_id = source_id.strip()
                # 拆分GO术语（逗号分隔）
                go_terms = [go.strip() for go in go_str.split(",") if go.strip().startswith("GO:")]
            
            # 去重并转为集合
            go_set = set(go_terms)
            sample_go_dict[source_id] = go_set
            # 加入全局GO集合
            all_go_set.update(go_set)
    
    print(f"✅ 文件读取完成 | 样本数：{len(sample_go_dict)} | 唯一GO数：{len(all_go_set)}")
    return sample_go_dict, all_go_set

# ===================== 第三步：验证补全结果 =====================
def validate_go_completion(sample_8_dict, sample_7_dict, go2ancestors, go_nodes):
    """
    验证7文件的GO补全是否正确：
    对每个样本，7的GO集合应等于8的GO集合的所有祖先的并集
    """
    print("\n" + "="*100)
    print("📊 开始验证GO补全结果（True Path Rule）")
    print("="*100)
    
    # 统计结果
    stats = {
        "total_samples": 0,
        "valid_samples": 0,  # 补全正确的样本
        "invalid_samples": 0,  # 补全错误的样本
        "missing_go_samples": 0,  # 7文件缺少父术语的样本
        "extra_go_samples": 0,  # 7文件多了无关GO的样本
        "unknown_go_samples": 0,  # 含不在本体中的GO的样本
        "empty_8_samples": 0,  # 8文件GO为空的样本
    }
    
    # 异常样本记录
    invalid_samples_detail = defaultdict(list)
    unknown_go_list = []  # 不在本体中的GO术语
    
    # 遍历所有样本（以8文件为基准）
    for source_id, go8_set in sample_8_dict.items():
        stats["total_samples"] += 1
        
        # 1. 获取7文件的GO集合
        go7_set = sample_7_dict.get(source_id, set())
        
        # 2. 检查8文件是否为空
        if not go8_set:
            stats["empty_8_samples"] += 1
            # 8文件为空时，7文件也应为空
            if go7_set:
                stats["invalid_samples"] += 1
                invalid_samples_detail["8文件为空但7文件有GO"].append({
                    "source_id": source_id,
                    "7文件GO": go7_set
                })
            else:
                stats["valid_samples"] += 1
            continue
        
        # 3. 检查是否有未知GO（不在本体中）
        unknown_8 = [go for go in go8_set if go not in go_nodes]
        unknown_7 = [go for go in go7_set if go not in go_nodes]
        all_unknown = unknown_8 + unknown_7
        if all_unknown:
            stats["unknown_go_samples"] += 1
            unknown_go_list.extend(all_unknown)
            invalid_samples_detail["含未知GO"].append({
                "source_id": source_id,
                "8文件未知GO": unknown_8,
                "7文件未知GO": unknown_7
            })
        
        # 4. 计算8文件GO应补全的所有祖先
        expected_7_set = set()
        for go in go8_set:
            if go in go2ancestors:
                expected_7_set.update(go2ancestors[go])
        
        # 5. 验证补全结果
        # 5.1 7文件缺少的GO（应补全但未补全）
        missing_go = expected_7_set - go7_set
        # 5.2 7文件多的GO（不该有但有）
        extra_go = go7_set - expected_7_set
        
        if not missing_go and not extra_go and not all_unknown:
            # 补全正确
            stats["valid_samples"] += 1
        else:
            # 补全错误
            stats["invalid_samples"] += 1
            if missing_go:
                stats["missing_go_samples"] += 1
                invalid_samples_detail["缺少父术语"].append({
                    "source_id": source_id,
                    "8文件GO": go8_set,
                    "缺失的父术语": missing_go,
                    "7文件实际GO": go7_set
                })
            if extra_go:
                stats["extra_go_samples"] += 1
                invalid_samples_detail["多了无关GO"].append({
                    "source_id": source_id,
                    "8文件GO": go8_set,
                    "多余的GO": extra_go,
                    "7文件实际GO": go7_set
                })
    
    # 去重未知GO
    unknown_go_set = set(unknown_go_list)
    
    # 输出统计结果
    print("\n" + "="*100)
    print("📋 补全验证统计结果")
    print("="*100)
    for k, v in stats.items():
        print(f"{k.replace('_', ' ')}：{v}")
    print(f"未知GO术语数：{len(unknown_go_set)}")
    print(f"补全正确率：{stats['valid_samples']/stats['total_samples']*100:.2f}%")
    
    # 输出异常样本详情（最多显示前5个）
    print("\n" + "="*100)
    print("🔍 异常样本详情（最多显示前5个）")
    print("="*100)
    for error_type, samples in invalid_samples_detail.items():
        print(f"\n【{error_type}】（共{len(samples)}个样本）：")
        show_samples = samples[:5]
        for idx, sample in enumerate(show_samples):
            print(f"  样本{idx+1} - Source_ID：{sample['source_id']}")
            for k, v in sample.items():
                if k != "source_id":
                    print(f"    {k}：{v}")
        if len(samples) > 5:
            print(f"    ... 还有{len(samples)-5}个样本未显示")
    
    # 输出未知GO术语（最多显示前20个）
    if unknown_go_set:
        print("\n" + "="*100)
        print("❌ 不在GO本体中的术语（最多显示前20个）：")
        print("="*100)
        unknown_list = list(unknown_go_set)[:20]
        for go in unknown_list:
            print(f"  {go}")
        if len(unknown_go_set) > 20:
            print(f"  ... 还有{len(unknown_go_set)-20}个未知GO")
    
    return stats, invalid_samples_detail, unknown_go_set

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 1. 解析GO本体
    go2ancestors, go2direct_parents, go_nodes = build_go_ancestor_mapping(OBO_PATH)
    
    # 2. 读取8文件和7文件
    sample_8_dict, all_8_go = read_go_csv(FILE_8_PATH)
    sample_7_dict, all_7_go = read_go_csv(FILE_7_PATH)
    
    # 3. 基础统计
    print("\n" + "="*100)
    print("📈 基础统计（唯一GO术语）")
    print("="*100)
    print(f"8文件（原始）唯一GO数：{len(all_8_go)}")
    print(f"7文件（补全后）唯一GO数：{len(all_7_go)}")
    print(f"7文件比8文件多的GO数：{len(all_7_go - all_8_go)}")
    print(f"8文件有但7文件没有的GO数：{len(all_8_go - all_7_go)}")
    
    # 4. 验证补全结果
    stats, invalid_detail, unknown_go = validate_go_completion(
        sample_8_dict, sample_7_dict, go2ancestors, go_nodes
    )
    
    # 5. 最终总结
    print("\n" + "="*100)
    print("🎯 最终验证总结")
    print("="*100)
    print(f"1. 8文件：样本数={len(sample_8_dict)}，唯一GO数={len(all_8_go)}")
    print(f"2. 7文件：样本数={len(sample_7_dict)}，唯一GO数={len(all_7_go)}")
    print(f"3. 补全正确率：{stats['valid_samples']/stats['total_samples']*100:.2f}%")
    print(f"4. 主要异常：")
    print(f"   - 缺少父术语的样本：{stats['missing_go_samples']}")
    print(f"   - 多了无关GO的样本：{stats['extra_go_samples']}")
    print(f"   - 含未知GO的样本：{stats['unknown_go_samples']}")
    print(f"   - 8文件为空但7文件有GO的样本：{len(invalid_detail.get('8文件为空但7文件有GO', []))}")
    print(f"5. 未知GO术语总数：{len(unknown_go)}")