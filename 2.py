import pickle
from collections import defaultdict

# ===================== 核心修正：TARGET_NAMESPACE改为bp =====================
OBO_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/数据/go.obo"
GO2IND_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/TALE/data/me/go2ind_mf.pickle"
# 输出文件名改为bp_go_1.pickle
OUTPUT_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/TALE/data/me/bp_go_1.pickle"
# 分支改为biological_process（BP）
TARGET_NAMESPACE = "biological_process"  

# ===================== 解析逻辑不变 =====================
def parse_go_obo(obo_path):
    go_info = {}
    go_children = defaultdict(list)
    current_go = None
    current_namespace = None
    current_parents = []
    current_name = None

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[Term]"):
                if current_go and current_namespace:
                    go_info[current_go] = {
                        "name": current_name,
                        "namespace": current_namespace,
                        "parents": current_parents
                    }
                    for parent in current_parents:
                        go_children[parent].append(current_go)
                current_go = None
                current_namespace = None
                current_parents = []
                current_name = None
                continue
            if line.startswith("id: GO:"):
                current_go = line.split(": ")[1].strip()
            elif line.startswith("name: "):
                current_name = line.split(": ")[1].strip()
            elif line.startswith("namespace: "):
                current_namespace = line.split(": ")[1].strip()
            elif line.startswith("is_a: GO:"):
                parent_go = line.split(": ")[1].split(" ! ")[0].strip()
                current_parents.append(parent_go)
    if current_go and current_namespace:
        go_info[current_go] = {
            "name": current_name,
            "namespace": current_namespace,
            "parents": current_parents
        }
        for parent in current_parents:
            go_children[parent].append(current_go)
    print(f"OBO文件解析完成：共{len(go_info)}个GO术语")
    return go_info, go_children

def generate_tale_ontology():
    go_info, go_children = parse_go_obo(OBO_PATH)
    with open(GO2IND_PATH, "rb") as f:
        go2ind = pickle.load(f)
    print(f"你的数据中包含{len(go2ind)}个GO术语")
    
    tale_ontology = {}
    for go_term, ind in go2ind.items():
        if go_term not in go_info:
            tale_ontology[go_term] = {
                "name": go_term,
                "ind": ind,
                "father": [],
                "child": []
            }
            continue
        # 只保留BP分支
        if go_info[go_term]["namespace"] != TARGET_NAMESPACE:
            continue
        fathers = [p for p in go_info[go_term]["parents"] if p in go2ind]
        children = [c for c in go_children.get(go_term, []) if c in go2ind]
        tale_ontology[go_term] = {
            "name": go_info[go_term]["name"],
            "ind": ind,
            "father": fathers,
            "child": children
        }
    
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(tale_ontology, f)
    
    total_go = len(tale_ontology)
    has_father = sum(1 for v in tale_ontology.values() if len(v["father"]) > 0)
    has_child = sum(1 for v in tale_ontology.values() if len(v["child"]) > 0)
    
    print("\n=== BP本体文件生成统计 ===")
    print(f"最终生成{total_go}个BP分支的GO术语")
    print(f"包含父节点的GO术语：{has_father}个")
    print(f"包含子节点的GO术语：{has_child}个")
    print(f"输出路径：{OUTPUT_PATH}")

if __name__ == "__main__":
    generate_tale_ontology()