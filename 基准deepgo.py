import os
import sys
import math
import pickle
from collections import defaultdict

# ===================== 配置 BP 数据路径 =====================
TRUE_LABEL_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/BP数据/test_bp_labels.txt"
PRED_PATH       = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/BP数据/test_bp_preds_bp.tsv"
IC_PATH         = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/go_ic_paper_standard_clean.pkl"

# ===================== 1. 加载 IC (Pickle 格式) =====================
def load_go_ic():
    """读取原始的 Pickle 格式 IC 文件"""
    with open(IC_PATH, 'rb') as f:
        go2ic = pickle.load(f)
    print(f"✅ 加载 IC 完成：{len(go2ic)} 个GO项")
    return go2ic

# ===================== 2. 解析真实标签 (空格分隔格式) =====================
def parse_true_labels():
    """解析 test_bp_labels.txt: ID GO1,GO2,GO3"""
    true_dict = {}
    with open(TRUE_LABEL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 使用 maxsplit=1 分割，防止ID或GO里有意外空格
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            prot_id = parts[0].strip()
            go_text = parts[1].strip()
            # 按逗号分割GO术语
            go_list = [go.strip() for go in go_text.split(",") if go.startswith("GO:")]
            true_dict[prot_id] = go_list
    print(f"✅ 真实标签：{len(true_dict)} 个蛋白")
    return true_dict

# ===================== 3. 解析预测结果 (DeepGO TSV 格式) =====================
def parse_predictions():
    """解析 test_bp_preds_bp.tsv: ID\tGO\tScore"""
    pred_dict = defaultdict(list)
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            prot_id = parts[0].strip()
            go_id = parts[1].strip()
            try:
                score = float(parts[2].strip())
            except ValueError:
                continue
            pred_dict[prot_id].append((go_id, score))
    print(f"✅ 预测结果：{len(pred_dict)} 个蛋白")
    return pred_dict

# ===================== 4. 评估核心逻辑 (Fmax / Smin / AUPR) =====================
def evaluate(true_labels, pred_probs, ic_dict):
    # 找出交集蛋白
    common_prots = set(true_labels.keys()) & set(pred_probs.keys())
    print(f"⚠️  用于评估的共同蛋白数：{len(common_prots)}")
    
    if len(common_prots) == 0:
        print("❌ 错误：真实标签和预测结果没有共同的蛋白ID！")
        return

    fmax = 0.0
    smin = float('inf')
    aupr = 0.0
    precisions = []
    recalls = []

    # 遍历阈值
    for t in range(0, 101, 1):
        threshold = t / 100.0
        total_tp = total_fp = total_fn = 0
        total_ru = 0.0
        total_mi = 0.0
        count = 0

        for prot in common_prots:
            true_gos = true_labels[prot]
            true_set = set(true_gos)
            pred_set = set()
            
            for go_id, score in pred_probs.get(prot, []):
                if score >= threshold:
                    pred_set.add(go_id)

            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            ru = sum(ic_dict.get(g, 0.0) for g in true_set - pred_set)
            mi = sum(ic_dict.get(g, 0.0) for g in pred_set - true_set)
            total_ru += ru
            total_mi += mi
            count += 1

        if count == 0:
            continue

        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        ru_avg = total_ru / count
        mi_avg = total_mi / count
        s_val = math.sqrt(ru_avg**2 + mi_avg**2)

        precisions.append(p)
        recalls.append(r)

        if f1 > fmax:
            fmax = f1
        if s_val < smin:
            smin = s_val

    # 计算 AUPR
    sorted_pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
    for i in range(1, len(sorted_pairs)):
        r0, p0 = sorted_pairs[i-1]
        r1, p1 = sorted_pairs[i]
        aupr += (r1 - r0) * (p0 + p1) / 2

    print("\n" + "=" * 60)
    print("         DeepGO-SE BP 评估结果")
    print("=" * 60)
    print(f"Fmax  = {fmax:.6f}")
    print(f"Smin  = {smin:.6f}")
    print(f"AUPR  = {aupr:.6f}")
    print("=" * 60)

# ===================== main =====================
def main():
    true_dict = parse_true_labels()
    pred_dict = parse_predictions()
    ic_dict = load_go_ic()
    evaluate(true_dict, pred_dict, ic_dict)

if __name__ == "__main__":
    main()