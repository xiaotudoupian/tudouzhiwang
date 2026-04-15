import os
import sys
import math
import re
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

# ===================== 【已全部替换为 BP 数据路径】 =====================
TRUE_LABEL_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/BP数据/test_bp_labels.txt"
PRED_PATH       = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/TALE/src/tale_bp_result.txt"
IC_PATH         = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/go_ic_paper_standard_clean.pkl"

# ===================== 加载 IC（pickle 格式，完美适配） =====================
def load_go_ic():
    with open(IC_PATH, 'rb') as f:
        go2ic = pickle.load(f)
    print(f"✅ 加载 BP IC 完成：{len(go2ic)} 个GO项")
    return go2ic

# ===================== 解析真实标签（空格分隔，完美适配你的格式） =====================
def parse_true_labels():
    true_dict = {}
    with open(TRUE_LABEL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按空格分割成两部分：ID 和 GO串
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            prot_id = parts[0].strip()
            go_text = parts[1].strip()
            go_list = [go.strip() for go in go_text.split(",") if go.startswith("GO:")]
            true_dict[prot_id] = go_list
    print(f"✅ 真实标签：{len(true_dict)} 个蛋白")
    return true_dict

# ===================== 解析预测结果（稳定正则版） =====================
def parse_predictions():
    pred_dict = defaultdict(list)
    go_pattern = re.compile(r"GO:\d+")
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prot_id = line.split()[0]
            go_matches = go_pattern.findall(line)
            if not go_matches:
                continue
            go_id = go_matches[0]
            score = float(line.split()[-1])
            pred_dict[prot_id].append((go_id, score))
    print(f"✅ 预测结果：{len(pred_dict)} 个蛋白")
    return pred_dict

# ===================== 评估：Fmax / Smin / AUPR =====================
def evaluate(true_labels, pred_probs, ic_dict):
    all_go_set = set()
    for gos in true_labels.values():
        all_go_set.update(gos)
    all_go_list = sorted(all_go_set)
    mlb = MultiLabelBinarizer().fit([all_go_list])
    go2ic = {go: ic_dict.get(go, 0.0) for go in all_go_list}

    fmax = 0.0
    smin = float('inf')
    aupr = 0.0
    precisions = []
    recalls = []

    for t in range(0, 101, 1):
        threshold = t / 100.0
        total_tp = total_fp = total_fn = 0
        total_ru = 0.0
        total_mi = 0.0
        count = 0

        for prot, true_gos in true_labels.items():
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

            ru = sum(go2ic.get(g, 0) for g in true_set - pred_set)
            mi = sum(go2ic.get(g, 0) for g in pred_set - true_set)
            total_ru += ru
            total_mi += mi
            count += 1

        if count == 0:
            continue

        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
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
    sorted_pairs = sorted(zip(recalls, precisions))
    for i in range(1, len(sorted_pairs)):
        r0, p0 = sorted_pairs[i-1]
        r1, p1 = sorted_pairs[i]
        aupr += (r1 - r0) * (p0 + p1) / 2

    print("\n" + "=" * 60)
    print("                 BP 评估结果")
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