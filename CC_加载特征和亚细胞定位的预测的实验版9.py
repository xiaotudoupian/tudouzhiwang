import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import gc
import pickle
import random
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict, Counter
import math
from decimal import Decimal
# 新增：解析GO本体所需库
import obonet
import networkx as nx

# ===================== 全局配置 =====================
# 随机种子固定
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 锁定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
assert torch.cuda.is_available(), "未检测到GPU！"
current_device = torch.cuda.current_device()
device = torch.device(f"cuda:{current_device}")
print(f"当前使用GPU: {current_device} - {torch.cuda.get_device_name(current_device)}")
print(f"PyTorch版本: {torch.__version__}")

# 关键配置（对齐CAFA论文）
THRESHOLD_STEP = 0.005  # 步长0.005（对应t/1000）
THRESHOLDS = np.arange(0.0, 1.0 + THRESHOLD_STEP, THRESHOLD_STEP)  # 0.0,0.005,...,1.0（共201个阈值）

# 全局保存路径（后续在train_model中赋值）
SAVE_DIR = ""

# 修改：IC文件路径改为CSV格式CC IC文件
PRECOMPUTED_IC_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/CC数据/CC_GO_IC_values_actual.csv"

# 新增：GO层级约束配置
GO_OBO_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/数据/go.obo"
# 层级约束损失权重（可根据训练效果调整，初始0.1）
LAMBDA_CONSTRAINT = 0.1

# 显存监控
def print_gpu_memory():
    print(f"GPU显存占用：{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU显存缓存：{torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"GPU总显存：{torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

print_gpu_memory()

# ===================== 核心修改：加载CSV格式的GO IC文件 =====================
def load_precomputed_go_ic():
    """
    修改：加载CSV格式的CC GO IC文件，构建GO ID到IC值的字典
    CSV格式：GO_ID,namespace,annotation_count,frequency,ic_value
    """
    if not os.path.exists(PRECOMPUTED_IC_PATH):
        raise FileNotFoundError(f"预计算的IC文件不存在：{PRECOMPUTED_IC_PATH}")
    
    # 读取CSV文件（处理可能的BOM头）
    df = pd.read_csv(PRECOMPUTED_IC_PATH, encoding='utf-8-sig')
    
    # 检查必要列是否存在
    required_cols = ['GO_ID', 'ic_value']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV文件缺少必要列！需要{required_cols}，实际列：{df.columns.tolist()}")
    
    # 构建GO ID到IC值的字典（去除空值和无效值）
    go_ic_dict = {}
    for _, row in df.iterrows():
        go_id = row['GO_ID'].strip()
        ic_value = row['ic_value']
        
        # 过滤无效值
        if pd.isna(ic_value) or not isinstance(ic_value, (int, float)):
            continue
        if not go_id.startswith('GO:'):
            continue
        
        go_ic_dict[go_id] = float(ic_value)
    
    # 输出加载信息
    print(f"✅ 加载CSV格式IC文件完成 | 总GO术语数: {len(go_ic_dict)} | IC值范围: {min(go_ic_dict.values()):.4f} ~ {max(go_ic_dict.values()):.4f}")
    # 随机展示几个示例（避免固定示例不存在）
    sample_go_ids = list(go_ic_dict.keys())[:5] if len(go_ic_dict)>=5 else list(go_ic_dict.keys())
    for go_id in sample_go_ids[:2]:
        print(f"   示例IC值 - {go_id}: {go_ic_dict[go_id]:.4f}")
    
    return go_ic_dict

# ===================== 新增：GO层级映射构建（True Path Rule核心） =====================
def build_go_parent_child_mapping(obo_file_path):
    """
    参考你补全GO术语的代码逻辑，修复GO本体解析：
    解析GO本体，构建两个核心映射：
    - parent2children: {父term: 子term列表} （用于概率约束）
    - child2parents: {子term: 父term列表} （用于快速遍历）
    """
    if not os.path.exists(obo_file_path):
        raise FileNotFoundError(f"GO OBO文件不存在：{obo_file_path}")
    
    # 参考你的补全代码，正确解析OBO文件
    print(f"🔧 开始解析GO本体文件：{obo_file_path}")
    go_graph = obonet.read_obo(obo_file_path)
    print(f"📌 原始OBO解析结果：总节点数={len(go_graph.nodes())} | 总边数={len(go_graph.edges())}")
    
    # 定义允许的层级关系（和你的补全代码一致）
    ALLOWED_RELATIONS = {"is_a", "part_of"}
    
    # 构建子→父映射（核心：遍历所有出边，只保留允许的关系）
    child2parents = defaultdict(list)
    parent2children = defaultdict(list)
    
    # 遍历所有边（参考你的补全代码逻辑，用keys=True获取关系类型）
    for child, parent, rel_type in go_graph.edges(keys=True):
        if rel_type in ALLOWED_RELATIONS:
            # 子→父：child的父节点是parent
            child2parents[child].append(parent)
            # 父→子：parent的子节点是child
            parent2children[parent].append(child)
    
    # 去重+排序（避免重复关系）
    for child in child2parents:
        child2parents[child] = sorted(list(set(child2parents[child])))
    for parent in parent2children:
        parent2children[parent] = sorted(list(set(parent2children[parent])))
    
    # 补充无父节点的根节点（避免KeyError）
    all_terms = set(go_graph.nodes())
    for term in all_terms:
        if term not in child2parents:
            child2parents[term] = []
        if term not in parent2children:
            parent2children[term] = []
    
    print(f"✅ GO本体解析完成 | 总术语数: {len(all_terms)} | 有效层级关系数: {sum(len(v) for v in child2parents.values())}")
    print(f"   示例 - GO:0006355的父节点: {child2parents.get('GO:0006355', [])}")
    return parent2children, dict(child2parents)  # 转为普通dict，方便后续保存

# ===================== 新增：GO层级约束损失函数（True Path Rule软约束） =====================
def hierarchical_constraint_loss(logits, child2parents, term2idx, lambda_constraint=0.1):
    """
    层级约束损失：惩罚父term的预测概率 < 子term的情况
    新增：防除零处理，避免child2parents为空时报错
    """
    # 将logits转为概率（sigmoid，适配多标签分类）
    probs = torch.sigmoid(logits)
    batch_size = probs.shape[0]
    constraint_loss = 0.0
    
    # 防除零：如果child2parents为空，直接返回0损失
    if len(child2parents) == 0:
        return torch.tensor(0.0, device=probs.device)
    
    # 遍历所有子term，检查其父term的概率
    for child_term, parent_terms in child2parents.items():
        if child_term not in term2idx or not parent_terms:
            continue  # 跳过无父/不在标签空间的term
        
        # 获取子term和父term的索引
        child_idx = term2idx[child_term]
        parent_idxs = [term2idx[p] for p in parent_terms if p in term2idx]
        if not parent_idxs:
            continue
        
        # 子term的概率
        child_probs = probs[:, child_idx]  # [batch_size]
        
        # 父term的概率（取最小的父概率，更严格的约束）
        parent_probs = probs[:, parent_idxs]  # [batch_size, num_parents]
        min_parent_probs = torch.min(parent_probs, dim=1)[0]  # [batch_size]
        
        # 计算惩罚：如果子概率 > 父概率，惩罚差值（仅惩罚违反约束的情况）
        penalty = torch.maximum(child_probs - min_parent_probs, torch.tensor(0.0, device=probs.device))
        constraint_loss += torch.sum(penalty)
    
    # 归一化（新增：防除零，分母至少为1）
    denominator = max(batch_size * len(child2parents), 1)
    constraint_loss = (lambda_constraint * constraint_loss) / denominator
    return constraint_loss

# ===================== CAFA标准：Smin计算（核心修改：使用预计算IC） =====================
def calculate_cafa_smin(true_labels_dict, pred_probs_dict, mlb, ic_dict, threshold):
    """
    严格对齐CAFA Text S7的Smin计算：
    Smin(t) = √(ru(t)² + mi(t)²)
    ru(t) = 漏检的真实GO的IC总和 / 总样本数
    mi(t) = 误报的预测GO的IC总和 / 总样本数
    核心修改：使用预计算的IC字典，而非内置简化版
    """
    go_classes = mlb.classes_
    # 映射MLB的GO到预计算的IC值（未找到的IC=0）
    go2ic = {go: ic_dict.get(go, 0.0) for go in go_classes}
    
    total_ru = 0.0  # 总漏检IC
    total_mi = 0.0  # 总误报IC
    total_samples = len(true_labels_dict)
    
    for protein_id in true_labels_dict:
        if protein_id not in pred_probs_dict:
            true_go = set(true_labels_dict[protein_id])
            total_ru += sum([go2ic.get(go, 0.0) for go in true_go])
            continue
        
        true_go = set(true_labels_dict[protein_id])
        pred_go_with_prob = pred_probs_dict[protein_id]
        pred_go_filtered = set([go for go, prob in pred_go_with_prob if prob >= threshold])
        
        # 1. 计算ru(t)：漏检的真实GO的IC之和（使用预计算IC）
        missed_go = true_go - pred_go_filtered
        ru_ic = sum([go2ic.get(go, 0.0) for go in missed_go])
        total_ru += ru_ic
        
        # 2. 计算mi(t)：误报的预测GO的IC之和（使用预计算IC）
        wrong_go = pred_go_filtered - true_go
        mi_ic = sum([go2ic.get(go, 0.0) for go in wrong_go])
        total_mi += mi_ic
    
    # 归一化到每个样本
    ru_t = total_ru / total_samples if total_samples > 0 else 0.0
    mi_t = total_mi / total_samples if total_samples > 0 else 0.0
    
    # Smin(t) = 欧式距离（CAFA标准公式）
    smin_t = math.sqrt(ru_t**2 + mi_t**2)
    
    return float(smin_t)

# ===================== CAFA标准：Fmax/P/R计算（无修改） =====================
def calculate_cafa_metrics(true_labels_dict, pred_probs_dict, mlb, threshold):
    """
    严格对齐CAFA Text S7的P/R/Fmax计算（全局指标，非样本均值）：
    P(t) = 总TP / (总TP + 总FP)
    R(t) = 总TP / (总TP + 总FN)
    Fmax(t) = 2*P*R/(P+R)
    """
    go_classes = mlb.classes_
    
    global_tp = 0  # 全局TP
    global_fp = 0  # 全局FP
    global_fn = 0  # 全局FN
    
    for protein_id in true_labels_dict:
        if protein_id not in pred_probs_dict:
            true_go = set(true_labels_dict[protein_id])
            global_fn += len(true_go)
            continue
        
        true_go = set(true_labels_dict[protein_id])
        pred_go_with_prob = pred_probs_dict[protein_id]
        pred_go_filtered = set([go for go, prob in pred_go_with_prob if prob >= threshold])
        
        # 单个样本的TP/FP/FN
        tp = len(true_go & pred_go_filtered)
        fp = len(pred_go_filtered - true_go)
        fn = len(true_go - pred_go_filtered)
        
        # 累加到全局
        global_tp += tp
        global_fp += fp
        global_fn += fn
    
    # 计算全局P/R/F1
    if global_tp + global_fp == 0:
        pr = 0.0
    else:
        pr = global_tp / (global_tp + global_fp)
    
    if global_tp + global_fn == 0:
        rc = 0.0
    else:
        rc = global_tp / (global_tp + global_fn)
    
    if pr + rc == 0:
        fmax_t = 0.0
    else:
        fmax_t = 2 * pr * rc / (pr + rc)
    
    return float(pr), float(rc), float(fmax_t), global_tp, global_fp, global_fn

# ===================== CAFA标准：梯形积分AUPR计算（无修改） =====================
def calculate_aupr_by_trapezoid(pr_list, rc_list):
    """
    梯形积分法计算AUPR（输入需为按Recall升序排序的列表）
    """
    aupr = 0.0
    for i in range(len(rc_list)-1):
        # 确保Recall非递减，避免负面积
        delta_rc = max(rc_list[i+1] - rc_list[i], 0)
        aupr += 0.5 * (pr_list[i] + pr_list[i+1]) * delta_rc
    return float(aupr)

# ===================== 数据加载工具（适配新的GAF/CSV/PT格式） =====================
def load_gaf_file(gaf_path):
    """
    适配当前GAF文件格式：
    - 制表符(\t)分隔，每行5个字段
    - 第1列=蛋白ID，第2列=GO术语(逗号分隔)，其余列忽略
    """
    # 读取TSV文件，只取前2列（蛋白ID+GO术语），跳过解析错误行
    df = pd.read_csv(
        gaf_path,
        sep='\t',                # 关键：制表符分隔（匹配文件实际格式）
        header=None,             # 无表头
        usecols=[0, 1],          # 只读取第1、2列（索引0和1），忽略其他列
        names=['Protein_ID', 'GO_Terms'],  # 给列命名
        encoding='us-ascii',     # 匹配file命令查到的编码
        on_bad_lines='skip',     # 跳过错误行，避免崩溃
        low_memory=False
    )
    
    # 数据清洗：过滤无效值
    df['Protein_ID'] = df['Protein_ID'].astype(str).str.strip()  # 去空格
    df['GO_Terms'] = df['GO_Terms'].astype(str).str.strip()
    
    # 过滤空值和非GO开头的记录
    df = df[df['Protein_ID'] != '']  # 过滤空蛋白ID
    df = df[df['GO_Terms'].str.startswith('GO:', na=False)]  # 只保留GO术语
    
    # 重置索引（可选）
    df = df.reset_index(drop=True)
    
    # 打印加载结果（调试用）
    print(f"✅ 加载GAF: {os.path.basename(gaf_path)}")
    print(f"   - 总有效行数: {len(df)}")
    print(f"   - 示例数据:\n{df.head(2)}")
    
    return df

def load_subcell_csv(csv_path):
    """加载亚细胞特征CSV文件"""
    df = pd.read_csv(csv_path, header=0, encoding='utf-8', low_memory=False)
    # 清理列名
    df.columns = [col.strip() for col in df.columns]
    # 确保Protein_ID列存在
    if 'Protein_ID' not in df.columns:
        raise ValueError(f"亚细胞CSV文件缺少Protein_ID列！列名：{df.columns.tolist()}")
    df['Protein_ID'] = df['Protein_ID'].astype(str).str.strip()
    print(f"✅ 加载亚细胞CSV: {os.path.basename(csv_path)} | 样本数: {len(df)} | 列数: {df.shape[1]}")
    return df

def load_esm2_pt(pt_path):
    """加载ESM2特征PT文件"""
    pt_data = torch.load(pt_path, map_location='cpu')
    print(f"✅ 加载ESM2 PT: {os.path.basename(pt_path)} | 样本数: {len(pt_data['protein_ids'])} | 特征维度: {pt_data['cls_features'].shape[1]}")
    return pt_data

def extract_source_ids(df):
    """提取Protein ID（适配新的列名）"""
    source_ids = df['Protein_ID'].tolist()
    assert len(source_ids) == len(set(source_ids)), "存在重复Protein_ID"
    return source_ids

def extract_go_labels(df):
    """从GAF文件提取GO标签"""
    go_terms_list = []
    for go_str in df['GO_Terms']:
        if not go_str:
            go_terms = []
        else:
            # 分割后过滤掉空值，只保留GO:开头的术语
            go_terms = [go.strip() for go in go_str.split(",") if go.strip().startswith("GO:")]
        go_terms_list.append(go_terms)
    print(f"🔧 提取GO标签: 有注释{sum(1 for x in go_terms_list if x)} | 总样本{len(go_terms_list)}")
    return go_terms_list

def extract_subcell_feat(df):
    """提取亚细胞特征"""
    feat_cols = [
        'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
        'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole',
        'Golgi apparatus', 'Peroxisome', 'Peripheral', 'Transmembrane',
        'Lipid anchor', 'Soluble'
    ]
    # 检查特征列是否存在
    missing_cols = [col for col in feat_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少亚细胞特征列：{missing_cols}")
    
    feat_np = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    print(f"🔧 提取亚细胞特征: 14维 | 样本数{feat_np.shape[0]}")
    return feat_np

def align_feat_by_id(ref_ids, src_ids, src_feats):
    """按Protein ID对齐特征"""
    id2data = {pid: feat for pid, feat in zip(src_ids, src_feats)}
    
    aligned_feats = []
    missing = 0
    feat_dim = src_feats.shape[1] if len(src_feats) > 0 else 0
    
    for pid in ref_ids:
        if pid in id2data:
            aligned_feats.append(id2data[pid])
        else:
            feat_zero = np.zeros(feat_dim, dtype=np.float32)
            aligned_feats.append(feat_zero)
            missing += 1
    
    aligned_np = np.array(aligned_feats, dtype=np.float32)
    assert aligned_np.shape[0] == len(ref_ids), "特征与参考ID样本数不匹配"
    
    if missing > 0:
        print(f"⚠️  特征对齐缺失{missing}个ID，已填充0")
    return aligned_np

# ===================== 模型定义（移除网络分支） =====================
class MLPBlock(nn.Module):
    """ESM2分支的基础MLP块（保留原逻辑）"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn(x)
        x = self.dropout(x)
        return x

class SubcellBranch(nn.Module):
    """亚细胞特征分支（完全保留原逻辑，无修改）"""
    def __init__(self, input_dim=14, hidden_dim=32, dropout_rate=0.2, branch_weight=0.9):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.branch_weight = branch_weight  
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.bn2(x)
        x = x * self.branch_weight
        return x

class ESM2_Subcell_MLP(nn.Module):
    """【消融版】仅保留ESM2+亚细胞分支的模型"""
    def __init__(self, esm_input_dim=2560, subcell_input_dim=14, 
                 hidden_dim=512, num_go_terms=4000, dropout_rate=0.3):
        super().__init__()
        self.esm_mlp1 = MLPBlock(esm_input_dim, hidden_dim, dropout_rate)
        self.esm_mlp2 = MLPBlock(hidden_dim, hidden_dim, dropout_rate)
        self.esm_mlp3 = MLPBlock(hidden_dim, hidden_dim, dropout_rate)  
        
        self.subcell_branch = SubcellBranch(subcell_input_dim, hidden_dim=32, branch_weight=0.9)
        
        # 【修改】注意力机制输入维度：仅ESM2(512) + 亚细胞(32)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # 【修改】仅2个分支的注意力权重
            nn.Softmax(dim=1)
        )
        
        # 【修改】输出层输入维度：仅ESM2(512) + 亚细胞(32)
        self.output_layer = nn.Linear(hidden_dim + 32, num_go_terms)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0.0)
        
    def forward(self, esm_feats, subcell_feats):
        checkpoint_kwargs = {"use_reentrant": False} if torch.__version__ >= "2.0.0" else {}
        
        esm_h = checkpoint(self.esm_mlp1, esm_feats,** checkpoint_kwargs)
        esm_h = checkpoint(self.esm_mlp2, esm_h, **checkpoint_kwargs)
        esm_h = checkpoint(self.esm_mlp3, esm_h,** checkpoint_kwargs)  
        
        subcell_h = checkpoint(self.subcell_branch, subcell_feats, **checkpoint_kwargs)
        
        # 【修改】仅拼接ESM2+亚细胞特征
        concat_h = torch.cat([esm_h, subcell_h], dim=1)
        attn_weights = self.attention(concat_h)
        esm_weight = attn_weights[:, 0:1]
        subcell_weight = attn_weights[:, 1:2]
        
        esm_h_weighted = esm_h * esm_weight
        subcell_h_weighted = subcell_h * subcell_weight
        # 【修改】仅融合两个分支
        fused_h = torch.cat([esm_h_weighted, subcell_h_weighted], dim=1)
        
        return self.output_layer(fused_h)

# ===================== 损失函数+硬负例挖掘（保留原逻辑） =====================
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, class_freq=None, common_weight=1.2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.common_weight = common_weight
        self.class_weights = torch.tensor([
            self.common_weight if freq > 100 else 1.0 
            for freq in class_freq
        ], dtype=torch.float32).to(device)
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        focal_loss = focal_loss * self.class_weights
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def update_sample_weights(model, train_loader, sample_weights, train_class_counts):
    model.eval()
    hard_neg_mask = torch.zeros_like(sample_weights, dtype=torch.bool)
    with torch.no_grad():
        for batch_idx, (esm_feat, subcell_feat, labels) in enumerate(train_loader):
            esm_feat = esm_feat.to(device, non_blocking=True)
            subcell_feat = subcell_feat.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(esm_feat, subcell_feat)
            preds = torch.sigmoid(outputs) > 0.45
            rare_class_mask = train_class_counts <= 5
            true_rare = labels[:, rare_class_mask] == 1
            pred_rare_wrong = preds[:, rare_class_mask] == 0
            has_rare_wrong = torch.any(true_rare & pred_rare_wrong, dim=1)
            start_idx = batch_idx * train_loader.batch_size
            end_idx = start_idx + esm_feat.size(0)
            hard_neg_mask[start_idx:end_idx] = has_rare_wrong
    sample_weights[hard_neg_mask] *= 2.0
    return sample_weights

# ===================== 标签预处理（无修改） =====================
def prepare_labels(all_go_terms_list, mlb=None):
    if mlb is None:
        mlb = MultiLabelBinarizer()
        labels_matrix = mlb.fit_transform(all_go_terms_list)
    else:
        labels_matrix = mlb.transform(all_go_terms_list)
    labels_tensor = torch.tensor(labels_matrix, dtype=torch.float32)
    print(f"✅ 标签矩阵: 形状{labels_tensor.shape} | GO总数{len(mlb.classes_)} | 非零标签{torch.sum(labels_tensor)}")
    return labels_tensor, mlb

# ===================== CAFA标准：论文级指标计算（核心修改：使用预计算IC） =====================
def calculate_paper_metrics(model, dataloader, ids_list, mlb, ic_dict, set_type="Test"):
    """
    完全对齐CAFA Text S7的指标计算（最终版）
    核心修改：使用预计算的IC字典计算Smin
    """
    model.eval()
    # 1. 收集所有样本的真实标签和预测概率
    true_labels_dict = {}  # {protein_id: [GO:xxxx, ...]}
    pred_probs_dict = {}   # {protein_id: [(GO:xxxx, prob), ...]}
    
    go_classes = mlb.classes_
    batch_start = 0
    
    with torch.no_grad():
        for esm, subcell, labels in dataloader:
            esm = esm.to(device, non_blocking=True)
            subcell = subcell.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(esm, subcell)
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # 处理当前批次的每个样本
            batch_size = esm.size(0)
            batch_ids = ids_list[batch_start:batch_start+batch_size]
            
            for idx in range(batch_size):
                protein_id = batch_ids[idx]
                # 真实GO术语
                true_go_idx = np.where(labels_np[idx] == 1)[0]
                true_go = [go_classes[i] for i in true_go_idx]
                true_labels_dict[protein_id] = true_go
                
                # 预测GO术语+概率（按概率降序）
                pred_prob = probs[idx]
                pred_go_with_prob = [(go_classes[i], float(pred_prob[i])) for i in range(len(go_classes))]
                pred_go_with_prob = sorted(pred_go_with_prob, key=lambda x: x[1], reverse=True)
                pred_probs_dict[protein_id] = pred_go_with_prob
            
            batch_start += batch_size
    
    # 2. 遍历所有阈值（0-1，步长0.005），计算CAFA标准指标
    pr_list = []
    rc_list = []
    fmax_list = []
    smin_list = []
    all_thresholds = []
    
    # 最优值初始化
    opt_t = 0
    opt_pr = 0.0
    opt_rc = 0.0
    opt_fmax = 0.0
    opt_smin = float('inf')
    opt_tp = 0
    
    # 保存阈值指标到文件的内容
    roc_line = ""
    
    for t in range(0, 1001, 5):
        threshold = t / 1000.0
        # 计算CAFA标准的P/R/Fmax
        pr, rc, fmax_t, tp, fp, fn = calculate_cafa_metrics(
            true_labels_dict, pred_probs_dict, mlb, threshold
        )
        # 计算CAFA标准的Smin（使用预计算IC）
        smin_t = calculate_cafa_smin(
            true_labels_dict, pred_probs_dict, mlb, ic_dict, threshold
        )
        
        # 记录指标
        pr_list.append(pr)
        rc_list.append(rc)
        fmax_list.append(fmax_t)
        smin_list.append(smin_t)
        all_thresholds.append(threshold)
        
        # 格式化输出
        roc_line += f"t={Decimal(threshold).quantize(Decimal('0.00000'))} " + \
                    f"pr={Decimal(pr).quantize(Decimal('0.00000'))} " + \
                    f"rc={Decimal(rc).quantize(Decimal('0.00000'))} " + \
                    f"fmax={Decimal(fmax_t).quantize(Decimal('0.00000'))} " + \
                    f"smin={Decimal(smin_t).quantize(Decimal('0.00000'))} " + \
                    f"tp={int(tp)} fp={int(fp)} fn={int(fn)}\n"
        
        # 更新最优FMAX（越大越好）
        if fmax_t > opt_fmax:
            opt_t = t
            opt_pr = pr
            opt_rc = rc
            opt_fmax = fmax_t
            opt_tp = tp
        
        # 更新最优Smin（越小越好）
        if smin_t < opt_smin:
            opt_smin = smin_t
    
    # 3. 修复AUPR计算：先按Recall排序，再梯形积分
    # 合并(rc, pr)并按rc升序排序
    rc_pr_pairs = sorted(zip(rc_list, pr_list), key=lambda x: x[0])
    sorted_rc = [x[0] for x in rc_pr_pairs]
    sorted_pr = [x[1] for x in rc_pr_pairs]
    # 梯形积分计算AUPR
    aupr = calculate_aupr_by_trapezoid(sorted_pr, sorted_rc)
    
    # 4. 保存阈值指标文件
    roc_file = os.path.join(SAVE_DIR, f"{set_type.lower()}_roc_metrics.txt")
    with open(roc_file, "w") as f:
        f.write(roc_line)
    print(f"✅ {set_type}阈值指标保存至: {roc_file}")
    
    return {
        "opt_t": opt_t,                  # 最优阈值对应的t值（0-1000）
        "opt_threshold": opt_t/1000.0,   # 最优阈值（0-1）
        "opt_pr": round(opt_pr, 6),      # 最优精确率
        "opt_rc": round(opt_rc, 6),      # 最优召回率
        "opt_fmax": round(opt_fmax, 6),  # 最优FMAX
        "opt_smin": round(opt_smin, 6),  # 最优Smin（CAFA标准）
        "opt_tp": opt_tp,                # 最优阈值下的全局TP
        "aupr": round(aupr, 6),          # 修复后的AUPR
        "all_thresholds": all_thresholds,
        "all_pr": pr_list,
        "all_rc": rc_list,
        "all_fmax": fmax_list,
        "all_smin": smin_list
    }

# ===================== 可视化（对齐论文指标） =====================
def plot_paper_pr_curve(pr_list, rc_list, aupr, save_dir, set_type="Test"):
    """绘制论文风格的PR曲线（梯形积分AUPR）"""
    # 先排序
    rc_pr_pairs = sorted(zip(rc_list, pr_list), key=lambda x: x[0])
    sorted_rc = [x[0] for x in rc_pr_pairs]
    sorted_pr = [x[1] for x in rc_pr_pairs]
    
    plt.figure(figsize=(10, 8))
    plt.plot(sorted_rc, sorted_pr, '#2E86AB', linewidth=2.5, label=f'AUPR = {aupr:.5f} (Trapezoid)')
    plt.fill_between(sorted_rc, sorted_pr, alpha=0.2, color='#2E86AB')
    plt.title(f'{set_type} Precision-Recall Curve (CAFA Standard)', fontsize=16, fontweight='bold')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{set_type.lower()}_paper_pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {set_type}论文风格PR曲线保存至: {save_dir}")

def plot_smin_fmax_curve(all_thresholds, all_smin, all_fmax, opt_threshold, opt_fmax, opt_smin, save_dir, set_type="Test"):
    """绘制Smin-FMAX曲线（对齐CAFA论文）"""
    plt.figure(figsize=(12, 7))
    # 双Y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # FMAX曲线
    ax1.plot(all_thresholds, all_fmax, '#C73E1D', linewidth=2.5, marker='.', label='FMAX', markersize=4)
    ax1.set_xlabel('Classification Threshold', fontsize=14)
    ax1.set_ylabel('FMAX', fontsize=14, color='#C73E1D')
    ax1.tick_params(axis='y', labelcolor='#C73E1D')
    ax1.set_xlim([0.0, 1.0])
    
    # Smin曲线
    ax2.plot(all_thresholds, all_smin, '#2E86AB', linewidth=2.5, marker='.', label='Smin', markersize=4)
    ax2.set_ylabel('Smin (CAFA Standard)', fontsize=14, color='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#2E86AB')
    
    # 标注最优值
    ax1.scatter(opt_threshold, opt_fmax, color='red', s=100, zorder=5, label=f'Opt FMAX={opt_fmax:.5f}')
    ax2.scatter(opt_threshold, opt_smin, color='blue', s=100, zorder=5, label=f'Opt Smin={opt_smin:.5f}')
    ax1.axvline(x=opt_threshold, color='gray', linestyle='--', alpha=0.7)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.title(f'{set_type} Smin vs FMAX (CAFA Standard)', fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{set_type.lower()}_smin_fmax_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {set_type} Smin-FMAX曲线保存至: {save_dir}")

# ===================== 数据加载主函数（适配新的GAF/CSV/PT格式） =====================
def load_dataset(data_dir, set_type):
    """加载train/val/test集（适配新的GAF/CSV/PT格式）"""
    print(f"\n===== 加载{set_type.upper()}集 =====")
    
    # 文件路径
    gaf_path = os.path.join(data_dir, f"{set_type}.gaf")
    subcell_path = os.path.join(data_dir, f"{set_type}.csv")
    esm2_path = os.path.join(data_dir, f"{set_type}.pt")
    
    # 加载GO标签（GAF文件）
    go_df = load_gaf_file(gaf_path)
    ref_ids = extract_source_ids(go_df)
    go_terms_list = extract_go_labels(go_df)
    
    # 加载亚细胞特征（CSV文件）
    subcell_df = load_subcell_csv(subcell_path)
    subcell_feat = extract_subcell_feat(subcell_df)
    subcell_ids = extract_source_ids(subcell_df)
    subcell_feat_aligned = align_feat_by_id(ref_ids, subcell_ids, subcell_feat)
    
    # 加载ESM2特征（PT文件）
    esm2_data = load_esm2_pt(esm2_path)
    esm2_feat = esm2_data['cls_features'].numpy().astype(np.float32)
    esm2_ids = esm2_data['protein_ids']
    esm2_feat_aligned = align_feat_by_id(ref_ids, esm2_ids, esm2_feat)
    
    # 验证样本数匹配
    assert len(ref_ids) == esm2_feat_aligned.shape[0] == subcell_feat_aligned.shape[0], \
        f"{set_type}集特征样本数不匹配"
    
    print(f"✅ {set_type.upper()}集加载完成: 样本数{len(ref_ids)} | ESM2(2560维) + 亚细胞(14维)")
    
    return ref_ids, esm2_feat_aligned, subcell_feat_aligned, go_terms_list

# ===================== 模型训练主函数（适配新数据源） =====================
def train_model():
    # 全局保存路径（区分消融实验）
    global SAVE_DIR
    # 修改数据目录为新的CC数据路径
    DATA_DIR = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/CC数据/split_data"
    BASE_DIR = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/CC数据"
    SAVE_DIR = os.path.join(BASE_DIR, "最终模型model_results_ablation_no_network")
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "go_best_model_ablation_no_network.pt")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📂 消融实验结果保存至: {SAVE_DIR}")

    # 1. 加载数据集（适配新格式）
    train_ids, train_esm2, train_subcell, train_go = load_dataset(DATA_DIR, "train")
    val_ids, val_esm2, val_subcell, val_go = load_dataset(DATA_DIR, "val")
    test_ids, test_esm2, test_subcell, test_go = load_dataset(DATA_DIR, "test")

    # 2. 准备GO标签矩阵
    train_labels, mlb = prepare_labels(train_go)
    val_labels, _ = prepare_labels(val_go, mlb)
    test_labels, _ = prepare_labels(test_go, mlb)
    num_go_terms = len(mlb.classes_)
    train_class_counts = train_labels.sum(dim=0).cpu().numpy()

    # 3. 加载预计算的CAFA标准IC字典
    go_ic_dict = load_precomputed_go_ic()

    # 4. 加载GO本体，构建父子映射
    parent2children, child2parents = build_go_parent_child_mapping(GO_OBO_PATH)
    term2idx = {term: idx for idx, term in enumerate(mlb.classes_)}
    print(f"✅ GO层级映射构建完成 | 标签空间GO数: {len(term2idx)} | 本体中GO数: {len(child2parents)}")

    # 5. 转换特征为Tensor
    train_esm2 = torch.tensor(train_esm2, dtype=torch.float32)
    train_subcell = torch.tensor(train_subcell, dtype=torch.float32)
    
    val_esm2 = torch.tensor(val_esm2, dtype=torch.float32)
    val_subcell = torch.tensor(val_subcell, dtype=torch.float32)
    
    test_esm2 = torch.tensor(test_esm2, dtype=torch.float32)
    test_subcell = torch.tensor(test_subcell, dtype=torch.float32)

    # 6. 创建Dataset和DataLoader
    train_dataset = TensorDataset(train_esm2, train_subcell, train_labels)
    val_dataset = TensorDataset(val_esm2, val_subcell, val_labels)
    test_dataset = TensorDataset(test_esm2, test_subcell, test_labels)

    # 7. 加权随机采样
    sample_weights = torch.ones(train_labels.shape[0], dtype=torch.float32)
    extreme_rare_mask = train_class_counts <= 2
    general_rare_mask = (train_class_counts > 2) & (train_class_counts <= 5)
    has_extreme_rare = torch.any(train_labels[:, extreme_rare_mask], dim=1).cpu()
    has_general_rare = (torch.any(train_labels[:, general_rare_mask], dim=1) & ~has_extreme_rare).cpu()
    sample_weights[has_extreme_rare] *= 3.5
    sample_weights[has_general_rare] *= 3.0
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    # 8. DataLoader配置
    batch_size = 96
    num_workers = 4 if os.cpu_count() >=4 else 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # 9. 初始化消融版模型
    model = ESM2_Subcell_MLP(
        esm_input_dim=2560, subcell_input_dim=14,
        hidden_dim=512, num_go_terms=num_go_terms, dropout_rate=0.3
    ).to(device)
    print(f"\n✅ 消融版模型初始化完成 | 输出GO数: {num_go_terms}")
    print("模型显存状态：")
    print_gpu_memory()

    # 10. 损失函数+优化器+调度器
    criterion = WeightedFocalLoss(
        alpha=0.8, gamma=2.0, class_freq=train_class_counts, common_weight=1.2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6
    )

    # 11. 训练参数
    num_epochs = 200
    patience = 20
    best_valid_loss = float('inf')
    early_stop_counter = 0
    
    train_loss_hist = []
    valid_loss_hist = []
    print(f"\n===== 开始消融实验训练（移除网络分支 + CAFA标准 + GO层级约束） =====")
    print(f"📌 轮次: {num_epochs} | 批大小: {batch_size} | 早停: {patience} | 初始LR: 9e-4")
    print(f"📌 阈值遍历: 0-1 (步长0.005) | CAFA标准IC/Smin/Fmax计算")
    print(f"📌 使用预计算IC字典: {PRECOMPUTED_IC_PATH}")
    print(f"📌 GO层级约束权重: {LAMBDA_CONSTRAINT} | OBO文件: {GO_OBO_PATH}")
    print(f"📌 消融实验：移除所有网络特征分支（仅保留ESM2+亚细胞）")

    # 12. 训练循环
    for epoch in range(num_epochs):
        if epoch % 2 == 0 and epoch > 0:
            sample_weights = update_sample_weights(model, train_loader, sample_weights, train_class_counts)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=True, drop_last=True
            )
            print(f"🔄 第{epoch}轮更新硬负例权重，重新创建采样器")

        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_running_hc_loss = 0.0
        for batch_idx, (esm, subcell, labels) in enumerate(train_loader):
            if esm.size(0) == 1:
                continue
            esm = esm.to(device, non_blocking=True)
            subcell = subcell.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(esm, subcell)
            # 计算主损失
            main_loss = criterion(outputs, labels)
            # 计算层级约束损失
            hc_loss = hierarchical_constraint_loss(outputs, child2parents, term2idx, LAMBDA_CONSTRAINT)
            # 总损失
            total_loss = main_loss + hc_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_running_loss += total_loss.item() * esm.size(0)
            train_running_hc_loss += hc_loss.item() * esm.size(0)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], " +
                      f"Total Loss: {total_loss.item():.6f}, Main Loss: {main_loss.item():.6f}, HC Loss: {hc_loss.item():.6f}")
            if (batch_idx + 1) % 40 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        train_epoch_hc_loss = train_running_hc_loss / len(train_loader.dataset)
        train_loss_hist.append(train_epoch_loss)
        print(f"\n训练集 - Epoch {epoch+1}: 总损失={train_epoch_loss:.6f}, 层级约束损失={train_epoch_hc_loss:.6f}")
        
        # 验证阶段
        model.eval()
        valid_running_loss = 0.0
        with torch.no_grad():
            for esm, subcell, labels in val_loader:
                esm = esm.to(device, non_blocking=True)
                subcell = subcell.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(esm, subcell)
                main_loss = criterion(outputs, labels)
                hc_loss = hierarchical_constraint_loss(outputs, child2parents, term2idx, LAMBDA_CONSTRAINT)
                total_loss = main_loss + hc_loss
                valid_running_loss += total_loss.item() * esm.size(0)
        
        valid_epoch_loss = valid_running_loss / len(val_loader.dataset)
        valid_loss_hist.append(valid_epoch_loss)
        print(f"验证集 - Epoch {epoch+1}: 总损失={valid_epoch_loss:.6f}")

        # 早停+保存最佳模型
        scheduler.step(valid_epoch_loss)
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            torch.save({
                'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_valid_loss,
                'mlb_classes': mlb.classes_, 'go_ic_dict': go_ic_dict,
                'child2parents': child2parents, 'term2idx': term2idx
            }, MODEL_SAVE_PATH)
            print(f"📈 保存最佳消融模型至: {MODEL_SAVE_PATH} (损失: {best_valid_loss:.6f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"⚠️  早停计数: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"🛑 早停触发！")
                break
        
        gc.collect()
        torch.cuda.empty_cache()
        print("当前显存状态：")
        print_gpu_memory()

    # 13. 测试集最终评估
    print(f"\n===== 测试集最终评估 =====")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 计算CAFA标准指标
    test_metrics = calculate_paper_metrics(model, test_loader, test_ids, mlb, go_ic_dict, set_type="Test")
    
    # 打印最终结果
    print(f"\n===== 测试集最终结果（消融实验 - 移除网络分支 + CAFA标准 + GO层级约束） =====")
    print(f"最优阈值t值: {test_metrics['opt_t']} (阈值={test_metrics['opt_threshold']:.3f})")
    print(f"最优精确率 (opt_pr): {test_metrics['opt_pr']:.6f}")
    print(f"最优召回率 (opt_rc): {test_metrics['opt_rc']:.6f}")
    print(f"最优FMAX: {test_metrics['opt_fmax']:.6f}")
    print(f"最优Smin (CAFA标准): {test_metrics['opt_smin']:.6f}")
    print(f"AUPR (梯形积分): {test_metrics['aupr']:.6f}")
    print(f"最优阈值下TP数: {test_metrics['opt_tp']}")

    # 14. 测试集可视化
    plot_paper_pr_curve(
        test_metrics['all_pr'], test_metrics['all_rc'], 
        test_metrics['aupr'], SAVE_DIR, set_type="Test"
    )
    plot_smin_fmax_curve(
        test_metrics['all_thresholds'], test_metrics['all_smin'], test_metrics['all_fmax'],
        test_metrics['opt_threshold'], test_metrics['opt_fmax'], test_metrics['opt_smin'],
        SAVE_DIR, set_type="Test"
    )

    # 15. 保存测试集指标
    test_metrics_save = {
        'opt_t': test_metrics['opt_t'],
        'opt_threshold': test_metrics['opt_threshold'],
        'opt_pr': test_metrics['opt_pr'],
        'opt_rc': test_metrics['opt_rc'],
        'opt_fmax': test_metrics['opt_fmax'],
        'opt_smin': test_metrics['opt_smin'],
        'aupr': test_metrics['aupr'],
        'opt_tp': test_metrics['opt_tp']
    }
    with open(os.path.join(SAVE_DIR, 'test_cafa_metrics_ablation_no_network.pkl'), 'wb') as f:
        pickle.dump(test_metrics_save, f)

    print(f"\n===== 消融实验训练完成（移除网络分支） =====")
    print(f"最佳模型: {MODEL_SAVE_PATH}")
    print(f"CAFA指标文件: {os.path.join(SAVE_DIR, 'test_roc_metrics.txt')}")
    print(f"测试集可视化: {SAVE_DIR}")

if __name__ == "__main__":
    train_model()