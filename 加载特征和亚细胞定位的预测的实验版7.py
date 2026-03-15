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
from collections import defaultdict

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

# 显存监控
def print_gpu_memory():
    print(f"GPU显存占用：{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU显存缓存：{torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"GPU总显存：{torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

print_gpu_memory()

# ===================== 通用CSV特征加载工具 =====================
def load_csv_file(csv_path):
    df = pd.read_csv(csv_path, header=0, encoding='utf-8', low_memory=False, dtype=str)
    df.columns = [col.strip() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.fillna("")
    print(f"✅ 加载CSV: {os.path.basename(csv_path)} | 样本数: {len(df)} | 列数: {df.shape[1]}")
    return df

def extract_source_ids(df):
    source_ids = df['Source_ID'].tolist()
    assert len(source_ids) == len(set(source_ids)), "CSV存在重复Source_ID"
    return source_ids

def extract_network_feat(df):
    feat_cols = [f"network_feat_{i}" for i in range(32)]
    feat_np = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    species = df['species'].tolist()  # 无空物种，所有值有效
    print(f"🔧 提取网络特征: 32维 | 样本数{feat_np.shape[0]} | 物种数{len(set(species))}")
    return feat_np, species

def extract_esm2_feat(df):
    feat_cols = [f"esm2_feat_{i}" for i in range(2560)]
    feat_np = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    print(f"🔧 提取ESM2特征: 2560维 | 样本数{feat_np.shape[0]}")
    return feat_np

def extract_subcell_feat(df):
    feat_cols = [
        'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
        'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole',
        'Golgi apparatus', 'Peroxisome', 'Peripheral', 'Transmembrane',
        'Lipid anchor', 'Soluble'
    ]
    feat_np = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    print(f"🔧 提取亚细胞特征: 14维 | 样本数{feat_np.shape[0]}")
    return feat_np

def extract_go_labels(df):
    go_terms_list = []
    for go_str in df['GO_Terms']:
        go_terms = [go for go in go_str.split(",") if go.strip().startswith("GO:")] if go_str else []
        go_terms_list.append(go_terms)
    print(f"🔧 提取GO标签: 有注释{sum(1 for x in go_terms_list if x)} | 总样本{len(go_terms_list)}")
    return go_terms_list

def align_feat_by_id(ref_ids, src_df, extract_fun):
    id2data = {}
    src_data = extract_fun(src_df)
    src_ids = extract_source_ids(src_df)
    if isinstance(src_data, tuple) and len(src_data) == 2:
        src_feats, src_species = src_data
        for pid, feat, sp in zip(src_ids, src_feats, src_species):
            id2data[pid] = (feat, sp)
    else:
        src_feats = src_data
        for pid, feat in zip(src_ids, src_feats):
            id2data[pid] = feat

    aligned_feats = []
    aligned_species = [] if isinstance(src_data, tuple) and len(src_data) == 2 else None
    missing = 0
    for pid in ref_ids:
        if pid in id2data:
            if aligned_species is not None:
                feat, sp = id2data[pid]
                aligned_feats.append(feat)
                aligned_species.append(sp)
            else:
                aligned_feats.append(id2data[pid])
        else:
            feat_zero = np.zeros(src_feats.shape[1], dtype=np.float32)
            aligned_feats.append(feat_zero)
            if aligned_species is not None:
                aligned_species.append("")  # 理论上无缺失，仅兜底
            missing += 1
    aligned_np = np.array(aligned_feats, dtype=np.float32)
    assert aligned_np.shape[0] == len(ref_ids), "特征与参考ID样本数不匹配"
    if missing > 0:
        print(f"⚠️  特征对齐缺失{missing}个ID，已填充0")
    if aligned_species is not None:
        return aligned_np, aligned_species
    else:
        return aligned_np

# ===================== 模型定义（核心：修复BatchNorm单样本 + 5物种专属MLP） =====================
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

class SpeciesSpecificNetworkBranch(nn.Module):
    """
    无空物种专属：5个物种=5个专属MLP子分支（无通用分支）
    【已修复核心报错】：单样本训练时自动跳过BatchNorm，和MLPBlock/SubcellBranch逻辑对齐
    结构与亚细胞分支对称（两层MLP+BN+Dropout），输出统一32维，消除物种间特征差异
    端到端训练：利用大模型的GO标签监督，学习各物种网络特征→GO功能的映射
    """
    def __init__(self, num_species, input_dim=32, hidden_dim=32, dropout_rate=0.2, branch_weight=0.92):
        super().__init__()
        self.branch_weight = branch_weight
        self.num_species = num_species
        self.species_mlps = nn.ModuleList([
            self._build_species_mlp(input_dim, hidden_dim, dropout_rate) 
            for _ in range(self.num_species)
        ])
        # 新增：物种间共享特征层（1层MLP，实现通用特征共享）
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    def _build_species_mlp(self, in_dim, hid_dim, drop_rate):
        mlp = nn.ModuleDict({
            "linear1": nn.Linear(in_dim, hid_dim),
            "bn1": nn.BatchNorm1d(hid_dim),
            "dropout1": nn.Dropout(drop_rate),
            "linear2": nn.Linear(hid_dim, hid_dim),
            "bn2": nn.BatchNorm1d(hid_dim),
            "dropout2": nn.Dropout(drop_rate),
            "shortcut": nn.Linear(in_dim, hid_dim)  # 残差捷径层
        })
        return mlp
        
    def forward(self, network_feat, species_ids):
        batch_size = network_feat.size(0)
        processed_feat = torch.zeros((batch_size, 32), device=network_feat.device, dtype=torch.float32)
        
        for sp_id in range(self.num_species):
            sp_mask = (species_ids == sp_id)
            if torch.any(sp_mask):
                x = network_feat[sp_mask]
                shortcut = self.species_mlps[sp_id]["shortcut"](x)  # 残差捷径
                # 第一层MLP
                x = F.relu(self.species_mlps[sp_id]["linear1"](x))
                if x.size(0) > 1 or not self.training:
                    x = self.species_mlps[sp_id]["bn1"](x)
                x = self.species_mlps[sp_id]["dropout1"](x)
                # 第二层MLP + 残差连接
                x = F.relu(self.species_mlps[sp_id]["linear2"](x) + shortcut)  # 残差相加
                if x.size(0) > 1 or not self.training:
                    x = self.species_mlps[sp_id]["bn2"](x)
                x = self.species_mlps[sp_id]["dropout2"](x)
                processed_feat[sp_mask] = x
        # 新增：所有物种的特征过共享MLP，实现特征共享
        processed_feat = self.shared_mlp(processed_feat)
        return processed_feat * self.branch_weight

class ESM2_Subcell_Network_MLP(nn.Module):
    """三分支融合主模型：ESM2+亚细胞+【物种专属MLP处理的网络特征】"""
    def __init__(self, num_species, esm_input_dim=2560, subcell_input_dim=14, network_input_dim=32, 
                 hidden_dim=512, num_go_terms=4000, dropout_rate=0.3):
        super().__init__()
        # ESM2分支：3层MLP（保留原checkpoint显存优化）
        self.esm_mlp1 = MLPBlock(esm_input_dim, hidden_dim, dropout_rate)
        self.esm_mlp2 = MLPBlock(hidden_dim, hidden_dim, dropout_rate)
        self.esm_mlp3 = MLPBlock(hidden_dim, hidden_dim, dropout_rate)  
        
        # 亚细胞分支（无修改）
        self.subcell_branch = SubcellBranch(subcell_input_dim, hidden_dim=32, branch_weight=0.9)
        # 网络分支：修复后的物种专属MLP分支
        self.network_branch = SpeciesSpecificNetworkBranch(
            num_species=num_species, input_dim=32, hidden_dim=32, branch_weight=0.8
        )
        
        # 三分支注意力机制（无修改，输入维度仍为512+32+32）
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # 输出层（无修改）
        self.output_layer = nn.Linear(hidden_dim + 32 + 32, num_go_terms)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0.0)
        
    def forward(self, esm_feats, subcell_feats, network_feats, species_ids):
        """
        前向传播：新增species_ids入参，传给网络分支做专属处理
        :return: logits - GO标签预测值 (B, num_go_terms)
        """
        # checkpoint兼容PyTorch2.0+
        checkpoint_kwargs = {"use_reentrant": False} if torch.__version__ >= "2.0.0" else {}
        
        # ESM2分支（保留checkpoint）
        esm_h = checkpoint(self.esm_mlp1, esm_feats,** checkpoint_kwargs)
        esm_h = checkpoint(self.esm_mlp2, esm_h, **checkpoint_kwargs)
        esm_h = checkpoint(self.esm_mlp3, esm_h,** checkpoint_kwargs)  
        
        # 亚细胞分支（保留checkpoint）
        subcell_h = checkpoint(self.subcell_branch, subcell_feats, **checkpoint_kwargs)
        # 网络分支：传入species_ids，做专属MLP处理
        network_h = checkpoint(self.network_branch, network_feats, species_ids,** checkpoint_kwargs)
        
        # 三分支注意力加权（无修改）
        concat_h = torch.cat([esm_h, subcell_h, network_h], dim=1)
        attn_weights = self.attention(concat_h)
        esm_weight = attn_weights[:, 0:1]
        subcell_weight = attn_weights[:, 1:2]
        network_weight = attn_weights[:, 2:3]
        
        # 特征加权+融合
        esm_h_weighted = esm_h * esm_weight
        subcell_h_weighted = subcell_h * subcell_weight
        network_h_weighted = network_h * network_weight
        fused_h = torch.cat([esm_h_weighted, subcell_h_weighted, network_h_weighted], dim=1)
        
        # 输出GO预测logits
        return self.output_layer(fused_h)

# ===================== 损失函数+硬负例挖掘（完全保留原逻辑） =====================
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
        for batch_idx, (esm_feat, subcell_feat, network_feat, species_ids, labels) in enumerate(train_loader):
            esm_feat = esm_feat.to(device, non_blocking=True)
            subcell_feat = subcell_feat.to(device, non_blocking=True)
            network_feat = network_feat.to(device, non_blocking=True)
            species_ids = species_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # 前向传播适配新增的species_ids入参
            outputs = model(esm_feat, subcell_feat, network_feat, species_ids)
            preds = torch.sigmoid(outputs) > 0.45
            rare_class_mask = train_class_counts <= 5
            true_rare = labels[:, rare_class_mask] == 1
            pred_rare_wrong = preds[:, rare_class_mask] == 0
            has_rare_wrong = torch.any(true_rare & pred_rare_wrong, dim=1)
            start_idx = batch_idx * train_loader.batch_size
            end_idx = start_idx + esm_feat.size(0)
            hard_neg_mask[start_idx:end_idx] = has_rare_wrong
    sample_weights[hard_neg_mask] *= 1.8
    return sample_weights

# ===================== 指标计算+可视化（完全保留原逻辑） =====================
def prepare_labels(all_go_terms_list, mlb=None):
    if mlb is None:
        mlb = MultiLabelBinarizer()
        labels_matrix = mlb.fit_transform(all_go_terms_list)
    else:
        labels_matrix = mlb.transform(all_go_terms_list)
    labels_tensor = torch.tensor(labels_matrix, dtype=torch.float32)
    print(f"✅ 标签矩阵: 形状{labels_tensor.shape} | GO总数{len(mlb.classes_)} | 非零标签{torch.sum(labels_tensor)}")
    return labels_tensor, mlb

def calculate_multilabel_metrics(logits, labels, class_freq=None, threshold=0.5):
    num_samples = logits.shape[0]
    if num_samples == 0:
        raise ValueError("logits样本数为0，无法计算指标")
    probs_np = torch.sigmoid(logits).detach().cpu().numpy()
    preds_np = (probs_np >= threshold).astype(int)
    labels_np = labels.cpu().numpy()

    f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
    f1_micro = f1_score(labels_np, preds_np, average='micro', zero_division=0)
    precision = precision_score(labels_np, preds_np, average='micro', zero_division=0)
    recall = recall_score(labels_np, preds_np, average='micro', zero_division=0)
    positive_coverage = np.mean(np.any(preds_np, axis=1))
    
    freq_based_metrics = {}
    if class_freq is not None:
        extreme_rare_mask = class_freq <= 2
        general_rare_mask = (class_freq > 2) & (class_freq <= 5)
        medium_mask = (class_freq > 5) & (class_freq <= 100)
        common_mask = class_freq > 100
        
        for name, mask in [("Extremely rare", extreme_rare_mask), ("Generally rare", general_rare_mask), 
                         ("Medium", medium_mask), ("Common", common_mask)]:
            if np.sum(mask) == 0:
                freq_based_metrics[name] = {"f1": 0.0, "count": 0}
                continue
            group_preds = preds_np[:, mask]
            group_labels = labels_np[:, mask]
            group_f1 = f1_score(group_labels, group_preds, average='micro', zero_division=0) if np.sum(group_labels) > 0 else 0.0
            freq_based_metrics[name] = {"f1": group_f1, "count": np.sum(mask)}

    del probs_np, preds_np, labels_np
    gc.collect()
    return {
        "f1_macro": round(f1_macro, 6), "f1_micro": round(f1_micro, 6),
        "precision": round(precision, 6), "recall": round(recall, 6),
        "positive_coverage": round(positive_coverage, 4), "threshold": threshold,
        "freq_groups": freq_based_metrics if class_freq is not None else {}
    }

def analyze_protein_predictions(model, valid_ids, valid_esm, valid_subcell, valid_network, valid_species_ids, valid_labels, mlb, epoch, save_dir, num_samples=5):
    model.eval()
    sample_indices = random.sample(range(len(valid_ids)), num_samples)
    sample_ids = [valid_ids[i] for i in sample_indices]
    sample_esm = valid_esm[sample_indices].to(device)
    sample_subcell = valid_subcell[sample_indices].to(device)
    sample_network = valid_network[sample_indices].to(device)
    sample_species_ids = valid_species_ids[sample_indices].to(device)
    sample_labels = valid_labels[sample_indices]
    
    with torch.no_grad():
        sample_outputs = model(sample_esm, sample_subcell, sample_network, sample_species_ids)
        sample_probs = torch.sigmoid(sample_outputs)
        sample_preds = (sample_probs >= 0.5).cpu()
    
    go_classes = mlb.classes_
    results = []
    print(f"\n📊 Epoch {epoch} 蛋白质预测详情：")
    print("-" * 100)
    for i, (protein_id, true_labels, pred_labels, probs) in enumerate(zip(sample_ids, sample_labels, sample_preds, sample_probs.cpu())):
        true_go_indices = torch.where(true_labels == 1)[0].tolist()
        true_go_terms = [go_classes[idx] for idx in true_go_indices]
        pred_go_indices = torch.where(pred_labels == 1)[0].tolist()
        pred_go_terms = [(go_classes[idx], float(probs[idx])) for idx in pred_go_indices]
        tp = set(true_go_indices) & set(pred_go_indices)
        fp = set(pred_go_indices) - set(true_go_indices)
        fn = set(true_go_indices) - set(pred_go_indices)
        print(f"\nProtein {i+1}: {protein_id}")
        print(f"  真实GO: {', '.join(true_go_terms) if true_go_terms else '无'}")
        print(f"  预测GO: {', '.join([f'{go} ({prob:.3f})' for go, prob in pred_go_terms]) if pred_go_terms else '无'}")
        if tp: print(f"  ✅ 正确: {', '.join([go_classes[idx] for idx in tp])}")
        if fp: print(f"  ⚠️  假阳: {', '.join([go_classes[idx] for idx in fp])}")
        if fn: print(f"  ❌ 假阴: {', '.join([go_classes[idx] for idx in fn])}")
        results.append({
            'epoch': epoch, 'protein_id': protein_id, 'true_go_terms': true_go_terms,
            'pred_go_terms': pred_go_terms, 'tp': len(tp), 'fp': len(fp), 'fn': len(fn),
            'acc': len(tp)/len(true_go_terms) if true_go_terms else 1.0
        })
    print("-" * 100)
    results_df = pd.DataFrame(results)
    log_path = os.path.join(save_dir, 'protein_pred_log.csv')
    if os.path.exists(log_path):
        results_df = pd.concat([pd.read_csv(log_path), results_df], ignore_index=True)
    results_df.to_csv(log_path, index=False)
    return results

def plot_loss_trend(train_loss, valid_loss, save_dir):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, '#2E86AB', linewidth=2.5, marker='o', label='Train Loss')
    plt.plot(epochs, valid_loss, '#F18F01', linewidth=2.5, marker='s', label='Val Loss')
    plt.title('Training & Validation Loss Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 损失趋势图保存至: {save_dir}")

def plot_metric_trend(train_metrics, valid_metrics, metric_name, save_dir):
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, '#2E86AB', linewidth=2.5, marker='o', label=f'Train {metric_name}')
    plt.plot(epochs, valid_metrics, '#F18F01', linewidth=2.5, marker='s', label=f'Val {metric_name}')
    plt.title(f'{metric_name} Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric_name.lower()}_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {metric_name}趋势图保存至: {save_dir}")

def plot_frequency_based_metrics(history, save_dir):
    epochs = range(1, len(history) + 1)
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, [h['freq_groups']['Extremely rare']['f1'] for h in history], '#C73E1D', label='Extremely rare (≤2)')
    plt.plot(epochs, [h['freq_groups']['Generally rare']['f1'] for h in history], '#E9724C', label='Generally rare (3-5)')
    plt.plot(epochs, [h['freq_groups']['Medium']['f1'] for h in history], '#F18F01', label='Medium (6-100)')
    plt.plot(epochs, [h['freq_groups']['Common']['f1'] for h in history], '#2E86AB', label='Common (>100)')
    plt.title('F1 Score Across Frequency Groups', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Micro F1', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'freq_based_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 频率分组F1图保存至: {save_dir}")

# ===================== 数据加载主函数 =====================
def load_dataset(data_dir, set_type):
    """加载train/val/test集，无CORAL对齐，返回物种标签（无空值）"""
    print(f"\n===== 加载{set_type.upper()}集 =====")
    # 定义文件路径（按你的命名规则）
    go_csv = os.path.join(data_dir, f"2_go_terms_matched_{set_type}.csv")
    network_csv = os.path.join(data_dir, f"1_line_feat_matched_{set_type}.csv")
    subcell_csv = os.path.join(data_dir, f"3_subcell_feat_matched_{set_type}.csv")
    esm2_csv = os.path.join(data_dir, f"4_esm2_feat_matched_{set_type}.csv")
    
    # 加载GO标签（作为参考ID）
    go_df = load_csv_file(go_csv)
    ref_ids = extract_source_ids(go_df)
    go_terms_list = extract_go_labels(go_df)
    
    # 加载并对齐各特征（均按GO的ref_ids排序）
    network_df = load_csv_file(network_csv)
    network_feat, network_species = align_feat_by_id(ref_ids, network_df, extract_network_feat)  # 无空物种
    
    subcell_df = load_csv_file(subcell_csv)
    subcell_feat = align_feat_by_id(ref_ids, subcell_df, extract_subcell_feat)
    
    esm2_df = load_csv_file(esm2_csv)
    esm2_feat = align_feat_by_id(ref_ids, esm2_df, extract_esm2_feat)
    
    # 验证样本数一致
    assert len(ref_ids) == esm2_feat.shape[0] == subcell_feat.shape[0] == network_feat.shape[0], \
        f"{set_type}集特征样本数不匹配"
    print(f"✅ {set_type.upper()}集加载完成: 样本数{len(ref_ids)} | 三特征维度匹配")
    
    return ref_ids, esm2_feat, subcell_feat, network_feat, go_terms_list, network_species

# ===================== 模型训练主函数 =====================
def train_model():
    # ========== 请替换为你的实际数据集根目录 ==========
    BASE_DIR = "/misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/数据"
    SAVE_DIR = os.path.join(BASE_DIR, "model_results_species_mlp")
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "go_best_model_species_mlp.pt")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📂 结果保存至: {SAVE_DIR}")

    # 1. 加载所有数据集（train/val/test均返回物种标签，无空值）
    train_ids, train_esm2, train_subcell, train_network, train_go, train_species = load_dataset(BASE_DIR, "train")
    val_ids, val_esm2, val_subcell, val_network, val_go, val_species = load_dataset(BASE_DIR, "val")
    test_ids, test_esm2, test_subcell, test_network, test_go, test_species = load_dataset(BASE_DIR, "test")

    # 2. 物种编码（核心：无空物种，所有物种从0开始连续编码，训练/val/test用同一映射）
    all_unique_species = sorted(list(set(train_species)))
    num_species = len(all_unique_species)  # 实际物种数（5）
    species2id = {sp: idx for idx, sp in enumerate(all_unique_species)}
    print(f"\n🔧 物种编码完成 | 总物种数: {num_species} | 映射: {species2id}")
    # 转换所有集的物种为数值ID（LongTensor）
    train_species_ids = torch.tensor([species2id[sp] for sp in train_species], dtype=torch.long)
    val_species_ids = torch.tensor([species2id[sp] for sp in val_species], dtype=torch.long)
    test_species_ids = torch.tensor([species2id[sp] for sp in test_species], dtype=torch.long)

    # 3. 准备GO标签矩阵（用训练集MLB统一编码）
    train_labels, mlb = prepare_labels(train_go)
    val_labels, _ = prepare_labels(val_go, mlb)
    test_labels, _ = prepare_labels(test_go, mlb)
    num_go_terms = len(mlb.classes_)
    train_class_counts = train_labels.sum(dim=0).cpu().numpy()  # 计算类别频率

    # 4. 转换所有特征为Tensor（float32）
    train_esm2 = torch.tensor(train_esm2, dtype=torch.float32)
    train_subcell = torch.tensor(train_subcell, dtype=torch.float32)
    train_network = torch.tensor(train_network, dtype=torch.float32)
    
    val_esm2 = torch.tensor(val_esm2, dtype=torch.float32)
    val_subcell = torch.tensor(val_subcell, dtype=torch.float32)
    val_network = torch.tensor(val_network, dtype=torch.float32)
    
    test_esm2 = torch.tensor(test_esm2, dtype=torch.float32)
    test_subcell = torch.tensor(test_subcell, dtype=torch.float32)
    test_network = torch.tensor(test_network, dtype=torch.float32)

    # 5. 创建Dataset（新增species_ids，四特征+标签）
    train_dataset = TensorDataset(train_esm2, train_subcell, train_network, train_species_ids, train_labels)
    val_dataset = TensorDataset(val_esm2, val_subcell, val_network, val_species_ids, val_labels)
    test_dataset = TensorDataset(test_esm2, test_subcell, test_network, test_species_ids, test_labels)

    # 6. 加权随机采样（解决类别不平衡，保留原逻辑，权重在CPU）
    sample_weights = torch.ones(train_labels.shape[0], dtype=torch.float32)
    extreme_rare_mask = train_class_counts <= 2
    general_rare_mask = (train_class_counts > 2) & (train_class_counts <= 5)
    has_extreme_rare = torch.any(train_labels[:, extreme_rare_mask], dim=1).cpu()
    has_general_rare = (torch.any(train_labels[:, general_rare_mask], dim=1) & ~has_extreme_rare).cpu()
    sample_weights[has_extreme_rare] *= 3.0
    sample_weights[has_general_rare] *= 2.5
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    # 7. 创建DataLoader
    batch_size = 64
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

    # 8. 初始化模型（传入实际物种数num_species）
    model = ESM2_Subcell_Network_MLP(
        num_species=num_species,  # 核心：传入5个物种，创建5个专属MLP
        esm_input_dim=2560, subcell_input_dim=14, network_input_dim=32,
        hidden_dim=512, num_go_terms=num_go_terms, dropout_rate=0.3
    ).to(device)
    print(f"\n✅ 模型初始化完成 | 物种专属MLP数: {num_species} | 输出GO数: {num_go_terms}")
    print("模型显存状态：")
    print_gpu_memory()

    # 9. 损失函数+优化器+学习率调度（保留原逻辑）
    criterion = WeightedFocalLoss(
        alpha=0.8, gamma=2.0, class_freq=train_class_counts, common_weight=1.2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6
    )

    # 10. 训练参数（早停）
    num_epochs = 60
    patience = 10
    best_valid_loss = float('inf')
    early_stop_counter = 0
    # 训练记录
    train_loss_hist = []
    valid_loss_hist = []
    train_f1_hist = []
    valid_f1_hist = []
    valid_metrics_hist = []

    print(f"\n===== 开始训练（物种专属MLP+三分支融合） =====")
    print(f"📌 轮次: {num_epochs} | 批大小: {batch_size} | 早停: {patience} | 初始LR: 9e-4")
    print(f"📌 物种数: {num_species} | 罕见类加权: 极端×3.0 | 一般×2.5")

    # 11. 训练循环
    for epoch in range(num_epochs):
        # 每2轮更新硬负例权重（保留原逻辑）
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
        train_preds = []
        train_targets = []
        for batch_idx, (esm, subcell, network, sp_id, labels) in enumerate(train_loader):
            # 兜底：跳过单样本批次（概率极低，防止极端情况）
            if esm.size(0) == 1:
                continue
            # 数据移至GPU
            esm = esm.to(device, non_blocking=True)
            subcell = subcell.to(device, non_blocking=True)
            network = network.to(device, non_blocking=True)
            sp_id = sp_id.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 前向传播（传入sp_id，物种专属处理）
            outputs = model(esm, subcell, network, sp_id)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_running_loss += loss.item() * esm.size(0)
            # 记录预测
            if batch_idx % 10 == 0:
                train_preds.append(outputs.detach())
                train_targets.append(labels.detach())
            # 打印批次信息
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")
            # 显存清理
            if (batch_idx + 1) % 40 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # 计算训练集指标
        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        train_loss_hist.append(train_epoch_loss)
        if train_preds:
            train_all_preds = torch.cat(train_preds, dim=0)
            train_all_labels = torch.cat(train_targets, dim=0)
            train_metrics = calculate_multilabel_metrics(train_all_preds, train_all_labels, train_class_counts)
            train_f1_hist.append(train_metrics['f1_micro'])
            print(f"\n训练集 - Epoch {epoch+1}: 损失={train_epoch_loss:.6f} | F1-micro={train_metrics['f1_micro']:.6f}")
        else:
            train_f1_hist.append(0.0)
        # 清理显存
        del train_preds, train_targets
        if 'train_all_preds' in locals():
            del train_all_preds, train_all_labels
        gc.collect()
        torch.cuda.empty_cache()

        # 验证阶段
        model.eval()
        valid_running_loss = 0.0
        valid_all_preds = None
        valid_all_labels = None
        with torch.no_grad():
            for esm, subcell, network, sp_id, labels in val_loader:
                esm = esm.to(device, non_blocking=True)
                subcell = subcell.to(device, non_blocking=True)
                network = network.to(device, non_blocking=True)
                sp_id = sp_id.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(esm, subcell, network, sp_id)
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item() * esm.size(0)
                # 记录预测
                if valid_all_preds is None:
                    valid_all_preds = outputs
                    valid_all_labels = labels
                else:
                    valid_all_preds = torch.cat([valid_all_preds, outputs], dim=0)
                    valid_all_labels = torch.cat([valid_all_labels, labels], dim=0)
        # 计算验证集指标
        valid_epoch_loss = valid_running_loss / len(val_loader.dataset)
        valid_loss_hist.append(valid_epoch_loss)
        valid_metrics = calculate_multilabel_metrics(valid_all_preds, valid_all_labels, train_class_counts)
        valid_f1_hist.append(valid_metrics['f1_micro'])
        valid_metrics_hist.append(valid_metrics)
        # 打印验证结果
        print(f"验证集 - Epoch {epoch+1}: 损失={valid_epoch_loss:.6f} | F1-micro={valid_metrics['f1_micro']:.6f}")
        # 打印频率分组F1
        freq_cn = {"Extremely rare":"极端罕见", "Generally rare":"一般罕见", "Medium":"中等", "Common":"常见"}
        for eng, m in valid_metrics['freq_groups'].items():
            print(f"  {freq_cn[eng]}: F1={m['f1']:.6f} (共{m['count']}个)")
        # 分析预测详情（每5轮/最后5轮）
        if epoch % 5 == 0 or epoch >= num_epochs - 5:
            analyze_protein_predictions(
                model, val_ids, val_esm2, val_subcell, val_network, val_species_ids,
                val_labels, mlb, epoch+1, SAVE_DIR
            )

        # 学习率调度+早停+保存最佳模型
        scheduler.step(valid_epoch_loss)
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            torch.save({
                'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_valid_loss,
                'best_f1': valid_metrics['f1_micro'], 'mlb_classes': mlb.classes_,
                'species2id': species2id, 'num_species': num_species
            }, MODEL_SAVE_PATH)
            print(f"📈 保存最佳模型至: {MODEL_SAVE_PATH} (损失: {best_valid_loss:.6f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"⚠️  早停计数: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"🛑 早停触发！")
                break
        # 显存清理
        del valid_all_preds, valid_all_labels
        gc.collect()
        torch.cuda.empty_cache()
        print("当前显存状态：")
        print_gpu_memory()

    # 12. 测试集最终评估
    print(f"\n===== 测试集评估 =====")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_running_loss = 0.0
    test_all_preds = None
    test_all_labels = None
    with torch.no_grad():
        for esm, subcell, network, sp_id, labels in test_loader:
            esm = esm.to(device, non_blocking=True)
            subcell = subcell.to(device, non_blocking=True)
            network = network.to(device, non_blocking=True)
            sp_id = sp_id.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(esm, subcell, network, sp_id)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * esm.size(0)
            if test_all_preds is None:
                test_all_preds = outputs
                test_all_labels = labels
            else:
                test_all_preds = torch.cat([test_all_preds, outputs], dim=0)
                test_all_labels = torch.cat([test_all_labels, labels], dim=0)
    # 计算测试集指标
    test_epoch_loss = test_running_loss / len(test_loader.dataset)
    test_metrics = calculate_multilabel_metrics(test_all_preds, test_all_labels, train_class_counts)
    print(f"\n===== 测试集最终结果 =====")
    print(f"损失: {test_epoch_loss:.6f} | F1-micro: {test_metrics['f1_micro']:.6f}")
    print(f"精确率: {test_metrics['precision']:.6f} | 召回率: {test_metrics['recall']:.6f}")
    for eng, m in test_metrics['freq_groups'].items():
        print(f"  {freq_cn[eng]}: F1={m['f1']:.6f} (共{m['count']}个)")
    # 分析测试集预测详情
    analyze_protein_predictions(
        model, test_ids, test_esm2, test_subcell, test_network, test_species_ids,
        test_labels, mlb, "Final", SAVE_DIR
    )

    # 13. 保存训练记录+可视化
    metrics_history = {
        'train_loss': train_loss_hist, 'valid_loss': valid_loss_hist,
        'train_f1': train_f1_hist, 'valid_f1': valid_f1_hist,
        'valid_metrics': valid_metrics_hist, 'test_loss': test_epoch_loss,
        'test_metrics': test_metrics, 'species2id': species2id
    }
    torch.save(metrics_history, os.path.join(SAVE_DIR, 'training_metrics.pt'))
    # 保存MLB
    with open(os.path.join(SAVE_DIR, 'mlb_classes.pkl'), 'wb') as f:
        pickle.dump(mlb.classes_, f)
    # 生成可视化图
    plot_loss_trend(train_loss_hist, valid_loss_hist, SAVE_DIR)
    plot_metric_trend(train_f1_hist, valid_f1_hist, "F1 Score (Micro)", SAVE_DIR)
    plot_frequency_based_metrics(valid_metrics_hist, SAVE_DIR)

    print(f"\n===== 训练完成 =====")
    print(f"最佳验证损失: {best_valid_loss:.6f} | 最佳验证F1: {checkpoint['best_f1']:.6f}")
    print(f"最佳模型: {MODEL_SAVE_PATH} | 所有结果: {SAVE_DIR}")

if __name__ == "__main__":
    train_model()