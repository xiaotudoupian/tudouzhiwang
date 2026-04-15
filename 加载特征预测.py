import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

# ========== 核心配置（固定阈值 + 路径） ==========
# 固定阈值（多标签分类标准默认值，模型表现最优）
FIXED_THRESHOLD = 0.3  

# 路径配置
MODEL_PATH = "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits/BP/model_training_results_loss_based/best_model_loss_based.pt"
TEST_FEATURES_PATH = "/misc/hard_disk/others_res/zhangdy/yuce/esm2_t36_cls_features_1fasta.pt"
MLB_CLASSES_PATH = "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits/BP/model_training_results_loss_based/mlb_classes.pkl"
OUTPUT_DIR = "/misc/hard_disk/others_res/zhangdy/GO_subontologies/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")
print(f"使用固定阈值: {FIXED_THRESHOLD}")

# ========== 模型定义（与训练代码保持一致） ==========
class MLPBlock(nn.Module):
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

class ESM2_MLP(nn.Module):
    def __init__(self, esm_input_dim=2560, hidden_dim=512, num_go_terms=3143, dropout_rate=0.3):
        super().__init__()
        self.esm_mlp1 = MLPBlock(esm_input_dim, hidden_dim, dropout_rate)
        self.esm_mlp2 = MLPBlock(hidden_dim, hidden_dim, dropout_rate)
        self.esm_mlp3 = MLPBlock(hidden_dim, hidden_dim, dropout_rate)  
        self.output_layer = nn.Linear(hidden_dim, num_go_terms)
        
    def forward(self, esm_feats):
        esm_h = self.esm_mlp1(esm_feats)
        esm_h = self.esm_mlp2(esm_h)
        esm_h = self.esm_mlp3(esm_h)  
        return self.output_layer(esm_h)

# ========== 核心函数 ==========
def load_model_and_mlb():
    """加载训练好的模型和MLB类别"""
    print("\n🔄 加载模型和配置...")
    
    # 加载模型检查点
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # 获取模型参数
    esm_input_dim = checkpoint.get('esm_input_dim', 2560)
    num_go_terms = checkpoint.get('num_go_terms', 3143)
    
    # 加载MLB类别
    if 'mlb_classes' in checkpoint:
        mlb_classes = checkpoint['mlb_classes']
    else:
        with open(MLB_CLASSES_PATH, 'rb') as f:
            mlb_classes = pickle.load(f)
    
    # 转换为列表（确保兼容性）
    if isinstance(mlb_classes, np.ndarray):
        mlb_classes = mlb_classes.tolist()
    
    # 创建模型并加载权重
    model = ESM2_MLP(
        esm_input_dim=esm_input_dim,
        hidden_dim=512,
        num_go_terms=num_go_terms
    ).to(DEVICE)
    
    # 兼容不同的权重保存方式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✅ 模型加载完成：")
    print(f"  - ESM输入维度: {esm_input_dim}")
    print(f"  - GO术语总数: {len(mlb_classes)}")
    
    return model, mlb_classes

def load_test_features():
    """加载测试集特征（自动适配不同格式）"""
    print("\n🔄 加载测试集特征...")
    # 加载特征文件
    data = torch.load(TEST_FEATURES_PATH, map_location=DEVICE)
    
    # ========== 第一步：检查并打印文件格式 ==========
    print("📌 特征文件格式检测结果：")
    if isinstance(data, dict):
        print(f"  - 文件类型: 字典")
        print(f"  - 所有键名: {list(data.keys())}")
        # 打印每个键的详细信息
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"    - 键 '{k}': 张量 | 形状: {v.shape} | 数据类型: {v.dtype}")
            elif isinstance(v, list):
                print(f"    - 键 '{k}': 列表 | 长度: {len(v)}")
            elif isinstance(v, np.ndarray):
                print(f"    - 键 '{k}': 数组 | 形状: {v.shape}")
            else:
                print(f"    - 键 '{k}': {type(v)}")
    else:
        print(f"  - 文件类型: {type(data)}")
        if isinstance(data, torch.Tensor):
            print(f"  - 张量形状: {data.shape} | 数据类型: {data.dtype}")
    
    # ========== 第二步：自动适配加载特征和ID ==========
    protein_ids = None
    features = None
    
    # 情况1：字典格式（优先匹配常见键名）
    if isinstance(data, dict):
        # 优先匹配标准键名
        if 'protein_ids' in data and 'features' in data:
            protein_ids = data['protein_ids']
            features = data['features']
        # 匹配ESM特征常见键名
        elif 'ids' in data and 'embeddings' in data:
            protein_ids = data['ids']
            features = data['embeddings']
        elif 'names' in data and 'feats' in data:
            protein_ids = data['names']
            features = data['feats']
        # 兜底：自动识别ID和特征
        else:
            # 找ID（列表/数组类型）
            id_candidates = [k for k in data.keys() if isinstance(data[k], (list, np.ndarray))]
            # 找特征（张量类型）
            feat_candidates = [k for k in data.keys() if isinstance(data[k], torch.Tensor)]
            
            if id_candidates and feat_candidates:
                protein_ids = data[id_candidates[0]]
                features = data[feat_candidates[0]]
                print(f"⚠️  未找到标准键名，自动匹配：")
                print(f"    - ID键: {id_candidates[0]}")
                print(f"    - 特征键: {feat_candidates[0]}")
            else:
                raise ValueError("❌ 无法从字典中识别ID和特征键，请检查文件内容")
    
    # 情况2：纯张量（无ID）
    elif isinstance(data, torch.Tensor):
        features = data
        # 生成默认ID
        protein_ids = [f"protein_{i:06d}" for i in range(len(features))]
        print(f"⚠️  文件为纯张量，自动生成蛋白质ID（protein_000000 ~ protein_{len(features)-1:06d}）")
    
    else:
        raise ValueError(f"❌ 不支持的文件类型: {type(data)}")
    
    # ========== 第三步：特征预处理 ==========
    # 转换为张量（确保类型）
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    # 调整数据类型和设备
    features = features.to(dtype=torch.float32).to(DEVICE)
    
    # 处理3维特征（ESM序列特征，取CLS token）
    if len(features.shape) == 3:
        print(f"⚠️  检测到3维特征({features.shape})，自动取第0位(CLS token)作为特征")
        features = features[:, 0, :]  # [N, L, D] → [N, D]
    
    # 处理ID格式（确保是列表）
    if isinstance(protein_ids, np.ndarray):
        protein_ids = protein_ids.tolist()
    # 确保ID长度匹配
    if len(protein_ids) != len(features):
        print(f"⚠️  ID数量({len(protein_ids)})与特征数量({len(features)})不匹配，重新生成ID")
        protein_ids = [f"protein_{i:06d}" for i in range(len(features))]
    
    # ========== 第四步：打印加载结果 ==========
    print(f"✅ 特征加载完成：")
    print(f"  - 蛋白质数量: {len(protein_ids)}")
    print(f"  - 特征形状: {features.shape}")
    print(f"  - 特征维度: {features.shape[-1]} (ESM输入维度)")
    
    return protein_ids, features

def batch_predict(model, features, batch_size=256):
    """批量预测（避免显存溢出）"""
    print(f"\n🚀 开始批量预测（阈值: {FIXED_THRESHOLD}）...")
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        # 分批次处理
        for i in tqdm(range(0, len(features), batch_size), desc="预测进度"):
            batch_feats = features[i:i+batch_size]
            outputs = model(batch_feats)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
    
    # 合并所有批次结果
    all_probs = np.concatenate(all_probs, axis=0)
    print(f"✅ 预测完成，概率矩阵形状: {all_probs.shape}")
    
    return all_probs

def generate_human_readable_results(protein_ids, all_probs, mlb_classes):
    """生成人类易读的结果：蛋白质ID + 预测GO术语（概率）"""
    print(f"\n🔧 生成易读结果（阈值: {FIXED_THRESHOLD}）...")
    
    # 1. 生成二分类预测（基于固定阈值）
    predictions_binary = (all_probs >= FIXED_THRESHOLD).astype(int)
    
    # 2. 构建核心结果列表（易读格式）
    core_results = []
    simplified_results = []  # 极简格式：每个蛋白一行
    
    for i, protein_id in enumerate(tqdm(protein_ids, desc="处理结果")):
        prob_vector = all_probs[i]
        pred_vector = predictions_binary[i]
        
        # 获取该蛋白的阳性GO功能（基于阈值）
        positive_indices = np.where(pred_vector == 1)[0]
        
        # 按概率排序阳性功能
        positive_pairs = []
        for idx in positive_indices:
            go_term = mlb_classes[idx]
            prob = float(prob_vector[idx])
            positive_pairs.append((go_term, prob))
        
        # 按概率降序排序
        positive_pairs_sorted = sorted(positive_pairs, key=lambda x: x[1], reverse=True)
        
        # 格式1：详细行（每个GO术语一行）
        for go_term, prob in positive_pairs_sorted:
            core_results.append({
                "protein_id": protein_id,
                "go_term": go_term,
                "probability": round(prob, 4),
                "threshold": FIXED_THRESHOLD,
                "is_predicted": True
            })
        
        # 格式2：极简行（每个蛋白一行，GO术语用逗号分隔）
        if positive_pairs_sorted:
            go_terms_str = ", ".join([f"{go} ({prob:.4f})" for go, prob in positive_pairs_sorted])
            total_go = len(positive_pairs_sorted)
            max_prob_go = positive_pairs_sorted[0][0]
            max_prob_value = positive_pairs_sorted[0][1]
        else:
            go_terms_str = "无"
            total_go = 0
            max_prob_go = "无"
            max_prob_value = 0.0
        
        simplified_results.append({
            "protein_id": protein_id,
            "total_predicted_go": total_go,
            "max_prob_go_term": max_prob_go,
            "max_probability": round(max_prob_value, 4),
            "predicted_go_terms": go_terms_str,
            "threshold_used": FIXED_THRESHOLD
        })
    
    # 转换为DataFrame
    core_df = pd.DataFrame(core_results)
    simplified_df = pd.DataFrame(simplified_results)
    
    print(f"✅ 结果生成完成：")
    print(f"  - 阳性预测总数: {len(core_results)}")
    print(f"  - 有预测结果的蛋白质数: {len(simplified_df[simplified_df['total_predicted_go'] > 0])}")
    
    return core_df, simplified_df

def save_results(core_df, simplified_df):
    """保存结果文件（优先易读格式）"""
    print("\n💾 保存结果文件...")
    
    # 文件命名
    core_filename = f"BP_test_predictions_detailed_threshold_{FIXED_THRESHOLD}.csv"
    simplified_filename = f"BP_test_predictions_simplified_threshold_{FIXED_THRESHOLD}.csv"
    
    # 保存详细结果（每个GO术语一行）
    core_path = os.path.join(OUTPUT_DIR, core_filename)
    core_df.to_csv(core_path, index=False, encoding='utf-8')
    
    # 保存极简结果（每个蛋白一行）
    simplified_path = os.path.join(OUTPUT_DIR, simplified_filename)
    simplified_df.to_csv(simplified_path, index=False, encoding='utf-8')
    
    # 生成统计摘要
    stats = {
        "使用阈值": FIXED_THRESHOLD,
        "总预测蛋白质数": len(simplified_df),
        "有GO注释的蛋白质数": len(simplified_df[simplified_df['total_predicted_go'] > 0]),
        "有GO注释的蛋白质占比": f"{len(simplified_df[simplified_df['total_predicted_go'] > 0])/len(simplified_df)*100:.2f}%",
        "阳性GO术语总数": len(core_df),
        "平均每个蛋白的GO术语数": f"{core_df.groupby('protein_id').size().mean():.2f}",
        "最高单蛋白GO术语数": f"{core_df.groupby('protein_id').size().max()}",
        "平均预测概率": f"{core_df['probability'].mean():.4f}",
        "最高预测概率": f"{core_df['probability'].max():.4f}"
    }
    
    stats_df = pd.DataFrame(list(stats.items()), columns=["统计项", "数值"])
    stats_path = os.path.join(OUTPUT_DIR, f"BP_test_prediction_stats_threshold_{FIXED_THRESHOLD}.csv")
    stats_df.to_csv(stats_path, index=False, encoding='utf-8')
    
    print(f"✅ 所有文件已保存至: {OUTPUT_DIR}")
    print(f"  📄 详细结果: {core_filename} (蛋白质ID+GO术语+概率，每行一个GO)")
    print(f"  📄 极简结果: {simplified_filename} (蛋白质ID+所有GO术语，每行一个蛋白)")
    print(f"  📄 统计摘要: BP_test_prediction_stats_threshold_{FIXED_THRESHOLD}.csv")
    
    return core_path, simplified_path, stats_path

# ========== 主程序 ==========
def main():
    print("="*80)
    print("📊 BP测试集GO功能预测（固定阈值版）")
    print("="*80)
    
    # 1. 加载模型和配置
    model, mlb_classes = load_model_and_mlb()
    
    # 2. 加载测试特征（自动适配格式）
    protein_ids, features = load_test_features()
    
    # 3. 批量预测
    all_probs = batch_predict(model, features)
    
    # 4. 生成易读结果
    core_df, simplified_df = generate_human_readable_results(protein_ids, all_probs, mlb_classes)
    
    # 5. 保存结果
    core_path, simplified_path, stats_path = save_results(core_df, simplified_df)
    
    # 6. 打印最终摘要
    print("\n" + "="*80)
    print("🎉 预测完成！结果文件说明：")
    print("="*80)
    print("📌 优先查看【极简结果文件】(simplified)：")
    print("   - 每行一个蛋白质ID")
    print("   - 包含：预测的所有GO术语（带概率）、总数、最高概率GO")
    print("   - 格式：protein_id, total_predicted_go, max_prob_go_term, max_probability, predicted_go_terms")
    print("\n📌 详细结果文件（detailed）：")
    print("   - 每行一个GO术语预测")
    print("   - 格式：protein_id, go_term, probability, threshold, is_predicted")
    print("\n📌 统计摘要文件（stats）：")
    print("   - 整体预测情况统计，快速了解数据分布")
    print("="*80)
    print(f"💡 使用阈值: {FIXED_THRESHOLD} (多标签分类标准默认值)")
    print(f"💡 结果格式: 蛋白质ID + GO术语（概率值）")
    print(f"💡 所有GO术语已按概率降序排列，便于查看")
    print("="*80)

if __name__ == "__main__":
    # 设置CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    main()