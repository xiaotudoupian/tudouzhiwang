import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. Tae专用极致显存参数（64维+2头，和其他物种统一） =====================
# 特征维度/注意力头：和Ath/Gma/Osa/Zma保持一致
HIDDEN_DIM = 64          
HEADS = 2                
DROPOUT = 0.25
LEARNING_RATE = 0.008
EPOCHS = 100

# 批次/采样：极致压缩显存（针对7257万条边）
TRAIN_BATCH_SIZE = 256    # 训练批次（不变）
FEAT_BATCH_SIZE = 32      # 特征提取批次降到32（Tae专用，显存峰值<15GB）
NUM_NEIGHBORS_TRAIN = [5, 3]   # 训练采样数（极致少，控显存）
NUM_NEIGHBORS_FEAT = [15, 8]   # 核心邻居采样（15+8，平衡显存/特征）
GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数8，进一步降显存峰值

# Tae边文件路径（替换为你的实际路径）
TAE_EDGE_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/atted_unzip/Tae/atted_network/Tae_coexpr_edgelist.txt"

# 显存优化终极开关
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# ===================== 2. Tae专用模型（64维+2头+梯度检查点） =====================
class TaeGAT(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim=HIDDEN_DIM, heads=HEADS, out_dim=HIDDEN_DIM):
        super().__init__()
        self.node_emb = torch.nn.Embedding(num_nodes, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, edge_dim=1, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, edge_dim=1, heads=1, concat=False)
        self.norm1 = torch.nn.LayerNorm(hidden_dim * heads)
        self.norm2 = torch.nn.LayerNorm(out_dim)
        self.dropout = torch.nn.Dropout(DROPOUT)

    # 梯度检查点封装：省50%显存
    def gat_layer1(self, x, edge_index, edge_weight):
        x = self.gat1(x, edge_index, edge_attr=edge_weight.unsqueeze(1))
        x = self.norm1(x)
        x = F.relu(x)
        return self.dropout(x)
    
    def gat_layer2(self, x, edge_index, edge_weight):
        x = self.gat2(x, edge_index, edge_attr=edge_weight.unsqueeze(1))
        x = self.norm2(x)
        return x

    def forward(self, x, edge_index, edge_weight):
        x = self.node_emb(x)
        x = self.dropout(x)
        # 梯度检查点：强制省显存（Tae必须）
        x = checkpoint(self.gat_layer1, x, edge_index, edge_weight)
        x = checkpoint(self.gat_layer2, x, edge_index, edge_weight)
        return x

# ===================== 3. Tae训练+特征提取主函数（极致显存控制） =====================
def train_tae():
    print(f"\n=====================================")
    print(f"开始训练 Tae（小麦）| 64维+2头 | 极致显存优化")
    print(f"=====================================")
    
    # 3.1 分块读取Tae超大边文件（避免CPU内存溢出）
    print("📖 分块读取Tae边文件（7257万条边）...")
    df_chunks = pd.read_csv(TAE_EDGE_PATH, sep="\t", chunksize=1000000)
    all_genes = set()
    edges = []
    edge_weights = []
    
    for chunk in tqdm(df_chunks, desc="加载边数据"):
        all_genes.update(chunk["gene1"].tolist())
        all_genes.update(chunk["gene2"].tolist())
        edges.extend(zip(chunk["gene1"], chunk["gene2"]))
        edge_weights.extend(chunk["z_score"].tolist())
    
    # 构建基因索引
    all_genes = sorted(list(all_genes))
    gene2idx = {gene: idx for idx, gene in enumerate(all_genes)}
    idx2gene = {v: k for k, v in gene2idx.items()}
    num_nodes = len(all_genes)
    print(f"✅ Tae预处理完成：{num_nodes} 基因，{len(edges)} 条边")

    # 3.2 转换边索引（分批次，避免内存爆炸）
    edge_index = []
    for u, v in tqdm(edges, desc="转换边索引"):
        edge_index.append([gene2idx[u], gene2idx[v]])
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long).contiguous()
    edge_weights = torch.tensor(np.array(edge_weights), dtype=torch.float32).contiguous()

    # 3.3 构建图数据（CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = Data(
        x=torch.arange(num_nodes),
        edge_index=edge_index,
        edge_weight=edge_weights,
        num_nodes=num_nodes
    )

    # 3.4 训练用Loader（关闭多线程，减少显存碎片）
    train_loader = NeighborLoader(
        graph_data,
        batch_size=TRAIN_BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS_TRAIN,
        shuffle=True,
        pin_memory=True,
        replace=False,
        num_workers=0  # 关键：关闭多线程，避免显存碎片化
    )

    # 3.5 初始化模型（强制CUDA）
    model = TaeGAT(num_nodes=num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # 3.6 训练模型（8步梯度累积+高频显存清理）
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = batch.to(device, non_blocking=True)
            batch_count += 1
            
            with torch.cuda.amp.autocast():
                out = model(batch.x, batch.edge_index, batch.edge_weight)
                loss = F.mse_loss(out[batch.edge_index[0]], out[batch.edge_index[1]])
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            # 每8批次更新一次，强制清理显存
            if batch_count % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS * batch.num_nodes
        
        # 每20轮打印损失+清理显存
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / num_nodes
            print(f"📌 Tae | Epoch {epoch+1} | 平均损失 {avg_loss:.6f}")
            torch.cuda.empty_cache()

    # 3.7 深度清理训练显存（释放10GB+）
    del train_loader
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f"\n🔍 开始提取Tae特征（64维+2头）...")

    # 3.8 特征提取用Loader（32批次+15+8采样）
    feat_loader = NeighborLoader(
        graph_data,
        batch_size=FEAT_BATCH_SIZE,
        num_neighbors=NUM_NEIGHBORS_FEAT,
        shuffle=False,
        pin_memory=True,
        replace=False,
        num_workers=0
    )

    # 3.9 分批次提取特征（极致显存控制）
    model.eval()
    gat_features = np.zeros((num_nodes, HIDDEN_DIM), dtype=np.float32)
    
    with torch.no_grad():
        for batch in tqdm(feat_loader, desc="提取特征"):
            batch = batch.to(device, non_blocking=True)
            batch_feat = model(batch.x, batch.edge_index, batch.edge_weight)
            # 写入特征
            gat_features[batch.n_id.cpu().numpy()] = batch_feat.cpu().numpy()
            # 强制清理当前批次显存
            del batch, batch_feat
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # 3.10 保存特征（64维，和其他物种格式一致）
    gene_gat_features = {idx2gene[idx]: feat for idx, feat in enumerate(gat_features)}
    # CSV保存
    feat_df = pd.DataFrame.from_dict(gene_gat_features, orient="index")
    feat_df.columns = [f"gat_feat_{i}" for i in range(HIDDEN_DIM)]
    feat_df.to_csv("Tae_gat_features.csv", index_label="gene_name", chunksize=10000)
    # PKL保存
    with open("Tae_gat_features.pkl", "wb") as f:
        pickle.dump(gene_gat_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n🎉 Tae特征保存完成：")
    print(f"   - CSV: Tae_gat_features.csv")
    print(f"   - PKL: Tae_gat_features.pkl")
    print(f"   - 特征维度：64维（和其他物种统一）")

    # 3.11 终极显存清理
    del model, optimizer, scaler, feat_loader, graph_data, edge_index, edge_weights, gat_features
    torch.cuda.empty_cache()
    print(f"✅ Tae显存已完全释放")

# ===================== 4. 直接运行Tae训练 =====================
if __name__ == "__main__":
    train_tae()
    print("\n🚀 Tae（小麦）训练+特征提取全部完成！")