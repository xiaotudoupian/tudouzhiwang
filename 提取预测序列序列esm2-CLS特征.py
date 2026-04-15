import torch
import esm
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import gc

# ========== 配置参数 ==========
FASTA_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/高粱/测试集100_蛋白序列.fasta"
OUTPUT_PATH = "/misc/hard_disk/others_res/zhangdy/wangluo/高粱/测试集100_蛋白序列.pt"
BATCH_SIZE = 2
MAX_SEQ_LEN = 2046
TARGET_LAYER = 36
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

# 清除GPU缓存
torch.cuda.empty_cache()
gc.collect()

print(f"✅ 设备初始化完成：使用GPU（{DEVICE}），设备名称：{torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else 'CPU'}")
print(f"⚠️  内存优化配置：批次大小={BATCH_SIZE}，最大序列长度={MAX_SEQ_LEN}")


class FastaDataset(Dataset):
    """FASTA数据集加载器（保留原始蛋白质ID）"""
    def __init__(self, fasta_path, max_seq_len=2046):
        self.fasta_path = fasta_path
        self.max_seq_len = max_seq_len
        self.protein_ids = []
        self.sequences = []
        self._parse_fasta()

    def _parse_fasta(self):
        """解析FASTA文件，提取ID和序列"""
        current_id = None
        current_seq = []
        
        with open(self.fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq:
                        self.protein_ids.append(current_id)
                        full_seq = "".join(current_seq)[:self.max_seq_len]
                        self.sequences.append(full_seq)
                    # 提取原始蛋白质ID（>后面的第一个字段）
                    current_id = line[1:].split()[0]
                    current_seq = []
                elif line and current_id:
                    current_seq.append(line)
        
        # 添加最后一个序列
        if current_id and current_seq:
            self.protein_ids.append(current_id)
            full_seq = "".join(current_seq)[:self.max_seq_len]
            self.sequences.append(full_seq)
        
        print(f"✅ FASTA文件解析完成：共{len(self.protein_ids)}条序列")
        if self.protein_ids:
            print(f"✅ 蛋白质ID列表：{self.protein_ids}")
        else:
            print("⚠️ 未解析到任何序列！")

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        return {
            "id": self.protein_ids[idx],
            "sequence": self.sequences[idx]
        }


def extract_esm2_features():
    """使用官方ESM-2模型提取CLS特征（修复数据类型错误）"""
    # 直接加载官方模型
    print(f"\n🔄 加载ESM-2模型（esm2_t36_3B_UR50D）...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    
    # 模型配置（不使用half()，token必须是long类型）
    model = model.to(DEVICE)
    model.eval()
    
    batch_converter = alphabet.get_batch_converter()

    # 加载数据集
    dataset = FastaDataset(FASTA_PATH, MAX_SEQ_LEN)
    if len(dataset) == 0:
        print("❌ 数据集为空，终止提取！")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    # 存储结果
    all_protein_ids = []
    all_cls_features = []

    print(f"\n🚀 开始提取第{TARGET_LAYER}层CLS特征，共{len(dataloader)}批次...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="提取进度")):
            batch_ids = batch["id"]
            batch_seqs = batch["sequence"]
            
            # 转换为模型输入格式
            batch_data = list(zip(batch_ids, batch_seqs))
            _, _, batch_tokens = batch_converter(batch_data)
            
            # 确保tokens是long类型（关键修复！）
            batch_tokens = batch_tokens.to(DEVICE, non_blocking=True, dtype=torch.long)
            
            try:
                # 模型推理
                outputs = model(
                    batch_tokens,
                    repr_layers=[TARGET_LAYER],
                    return_contacts=False
                )

                # 提取CLS特征
                cls_features = outputs["representations"][TARGET_LAYER][:, 0, :].cpu()
                
                # 保存结果
                all_protein_ids.extend(batch_ids)
                all_cls_features.append(cls_features)
                
            except RuntimeError as e:
                print(f"\n⚠️  处理批次{batch_idx}时出错：{str(e)}")
                print(f"   尝试使用CPU提取...")
                
                # 降级到CPU
                batch_tokens = batch_tokens.cpu()
                model = model.cpu()
                
                outputs = model(
                    batch_tokens,
                    repr_layers=[TARGET_LAYER],
                    return_contacts=False
                )
                
                cls_features = outputs["representations"][TARGET_LAYER][:, 0, :]
                all_protein_ids.extend(batch_ids)
                all_cls_features.append(cls_features)
                
                # 恢复到GPU
                model = model.to(DEVICE)
            
            # 清理内存
            del batch_tokens, outputs
            torch.cuda.empty_cache()
    
    # 合并特征张量
    if all_cls_features:
        all_cls_features = torch.cat(all_cls_features, dim=0)
    else:
        print("❌ 未提取到任何特征！")
        return
    
    # 保存结果
    torch.save({
        "protein_ids": all_protein_ids,
        "cls_features": all_cls_features,
        "model_name": "esm2_t36_3B_UR50D",
        "target_layer": TARGET_LAYER,
        "max_sequence_length": MAX_SEQ_LEN
    }, OUTPUT_PATH)

    # 验证结果
    print(f"\n🎉 特征提取完成！")
    print(f"📊 结果统计：")
    print(f"  - 总序列数：{len(all_protein_ids)}")
    print(f"  - 特征形状：{all_cls_features.shape}")
    print(f"  - 特征维度：{all_cls_features.shape[1]}")
    print(f"  - 保存路径：{OUTPUT_PATH}")
    print(f"  - ID与特征数量匹配：{len(all_protein_ids) == all_cls_features.shape[0]}")
    
    # 特征统计
    print(f"\n📈 特征统计信息：")
    print(f"  - 特征均值：{torch.mean(all_cls_features):.6f}")
    print(f"  - 特征标准差：{torch.std(all_cls_features):.6f}")
    print(f"  - 特征最小值：{torch.min(all_cls_features):.6f}")
    print(f"  - 特征最大值：{torch.max(all_cls_features):.6f}")
    
    # 最终清理
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # 检查文件
    if not os.path.exists(FASTA_PATH):
        raise FileNotFoundError(f"❌ FASTA文件不存在：{FASTA_PATH}")
    
    # 创建输出目录
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 开始提取
    extract_esm2_features()
    
    # 加载验证
    if os.path.exists(OUTPUT_PATH):
        saved_data = torch.load(OUTPUT_PATH)
        print(f"\n✅ 验证输出文件：")
        print(f"  - 蛋白质ID数量：{len(saved_data['protein_ids'])}")
        print(f"  - 特征张量形状：{saved_data['cls_features'].shape}")
        if saved_data['protein_ids']:
            print(f"  - 蛋白质ID列表：{saved_data['protein_ids']}")
        print(f"  - 模型信息匹配：{saved_data['model_name'] == 'esm2_t36_3B_UR50D'}")
    else:
        print("❌ 输出文件未生成！")