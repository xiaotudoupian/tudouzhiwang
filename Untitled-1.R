# 加载包
library(igraph)
library(dplyr)
library(readr)

# ========== 适配Zma数据集：修改路径+保持核心优化逻辑 ==========
# Zma数据集路径（替换为你的目标路径）
data_dir <- "/misc/hard_disk/others_res/zhangdy/wangluo/atted_unzip/Zma/Zma-u.v22-04.G29820-S6091.combat_pca.subagging.z.d/"
# Zma数据集输出目录（自动创建，独立于其他数据集）
output_dir <- "/misc/hard_disk/others_res/zhangdy/wangluo/atted_unzip/Zma/atted_network/"
temp_dir <- paste0(output_dir, "temp_batches/")  # 临时批次文件目录
z_threshold <- 2  # 保持一致的阈值，如需调整可修改
batch_size <- 2000  # Zma文件数29820个，2000批次足够（共15批）

# 内存优化配置
options(mem.max_size = 16*1024^3)
options(stringsAsFactors = FALSE)
invisible(gc())

# 创建目录（自动创建多级目录，避免路径不存在报错）
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)

cat("===== 开始处理Zma数据集 =====\n")
cat("使用1核串行读取+过滤文件（仅保留有效数据）...\n")

# 获取Zma数据集的文件列表
file_list <- list.files(
  path = data_dir,
  pattern = "^[0-9]+$",  # 匹配数字命名的文件（和前两个数据集格式一致）
  full.names = TRUE
)

# 检查文件列表是否为空（防路径错误）
if (length(file_list) == 0) {
  stop("错误：未找到Zma数据集的数字命名文件！请检查data_dir路径是否正确。")
}
total_files <- length(file_list)
cat("找到Zma数据集文件数：", total_files, "个\n")

# ========== 核心逻辑：边读边过滤 + 分批次写磁盘 ==========
batch_num <- ceiling(total_files / batch_size)
temp_file_paths <- c()  # 记录所有临时文件路径

for (batch in 1:batch_num) {
  # 计算当前批次文件范围
  start_idx <- (batch - 1) * batch_size + 1
  end_idx <- min(batch * batch_size, total_files)
  batch_files <- file_list[start_idx:end_idx]
  
  # 初始化批次临时数据框（仅保留有效数据）
  batch_data <- data.frame(
    gene1 = character(), 
    gene2 = character(), 
    z_score = numeric(), 
    stringsAsFactors = FALSE
  )
  
  # 读取并过滤当前批次的每个文件
  for (i in 1:length(batch_files)) {
    file <- batch_files[i]
    target_gene <- basename(file)
    current_total <- start_idx + i - 1
    
    # 读取文件并立即过滤（核心：只保留有效数据）
    temp_data <- tryCatch({
      read_delim(
        file = file,
        delim = "\t",
        col_names = c("gene2", "z_score"),
        show_col_types = FALSE,
        col_types = cols(
          gene2 = col_character(),
          z_score = col_double()
        )
      ) %>%
        filter(z_score >= z_threshold) %>%  # 边读边过滤
        filter(!is.na(z_score)) %>%         # 过滤NA值
        mutate(gene1 = as.character(target_gene)) %>%  # 添加gene1列
        filter(gene1 != gene2)              # 过滤自环
    }, error = function(e) {
      # 出错时返回空数据框，避免程序中断
      return(data.frame(gene1 = character(), gene2 = character(), z_score = numeric()))
    })
    
    # 合并到批次数据（仅有效数据）
    if (nrow(temp_data) > 0) {
      batch_data <- bind_rows(batch_data, temp_data)
    }
    
    # 每1000个文件清理一次内存
    if (current_total %% 1000 == 0) {
      invisible(gc())
      cat("Zma数据集：已读取+过滤", current_total, "个文件，清理内存...\n")
    }
  }
  
  # 将当前批次有效数据写入临时文件
  if (nrow(batch_data) > 0) {
    temp_file <- paste0(temp_dir, "batch_", batch, ".tsv")
    write_tsv(batch_data, temp_file, col_names = TRUE)
    temp_file_paths <- c(temp_file_paths, temp_file)
    cat("Zma数据集：第", batch, "/", batch_num, "批次完成，写入临时文件：", temp_file, "（有效行数：", nrow(batch_data), "）\n")
  } else {
    cat("Zma数据集：第", batch, "/", batch_num, "批次无有效数据，跳过...\n")
  }
  
  # 清理批次内存
  rm(batch_data, temp_data)
  invisible(gc())
}

# ========== 合并所有临时文件（仅加载有效数据） ==========
if (length(temp_file_paths) == 0) {
  stop("错误：Zma数据集无有效边！请降低z_threshold（如1.5）重试。")
}

cat("\nZma数据集：合并所有临时批次文件...\n")
# 批量读取所有临时文件并去重
coexpr_clean <- temp_file_paths %>%
  lapply(function(x) read_tsv(x, show_col_types = FALSE)) %>%
  bind_rows() %>%
  mutate(gene_pair = paste(pmin(gene1, gene2), pmax(gene1, gene2), sep = "_")) %>%
  distinct(gene_pair, .keep_all = TRUE) %>%
  select(-gene_pair) %>%
  select(gene1, gene2, z_score)

# 清理临时文件（释放磁盘空间）
unlink(temp_dir, recursive = TRUE)
cat("Zma数据集：临时文件清理完成，有效边总数：", nrow(coexpr_clean), "\n")

# ========== 网络构建与导出（兼容igraph 2.0.0+） ==========
# 构建无向共表达网络
coexpr_network <- graph_from_data_frame(
  d = coexpr_clean,
  directed = FALSE,
  vertices = unique(c(coexpr_clean$gene1, coexpr_clean$gene2))
)

cat("Zma数据集：共表达网络构建完成！\n")
cat("节点数（基因数）：", vcount(coexpr_network), "\n")
cat("边数（共表达关系数）：", ecount(coexpr_network), "\n")
# 使用新版edge_density替代废弃的graph.density
cat("网络密度：", round(edge_density(coexpr_network), 6), "\n")

# 导出边列表和网络文件
edgelist_path <- paste0(output_dir, "Zma_coexpr_edgelist.txt")
write.table(
  coexpr_clean,
  file = edgelist_path,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
gml_path <- paste0(output_dir, "Osa_coexpr_network.gml")
# 修正：Zma的GML文件名（之前手误写成Osa，已修复）
gml_path <- paste0(output_dir, "Zma_coexpr_network.gml")
# 使用新版write_graph替代废弃的write.graph
write_graph(coexpr_network, gml_path, format = "gml")

cat("Zma数据集：文件导出完成：\n- ", edgelist_path, "\n- ", gml_path, "\n")

# 输出核心基因（度最高的10个）
node_degree <- degree(coexpr_network)
top_10_genes <- sort(node_degree, decreasing = TRUE)[1:10]
cat("\nZma数据集：度最高的10个核心基因：\n")
print(top_10_genes)

# 最终清理内存
rm(coexpr_clean, coexpr_network, node_degree, top_10_genes)
invisible(gc())

cat("\n===== Zma数据集处理全部完成！=====\n")