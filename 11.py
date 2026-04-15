(deeploc2) [tmpuser@malab25 node网络特征32维]$ python 11.py

================================================================================
📊 探查GAF文件（GO标签）
================================================================================

================================================================================
📊 TRAIN GAF文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/train.gaf
📌 train GAF文件基本信息:
   行数: 126815, 列数: 5
   列名: [0, 1, 2, 3, 4]
   前5行数据:
            0                                                  1  2   3  4
0  A0A0B5E2H5  GO:0003674,GO:0003824,GO:0004364,GO:0015036,GO...  4  12  8
1  A0A0B5EC24  GO:0003674,GO:0003824,GO:0004364,GO:0016740,GO...  2   5  3
2  A0A0B5EC52  GO:0003674,GO:0003824,GO:0004364,GO:0016034,GO...  4   8  4
3  A0A0G4DBR5  GO:0003674,GO:0003824,GO:0008194,GO:0016740,GO...  6   9  3
4  A0A0K2CSW0  GO:0003674,GO:0003676,GO:0003677,GO:0003700,GO...  2   6  4

   Protein ID信息:
   总数: 126815, 唯一数: 126815
   前5个ID: ['A0A0B5E2H5', 'A0A0B5EC24', 'A0A0B5EC52', 'A0A0G4DBR5', 'A0A0K2CSW0']
   是否有重复ID: False

   GO术语信息:
   前5个GO术语: ['GO:0003674,GO:0003824,GO:0004364,GO:0015036,GO:0015038,GO:0016209,GO:0016491,GO:0016667,GO:0016672,GO:0016740,GO:0016765,GO:0045174', 'GO:0003674,GO:0003824,GO:0004364,GO:0016740,GO:0016765', 'GO:0003674,GO:0003824,GO:0004364,GO:0016034,GO:0016740,GO:0016765,GO:0016853,GO:0016859', 'GO:0003674,GO:0003824,GO:0008194,GO:0016740,GO:0016757,GO:0016758,GO:0033838,GO:0035251,GO:0046527', 'GO:0003674,GO:0003676,GO:0003677,GO:0003700,GO:0005488,GO:0140110']
   有效GO术语数 (以GO:开头): 126815/126815

================================================================================
📊 VAL GAF文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/val.gaf
📌 val GAF文件基本信息:
   行数: 15852, 列数: 5
   列名: [0, 1, 2, 3, 4]
   前5行数据:
            0                                                  1  2   3   4
0  A0A0B5EA67  GO:0003674,GO:0003824,GO:0004364,GO:0016740,GO...  2   5   3
1  A0A0K2CTB2  GO:0000976,GO:0001067,GO:0003674,GO:0003676,GO...  2   9   7
2  A0A0R0E2R0  GO:0003674,GO:0003824,GO:0004842,GO:0005488,GO...  3  13  10
3  A0A0R0E3G0  GO:0000166,GO:0003674,GO:0003824,GO:0005215,GO...  5  31  26
4  A0A0R0E3K4        GO:0003674,GO:0005488,GO:0005515,GO:0046983  1   4   3

   Protein ID信息:
   总数: 15852, 唯一数: 15852
   前5个ID: ['A0A0B5EA67', 'A0A0K2CTB2', 'A0A0R0E2R0', 'A0A0R0E3G0', 'A0A0R0E3K4']
   是否有重复ID: False

   GO术语信息:
   前5个GO术语: ['GO:0003674,GO:0003824,GO:0004364,GO:0016740,GO:0016765', 'GO:0000976,GO:0001067,GO:0003674,GO:0003676,GO:0003677,GO:0003690,GO:0005488,GO:0043565,GO:1990837', 'GO:0003674,GO:0003824,GO:0004842,GO:0005488,GO:0005515,GO:0016740,GO:0016746,GO:0016755,GO:0019787,GO:0051087,GO:0061630,GO:0061659,GO:0140096', 'GO:0000166,GO:0003674,GO:0003824,GO:0005215,GO:0005319,GO:0005488,GO:0005524,GO:0015399,GO:0016462,GO:0016787,GO:0016817,GO:0016818,GO:0016887,GO:0017076,GO:0017111,GO:0022804,GO:0022857,GO:0030554,GO:0032553,GO:0032555,GO:0032559,GO:0035639,GO:0036094,GO:0042626,GO:0043167,GO:0043168,GO:0097367,GO:0140359,GO:0140657,GO:1901265,GO:1901363', 'GO:0003674,GO:0005488,GO:0005515,GO:0046983']
   有效GO术语数 (以GO:开头): 15852/15852

================================================================================
📊 TEST GAF文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/test.gaf
📌 test GAF文件基本信息:
   行数: 15853, 列数: 5
   列名: [0, 1, 2, 3, 4]
   前5行数据:
            0                                                  1  2   3   4
0  A0A075W8S1  GO:0003674,GO:0005215,GO:0015291,GO:0015297,GO...  1   6   5
1  A0A0B5E2M0  GO:0003674,GO:0003824,GO:0004364,GO:0016740,GO...  2   5   3
2  A0A0B5E8V1  GO:0003674,GO:0003824,GO:0004364,GO:0016740,GO...  2   5   3
3  A0A0K2CTV0        GO:0003674,GO:0003676,GO:0003677,GO:0005488  1   4   3
4  A0A0R0E2A0  GO:0000295,GO:0003674,GO:0005215,GO:0005346,GO...  2  16  14

   Protein ID信息:
   总数: 15853, 唯一数: 15853
   前5个ID: ['A0A075W8S1', 'A0A0B5E2M0', 'A0A0B5E8V1', 'A0A0K2CTV0', 'A0A0R0E2A0']
   是否有重复ID: False

   GO术语信息:
   前5个GO术语: ['GO:0003674,GO:0005215,GO:0015291,GO:0015297,GO:0022804,GO:0022857', 'GO:0003674,GO:0003824,GO:0004364,GO:0016740,GO:0016765', 'GO:0003674,GO:0003824,GO:0004364,GO:0016740,GO:0016765', 'GO:0003674,GO:0003676,GO:0003677,GO:0005488', 'GO:0000295,GO:0003674,GO:0005215,GO:0005346,GO:0008514,GO:0015215,GO:0015216,GO:0015291,GO:0015297,GO:0015605,GO:0015932,GO:0022804,GO:0022857,GO:0046964,GO:1901505,GO:1901682']
   有效GO术语数 (以GO:开头): 15853/15853

================================================================================
📊 探查CSV文件（亚细胞特征）
================================================================================

================================================================================
📊 TRAIN CSV文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/train.csv
📌 train CSV文件基本信息:
   行数: 126815, 列数: 18
   列名: ['Protein_ID', 'Localizations', 'Signals', 'Membrane types', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome', 'Peripheral', 'Transmembrane', 'Lipid anchor', 'Soluble']
   数据类型:
Protein_ID                object
Localizations             object
Signals                   object
Membrane types            object
Cytoplasm                float64
Nucleus                  float64
Extracellular            float64
Cell membrane            float64
Mitochondrion            float64
Plastid                  float64
Endoplasmic reticulum    float64
Lysosome/Vacuole         float64
Golgi apparatus          float64
Peroxisome               float64
Peripheral               float64
Transmembrane            float64
Lipid anchor             float64
Soluble                  float64
dtype: object
   前5行数据:
   Protein_ID Localizations                      Signals      Membrane types  Cytoplasm  Nucleus  Extracellular  ...  Lysosome/Vacuole  Golgi apparatus  Peroxisome  Peripheral  Transmembrane  Lipid anchor  Soluble
0  A0A0B5E2H5       Plastid  Chloroplast transit peptide  Peripheral|Soluble     0.2037   0.1683         0.0449  ...            0.0821           0.0571      0.1492      0.6643         0.0757        0.0460   0.6576
1  A0A0B5EC24     Cytoplasm                          NaN             Soluble     0.7442   0.3832         0.1267  ...            0.2721           0.2078      0.3233      0.3203         0.0477        0.0575   0.8406
2  A0A0B5EC52     Cytoplasm                          NaN             Soluble     0.6799   0.3535         0.0285  ...            0.3546           0.1837      0.2649      0.3317         0.0699        0.0472   0.7992
3  A0A0G4DBR5     Cytoplasm                          NaN             Soluble     0.7526   0.4633         0.0265  ...            0.1192           0.0745      0.0156      0.4898         0.0497        0.0320   0.7838
4  A0A0K2CSW0       Nucleus  Nuclear localization signal             Soluble     0.2597   0.9660         0.0021  ...            0.0260           0.0025      0.0028      0.3746         0.1270        0.0254   0.7010

[5 rows x 18 columns]

   Protein ID列: Protein_ID
   总数: 126815, 唯一数: 126815
   前5个ID: ['A0A0B5E2H5', 'A0A0B5EC24', 'A0A0B5EC52', 'A0A0G4DBR5', 'A0A0K2CSW0']

   数值特征列: ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome', 'Peripheral', 'Transmembrane', 'Lipid anchor', 'Soluble']
   数值特征列数: 14
   特征值范围 (第一列): 0.0281000006943941 ~ 0.9352999925613404

================================================================================
📊 VAL CSV文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/val.csv
📌 val CSV文件基本信息:
   行数: 15852, 列数: 18
   列名: ['Protein_ID', 'Localizations', 'Signals', 'Membrane types', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome', 'Peripheral', 'Transmembrane', 'Lipid anchor', 'Soluble']
   数据类型:
Protein_ID                object
Localizations             object
Signals                   object
Membrane types            object
Cytoplasm                float64
Nucleus                  float64
Extracellular            float64
Cell membrane            float64
Mitochondrion            float64
Plastid                  float64
Endoplasmic reticulum    float64
Lysosome/Vacuole         float64
Golgi apparatus          float64
Peroxisome               float64
Peripheral               float64
Transmembrane            float64
Lipid anchor             float64
Soluble                  float64
dtype: object
   前5行数据:
   Protein_ID                                      Localizations                              Signals      Membrane types  Cytoplasm  ...  Peroxisome  Peripheral  Transmembrane  Lipid anchor  Soluble
0  A0A0B5EA67                                          Cytoplasm         Peroxisomal targeting signal             Soluble     0.4960  ...      0.7085      0.3292         0.0512        0.0476   0.8338
1  A0A0K2CTB2                                            Nucleus          Nuclear localization signal             Soluble     0.2877  ...      0.0699      0.4344         0.0527        0.0381   0.8505
2  A0A0R0E2R0                                  Cytoplasm|Nucleus          Nuclear localization signal  Peripheral|Soluble     0.4895  ...      0.0318      0.6361         0.0350        0.1002   0.8068
3  A0A0R0E3G0  Cell membrane|Endoplasmic reticulum|Lysosome/V...  Signal peptide|Transmembrane domain       Transmembrane     0.2067  ...      0.0661      0.0951         0.9926        0.0720   0.0842
4  A0A0R0E3K4                                            Nucleus          Nuclear localization signal             Soluble     0.2492  ...      0.1251      0.3682         0.1048        0.2451   0.7972

[5 rows x 18 columns]

   Protein ID列: Protein_ID
   总数: 15852, 唯一数: 15852
   前5个ID: ['A0A0B5EA67', 'A0A0K2CTB2', 'A0A0R0E2R0', 'A0A0R0E3G0', 'A0A0R0E3K4']

   数值特征列: ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome', 'Peripheral', 'Transmembrane', 'Lipid anchor', 'Soluble']
   数值特征列数: 14
   特征值范围 (第一列): 0.0320000015199184 ~ 0.9341999888420104

================================================================================
📊 TEST CSV文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/test.csv
📌 test CSV文件基本信息:
   行数: 15853, 列数: 18
   列名: ['Protein_ID', 'Localizations', 'Signals', 'Membrane types', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome', 'Peripheral', 'Transmembrane', 'Lipid anchor', 'Soluble']
   数据类型:
Protein_ID                object
Localizations             object
Signals                   object
Membrane types            object
Cytoplasm                float64
Nucleus                  float64
Extracellular            float64
Cell membrane            float64
Mitochondrion            float64
Plastid                  float64
Endoplasmic reticulum    float64
Lysosome/Vacuole         float64
Golgi apparatus          float64
Peroxisome               float64
Peripheral               float64
Transmembrane            float64
Lipid anchor             float64
Soluble                  float64
dtype: object
   前5行数据:
   Protein_ID                   Localizations                              Signals Membrane types  Cytoplasm  Nucleus  ...  Golgi apparatus  Peroxisome  Peripheral  Transmembrane  Lipid anchor  Soluble
0  A0A075W8S1                   Cell membrane                 Transmembrane domain  Transmembrane     0.0824   0.0984  ...           0.3983      0.0098      0.0570         0.9938        0.1010   0.0917
1  A0A0B5E2M0                       Cytoplasm         Peroxisomal targeting signal        Soluble     0.7393   0.5042  ...           0.1536      0.4750      0.2331         0.0449        0.0482   0.8638
2  A0A0B5E8V1                       Cytoplasm         Peroxisomal targeting signal        Soluble     0.6858   0.4285  ...           0.3531      0.4732      0.3421         0.0344        0.0659   0.8567
3  A0A0K2CTV0                         Nucleus          Nuclear localization signal        Soluble     0.1810   0.9491  ...           0.0134      0.0073      0.3892         0.0638        0.0228   0.7996
4  A0A0R0E2A0  Cell membrane|Lysosome/Vacuole  Signal peptide|Transmembrane domain  Transmembrane     0.1395   0.0872  ...           0.6020      0.2203      0.2086         0.9644        0.0853   0.1746

[5 rows x 18 columns]

   Protein ID列: Protein_ID
   总数: 15853, 唯一数: 15853
   前5个ID: ['A0A075W8S1', 'A0A0B5E2M0', 'A0A0B5E8V1', 'A0A0K2CTV0', 'A0A0R0E2A0']

   数值特征列: ['Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane', 'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome', 'Peripheral', 'Transmembrane', 'Lipid anchor', 'Soluble']
   数值特征列数: 14
   特征值范围 (第一列): 0.0291000008583068 ~ 0.9358999729156494

================================================================================
📊 探查PT文件（ESM2特征）
================================================================================

================================================================================
📊 TRAIN PT文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/train.pt
📌 train PT文件基本信息:
   数据类型: <class 'dict'>
   字典键: ['protein_ids', 'cls_features', 'model_name', 'target_layer', 'max_sequence_length']

   protein_ids信息:
   数量: 126815
   前5个ID: ['A0A0B5E2H5', 'A0A0B5EC24', 'A0A0B5EC52', 'A0A0G4DBR5', 'A0A0K2CSW0']
   类型: <class 'list'>

   cls_features信息:
   形状: torch.Size([126815, 2560])
   数据类型: torch.float32
   数值范围: -2.036921739578247 ~ 15.239013671875
   维度: 2560 (ESM2特征维度)

================================================================================
📊 VAL PT文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/val.pt
📌 val PT文件基本信息:
   数据类型: <class 'dict'>
   字典键: ['protein_ids', 'cls_features', 'model_name', 'target_layer', 'max_sequence_length']

   protein_ids信息:
   数量: 15852
   前5个ID: ['A0A0B5EA67', 'A0A0K2CTB2', 'A0A0R0E2R0', 'A0A0R0E3G0', 'A0A0R0E3K4']
   类型: <class 'list'>

   cls_features信息:
   形状: torch.Size([15852, 2560])
   数据类型: torch.float32
   数值范围: -2.0366978645324707 ~ 15.232596397399902
   维度: 2560 (ESM2特征维度)

================================================================================
📊 TEST PT文件
================================================================================
✅ 文件存在: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/test.pt
📌 test PT文件基本信息:
   数据类型: <class 'dict'>
   字典键: ['protein_ids', 'cls_features', 'model_name', 'target_layer', 'max_sequence_length']

   protein_ids信息:
   数量: 15853
   前5个ID: ['A0A075W8S1', 'A0A0B5E2M0', 'A0A0B5E8V1', 'A0A0K2CTV0', 'A0A0R0E2A0']
   类型: <class 'list'>

   cls_features信息:
   形状: torch.Size([15853, 2560])
   数据类型: torch.float32
   数值范围: -2.2031259536743164 ~ 15.229881286621094
   维度: 2560 (ESM2特征维度)

================================================================================
📊 跨文件一致性检查
================================================================================

================================================================================
📊 TRAIN 集一致性检查
================================================================================
   GAF ID数: 126815
   CSV ID数: 126815
   PT ID数: 126815
   三者共同ID数: 126815
   GAF-CSV交集: 126815
   GAF-PT交集: 126815
   CSV-PT交集: 126815
   仅GAF存在的ID数: 0
   仅CSV存在的ID数: 0
   仅PT存在的ID数: 0

================================================================================
📊 VAL 集一致性检查
================================================================================
   GAF ID数: 15852
   CSV ID数: 15852
   PT ID数: 15852
   三者共同ID数: 15852
   GAF-CSV交集: 15852
   GAF-PT交集: 15852
   CSV-PT交集: 15852
   仅GAF存在的ID数: 0
   仅CSV存在的ID数: 0
   仅PT存在的ID数: 0

================================================================================
📊 TEST 集一致性检查
================================================================================
   GAF ID数: 15853
   CSV ID数: 15853
   PT ID数: 15853
   三者共同ID数: 15853
   GAF-CSV交集: 15853
   GAF-PT交集: 15853
   CSV-PT交集: 15853
   仅GAF存在的ID数: 0
   仅CSV存在的ID数: 0
   仅PT存在的ID数: 0

================================================================================
📊 数据格式探查完成！
================================================================================

✅ 探查结果已保存至: /misc/hard_disk/others_res/zhangdy/wangluo/node网络特征32维/MF数据/split_data/data_exploration_results.pkl