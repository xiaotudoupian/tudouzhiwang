# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 数据 =====================
labels = ['Overall', 'Extremely Rare', 'Generally Rare', 'Medium', 'Common']
coral = [0.8177, 0.1742, 0.3799, 0.7852, 0.8496]
mlp =   [0.8499, 0.3212, 0.5433, 0.8220, 0.8714]

x = np.arange(len(labels))
width = 0.35

# ===================== 2. 最简单的画图方式 =====================
fig, ax = plt.subplots(figsize=(12, 8))

# 画柱子
rects1 = ax.bar(x - width/2, coral, width, label='CORAL Domain Adaptation', color='#1f77b4')
rects2 = ax.bar(x + width/2, mlp,   width, label='Species-Specific MLP Branch', color='#ff7f0e')

# 加数值
ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')

# ===================== 3. 简单的设置 =====================
ax.set_ylabel('F1 Score')
ax.set_xlabel('GO Term Category')
ax.set_title('Performance Comparison of Network Feature Processing Strategies')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1) # 顶部多留一点空间
ax.legend(loc='lower right') # 图例放在右下角，绝对安全

# ===================== 4. 保存并显示 =====================
plt.tight_layout()
plt.savefig('Simple_Correct_Plot.png', dpi=300)
plt.savefig('Simple_Correct_Plot.pdf')
plt.show()