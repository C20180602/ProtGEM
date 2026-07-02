import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ========== 数据整理 ==========
# 本体
ontologies = ['BPO', 'CCO', 'MFO']

# 1. 模块消除类（蓝色）
remove_methods = ['w/o seq', 'w/o struct', 'w/o InterPro', 'only global', 'only local']
remove_BP = [0.4903, 0.5176, 0.2325, 0.5175, 0.1911]
remove_CC = [0.6941, 0.7482, 0.5806, 0.7519, 0.3446]
remove_MF = [0.8176, 0.8281, 0.3211, 0.8278, 0.1584]

# 2. 模块替换类（紫色）
replace_methods = ['Cross-attn', 'Recon loss', 'Pretrain GO', 'Norm vec']
replace_BP = [0.4908, 0.4888, 0.5015, 0.5019]
replace_CC = [0.7435, 0.7246, 0.7446, 0.7457]
replace_MF = [0.8153, 0.8262, 0.8209, 0.8201]

# 3. 完整模型（橙色）
full_BP = 0.5176
full_CC = 0.7518
full_MF = 0.8282

# 合并顺序：蓝色组 + 紫色组 + Full
all_methods = remove_methods + replace_methods + ['Full']
all_data = {
    'BPO': remove_BP + replace_BP + [full_BP],
    'CCO': remove_CC + replace_CC + [full_CC],
    'MFO': remove_MF + replace_MF + [full_MF],
}
full_values = {'BPO': full_BP, 'CCO': full_CC, 'MFO': full_MF}

# 蓝色系（5种，由浅到深）
blue_palette = ['#b3cde3', '#8cb3d9', '#6599c9', '#3e7fb3', '#1a5a96']
# 紫色系（4种，由浅到深）
purple_palette = ['#d1b3d8', '#b58bc4', '#9a64b0', '#7e3d9c']
orange = '#e69f00'

colors = blue_palette + purple_palette + [orange]

# ========== 绘图 ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 5))  # 宽度略微收窄更紧凑
x = np.arange(len(all_methods))
width = 0.75  # 收窄柱宽，让分组更紧凑

for ax, ont in zip(axes, ontologies):
    values = all_data[ont]
    bars = ax.bar(x, values, width, color=colors, edgecolor='black', linewidth=0.4)
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Fmax', fontsize=11)
    ax.set_title(ont, fontsize=13)
    ax.set_ylim([0, 0.9])  # 留出顶部空间标注数值
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 标注橙色Full模型的水平虚线及顶端数值
    full_val = full_values[ont]
    ax.axhline(y=full_val, color=orange, linestyle='--', linewidth=1.5, alpha=0.8)
    full_x_pos = x[-1]  # Full 在最后一个位置
    ax.text(full_x_pos, full_val + 0.015, f'{full_val:.4f}', 
            ha='center', va='bottom', color=orange, fontweight='bold', fontsize=9)

legend_elements = [
    Patch(facecolor=blue_palette[0], label='Ablation Study', alpha=0.8),
    Patch(facecolor=purple_palette[0], label='Alternative Strategies', alpha=0.8),
    Patch(facecolor=orange, label='Full model', alpha=0.8),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, 
           bbox_to_anchor=(0.5, -0.12))

plt.savefig('ablation_fmax_updated.png', dpi=300, bbox_inches='tight')
plt.show()