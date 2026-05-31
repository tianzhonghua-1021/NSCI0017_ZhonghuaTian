import pandas as pd
import os

# ======================
# 路径设置
# ======================
structure_folder = r"C:\Users\ttbas\Documents\0314analysis\0430summary\dc_gnn\graph_T,P_TDA\data_prepare"
dataset_path = r"C:\Users\ttbas\Documents\0314analysis\0430summary\dc_gnn\graph_T,P_TDA\dataset_form.csv"
output_path = r"C:\Users\ttbas\Documents\0314analysis\0430summary\dataset_with_wt.csv"

# ======================
# 原子质量 (g/mol)
# ======================
M_C = 12.011
M_H = 1.008
M_H2 = 2.016

# ======================
# 读取数据集
# ======================
df = pd.read_csv(dataset_path)

# ======================
# 缓存结构质量（避免重复读取）
# ======================
mass_cache = {}

def get_structure_mass(file_id):
    if file_id in mass_cache:
        return mass_cache[file_id]

    file_path = os.path.join(structure_folder, f"{file_id}.csv")

    structure = pd.read_csv(file_path)

    # 统计元素数量
    n_C = (structure['Element'] == 'C').sum()
    n_H = (structure['Element'] == 'H').sum()

    # 基底质量
    m_sub = n_C * M_C + n_H * M_H

    # 默认：吸附位点 = C原子数
    n_site = n_C

    mass_cache[file_id] = (m_sub, n_site)
    return m_sub, n_site

# ======================
# 计算 wt%
# ======================
wt_list = []

for _, row in df.iterrows():
    file_id = int(row['File ID'])
    theta = row['theta']

    m_sub, n_site = get_structure_mass(file_id)

    # 吸附H2数
    n_H2 = theta * n_site

    # H2质量
    m_H2 = n_H2 * M_H2

    # wt%
    wt_percent = (m_H2 / (m_sub + m_H2)) * 100 if (m_sub + m_H2) > 0 else 0

    wt_list.append(wt_percent)

# 添加列
df['wt%'] = wt_list

# ======================
# 保存
# ======================
df.to_csv(output_path, index=False)

print("✅ 已生成新文件:", output_path)