import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载提供的Excel文件
data = pd.read_excel("附件.xlsx")

# 从第一行提取X值（经度），跳过前两列
X = data.iloc[0, 2:].values

# 从第二列提取Y值（纬度），跳过第一行
Y = data.iloc[1:, 1].dropna().values

# 跳过第一行和前两列，提取Z值（深度）
Z = data.iloc[1:, 2:].values

# 为X和Y创建一个网格
X_mesh, Y_mesh = np.meshgrid(X, Y)

# 绘制热图
plt.figure(figsize=(10, 8))
plt.contourf(X_mesh, Y_mesh, Z, cmap="viridis")
plt.colorbar(label="Depth (m)")
plt.xlabel("Longitude (NM)")
plt.ylabel("Latitude (NM)")
plt.title("Seabed Depth Heatmap")
plt.show()

# 绘制3D曲面图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap="viridis", edgecolor='none')

# 设置标签和标题
ax.set_xlabel('Longitude (NM)')
ax.set_ylabel('Latitude (NM)')
ax.set_zlabel('Depth (m)')
ax.set_title('Seabed Depth 3D Surface Plot')
fig.colorbar(surf, ax=ax, label="Depth (m)", pad=0.1)

plt.show()


