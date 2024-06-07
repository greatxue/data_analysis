# 导入必要的库
import numpy as np
import pandas as pd
from numpy import pi as pi
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan
from numpy import arctan as act

# 设置pandas的显示选项，使其能够显示所有的列和行
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 定义常数
sp = np.deg2rad(1.5)  # 将1.5度转换为弧度

# 定义函数get_width，用于计算宽度
def get_width(beta, length):
    depth = 120 - length * sin(beta - pi/2) * tan(sp)
    alpha = act(sin(beta) * tan(sp))
    cf = depth * sin(pi / 3) / cos(pi/3 + alpha) + depth * sin(pi / 3) / cos(pi/3 - alpha)
    return cf

# 定义输入角度和长度的数组
angle = np.array([0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4])
angle_string = np.array(['0', 'pi/4', 'pi/2', '3*pi/4', 'pi', '5*pi/4', '3*pi/2', '7*pi/4'])
length = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1]) * 1852  # 1852是海里到米的转换系数

# 初始化一个8x8的矩阵来存储结果
graph = np.zeros((8, 8))

# 用双重循环计算每个角度和长度组合的宽度
for m in range(0, 8):
    for n in range(0, 8):
        graph[m, n] = round(get_width(angle[m], length[n]))

# 将结果矩阵转换为DataFrame，以便于查看和保存
df = pd.DataFrame(graph, index=angle_string, columns=length/1852)

# 打印DataFrame
print(df)

# 将DataFrame保存为CSV文件
df.to_csv('data.csv')