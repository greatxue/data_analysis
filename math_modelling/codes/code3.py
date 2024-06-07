import numpy as np
from scipy.optimize import root
from numpy import pi as pi
import matplotlib.pyplot as plt

# 初始化深度数据
center_depth = 110
west_depth = 110 + 2 * 1852 * np.tan(np.deg2rad(1.5))
east_depth = 110 - 2 * 1852 * np.tan(np.deg2rad(1.5))

def compute_distance_from_east_to_west():
    """从东向西计算船的检测线距离"""
    # 初始化船的位置
    def first_depth(x):
        return -(x * np.sin(pi/3)) / np.cos(pi/3 - np.deg2rad(1.5)) * np.sin(np.deg2rad(1.5)) + x - east_depth

    def get_d(x):
        return 1 + (x / (2 * D)) * (-1/np.tan(pi/3) + np.tan(np.deg2rad(1.5))) - 0.1

    def get_fj(x):
        return (x * np.sin(pi/3) / np.cos(pi/3 + np.deg2rad(1.5))) * np.cos(np.deg2rad(1.5))

    x0 = 110
    result = root(first_depth, x0)
    first_depth_west = result.x[0]
    first_d = first_depth_west * np.sin(pi/3) * np.cos(np.deg2rad(1.5)) / np.cos(pi/3 + np.deg2rad(1.5))
    d_sum = first_d
    D = first_depth_west
    n = 1
    distance_list = [d_sum]
    while d_sum + get_fj(D) <= 4 * 1852:
        result = root(get_d, 100)
        new_d = result.x[0]
        d_sum += new_d
        distance_list.append(d_sum)
        D = east_depth + d_sum * np.tan(np.deg2rad(1.5))
        n += 1

    print('从东向西一共有', n, '条测线，总路程为', n * 2 * 1852, '米', n * 2, '海里')
    return distance_list

def compute_distance_from_west_to_east():
    """从西向东计算船的检测线距离"""
    # 初始化船的位置
    def first_depth(x):
        return ((x*np.sin(pi/3)) / np.cos(pi/3 + np.deg2rad(1.5))) * np.sin(np.deg2rad(1.5)) + x - west_depth

    def get_d(x):
        return 1 - (x / (2 * D)) * (1/np.tan(pi/3) + np.tan(np.deg2rad(1.5))) - 0.1

    def get_fj(x):
        return (x * np.sin(pi/3) / np.cos(pi/3 - np.deg2rad(1.5))) * np.cos(np.deg2rad(1.5))

    x0 = 110
    result = root(first_depth, x0)
    first_depth_west = result.x[0]
    first_d = first_depth_west * np.sin(pi/3) * np.cos(np.deg2rad(1.5)) / np.cos(pi/3 + np.deg2rad(1.5))
    d_sum = first_d
    D = first_depth_west
    n = 1
    distance_list = [d_sum]
    while d_sum + get_fj(D) <= 4 * 1852:
        result = root(get_d, 100)
        new_d = result.x[0]
        d_sum += new_d
        distance_list.append(d_sum)
        D = west_depth - d_sum * np.tan(np.deg2rad(1.5))
        n += 1

    print('从西向东一共有', n, '条测线，总路程为', n * 2 * 1852, '米', n * 2, '海里')
    return distance_list

# 计算从东向西和从西向东的船的检测线距离
distance_list_east = compute_distance_from_east_to_west()
distance_list_west = compute_distance_from_west_to_east()

# 具体参数
print("东起", distance_list_east)
print("西起", distance_list_west)

# 创建图形显示结果
fig, ax = plt.subplots()

# 绘制从东向西的检测线
x = [0, 2]
for item in distance_list_east:
    y = [4 - item / 1852, 4 - item / 1852]
    ax.plot(x, y, color='red', linewidth=1)

# 绘制从西向东的检测线
for item in distance_list_west:
    y = [item / 1852, item / 1852]
    ax.plot(x, y, color='blue', linewidth=1)

ax.set_xlim(0, 2)
ax.set_ylim(0, 4)
ax.set_aspect(1)
plt.show()