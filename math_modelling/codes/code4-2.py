import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi as pi
from scipy.optimize import root


def plan(deep, alpha, limit, __x, __y, __z, direction):
    def first_depth(x):
        # try to initialize the starting location of the ship
        return ((x * np.sin(pi / 3)) / np.cos(pi / 3 + alpha)) * np.sin(alpha) + x - deep

    def get_d(x):
        # hold overlap rate to the least value 0.1 to get to the largest step length
        yita = 1 - (x / (2 * D)) * (1 / np.tan(pi / 3) + np.tan(alpha)) - 0.1
        return yita

    def get_fj(x):
        # get the remaining length of the detection and use it to test if mission is accomplished
        return (x * np.sin(pi / 3) / np.cos(pi / 3 - alpha)) * np.cos(alpha)

    # get the depth of the first detection
    x0 = 30
    result = root(first_depth, x0)

    # get its distance the west edge
    first_depth_west = result.x[0]
    first_d = first_depth_west * np.sin(pi / 3) * np.cos(alpha) / np.cos(pi / 3 + alpha)
    # set the initial length
    d_sum = first_d
    D = first_depth_west
    n = 1
    distance_list = [d_sum]

    # iteration, begin!
    while d_sum + get_fj(D) <= limit * 1852:  # stop if the distance is larger than 4
        # step1 get new d between detecting lines
        result = root(get_d, 30)
        new_d = result.x[0]
        # step2 calculate the distance to the west edge
        d_sum += new_d
        distance_list.append(d_sum)
        # step3 calculate the new depth
        D = deep - d_sum * np.tan(alpha)
        n += 1

    # print out the outcome
    if limit == 2:
        width = 2.5
    else:
        width = 2
    print('一共有', n, '条测线，总路程为', n * width * 1852, '米', n * width, '海里', '遗漏率为', missing(__x, __y, __z, direction, distance_list))
    # 创建一个新的图形对象
    fig, ax = plt.subplots()

    # 绘制南北向的线
    x = [0, 2.5]  # x坐标，这里假设示意图的宽度是1
    for item in distance_list:
        y = [item / 1852, item / 1852]  # y坐标，这里假设示意图的高度是1，线条位于垂直方向的中间位置
        ax.plot(x, y, color='green', linewidth=1)  # 设置线条的颜色为红色，线宽为2

    # 设置图形对象的属性
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.1)
    ax.set_aspect(1)


def missing(__x, __y, __z, direction, d_list):
    d_list = np.array(d_list)
    missing_number = 0
    if direction == 'x':
        d_list += (__x[0] - 0.02) * 1852
        for i in range(len(__x)):
            xx = __x[i] * 1852
            zz = __z[i]
            for index in range(len(d_list) - 1):
                if d_list[index] < xx < d_list[index + 1]:
                    if zz / (xx - d_list[index]) < np.tan(pi/6) and zz / (d_list[index + 1] - xx) < np.tan(pi/6):
                        missing_number += 1
                    break
                else:
                    continue
    else:
        d_list += (__y[0] - 0.02) * 1852
        for i in range(len(__x)):
            yy = __y[i] * 1852
            zz = __z[i]
            for index in range(len(d_list) - 1):
                if d_list[index] < yy < d_list[index + 1]:
                    if zz / (yy - d_list[index]) < np.tan(pi / 6) and zz / (d_list[index + 1] - yy) < np.tan(pi / 6):
                        missing_number += 1
                    break
                else:
                    continue
    return missing_number / len(__x)


def get_angle(pa, pb):
    # 计算拟合面的法向量
    normal_vector = np.array([pa, pb, -1])  # 拟合面的法向量为(a, b, -1)

    # 水平面法向量（假设为竖直向上的）
    horizontal_vector = np.array([0, 0, 1])

    # 计算法向量之间的夹角
    cosine_angle = np.dot(normal_vector, horizontal_vector) / (
        np.linalg.norm(normal_vector) * np.linalg.norm(horizontal_vector)
    )
    angle_in_radians = np.arccos(cosine_angle)
    if angle_in_radians >= 3:
        return pi - angle_in_radians
    else:
        return angle_in_radians


df = pd.read_excel('/Users/zhangshuhan/Desktop/副本附件(1).xls', engine='xlrd')
matrix = np.array(df.values[0:, 1:])
x = matrix[0, 1:]
y = matrix[1:, 0]
z = matrix[1:, 1:]
X = []
Y = []
Z = []
for m, _x in enumerate(x):
    if m == 100:
        break
    for n, _y in enumerate(y):
        if n == 125:
            break
        X.append(_x)
        Y.append(_y)
        Z.append(z[n, m])
# 进行线性回归拟合
A = np.column_stack([X, Y, np.ones_like(X)])
coefficients, residuals, _, _ = np.linalg.lstsq(A, Z, rcond=-1)
# 提取回归系数
a, b, c = coefficients
# 生成拟合的平面面点坐标
x_fit = np.linspace(min(X), max(X), 10)
y_fit = np.linspace(min(Y), max(Y), 10)
x_fit, y_fit = np.meshgrid(x_fit, y_fit)
z_fit = a * x_fit + b * y_fit + c

# 绘制原始数据的散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

angle = get_angle(a / 1852, b / 1852)
depth = a * 2 + b * 1.25 + c
print('blue')
plan(depth, angle, limit=2, __x=X, __y=Y, __z=Z, direction='x')
# 绘制拟合的平面
ax.plot_surface(x_fit, y_fit, z_fit, color='blue', alpha=0.5, label='Fitted Plane')


X = []
Y = []
Z = []
for m, _x in enumerate(x[100:]):
    for n, _y in enumerate(y[:125]):
        X.append(_x)
        Y.append(_y)
        Z.append(z[n, m + 100])
# 进行线性回归拟合
A = np.column_stack([X, Y, np.ones_like(X)])
coefficients, residuals, _, _ = np.linalg.lstsq(A, Z, rcond=-1)

# 提取回归系数
a, b, c = coefficients

# 生成拟合的平面面点坐标
x_fit = np.linspace(min(X), max(X), 10)
y_fit = np.linspace(min(Y), max(Y), 10)
x_fit, y_fit = np.meshgrid(x_fit, y_fit)
z_fit = a * x_fit + b * y_fit + c

angle = get_angle(a / 1852, b / 1852)
depth = a * 4 + b * 1.25 + c
print('green')
plan(depth, angle, limit=2, __x=X, __y=Y, __z=Z, direction='x')


# 绘制拟合的平面
ax.plot_surface(x_fit, y_fit, z_fit, color='green', alpha=0.5, label='Fitted Plane')


X = []
Y = []
Z = []
for m, _x in enumerate(x[:100]):
    for n, _y in enumerate(y[125:]):
        X.append(_x)
        Y.append(_y)
        Z.append(z[n + 125, m])
# 进行线性回归拟合
A = np.column_stack([X, Y, np.ones_like(X)])
coefficients, residuals, _, _ = np.linalg.lstsq(A, Z, rcond=-1)

# 提取回归系数
a, b, c = coefficients

# 生成拟合的平面面点坐标
x_fit = np.linspace(min(X), max(X), 10)
y_fit = np.linspace(min(Y), max(Y), 10)
x_fit, y_fit = np.meshgrid(x_fit, y_fit)
z_fit = a * x_fit + b * y_fit + c

angle = get_angle(a / 1852, b / 1852)
depth = a * 1 + b * 5 + c
print('red')
plan(depth, angle, limit=2.5, __x=X, __y=Y, __z=Z, direction='y')


# 绘制拟合的平面
ax.plot_surface(x_fit, y_fit, z_fit, color='red', alpha=0.5, label='Fitted Plane')


X = []
Y = []
Z = []
for m, _x in enumerate(x[100:]):
    for n, _y in enumerate(y[125:]):
        X.append(_x)
        Y.append(_y)
        Z.append(z[n + 125, m + 100])
# 进行线性回归拟合
A = np.column_stack([X, Y, np.ones_like(X)])
coefficients, residuals, _, _ = np.linalg.lstsq(A, Z, rcond=-1)

# 提取回归系数
a, b, c = coefficients

# 生成拟合的平面面点坐标
x_fit = np.linspace(min(X), max(X), 10)
y_fit = np.linspace(min(Y), max(Y), 10)
x_fit, y_fit = np.meshgrid(x_fit, y_fit)
z_fit = a * x_fit + b * y_fit + c

angle = get_angle(a / 1852, b / 1852)
depth = a * 4 + b * 3.75 + c
print('yellow')
plan(depth, angle, limit=2, __x=X, __y=Y, __z=Z, direction='x')


# 绘制拟合的平面
ax.plot_surface(x_fit, y_fit, z_fit, color='yellow', alpha=0.5, label='Fitted Plane')


# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# 显示图像
# plt.show()
