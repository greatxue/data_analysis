## 问题1

首先绘制出几何简图：

​                                                      <img src="plots/plot1-1.png" alt="plot1-1" style="zoom:25%;" />	

过波束张角$\angle CPF$的平分线$PJ$与坡面的交点$J$点作水平线，交波束$PC$和$PF$于$H$点和$G$点。

在$\Delta CHJ$中由正弦定理得，$\frac{\sin\angle HCJ}{HJ} = \frac{\sin\angle CHJ}{CJ}$

在$\Delta GFJ$中由正弦定理得，$\frac{\sin\angle GFJ}{GJ} = \frac{\sin\angle FGJ}{FJ}$

不难发现，$HJ = JG = D \tan\frac{\theta}{2}$

并且有角度关系 $\angle FGJ = \frac{\pi}{2}-\frac{\theta}{2}$，$\angle CHJ = \frac{\pi}{2}+\frac{\theta}{2}$；

​					  	 $\angle FGJ = \frac{\pi}{2}+\frac{\theta}{2}-\alpha$，$\angle HCJ = \frac{\pi}{2}-\frac{\theta}{2}-\alpha$

代入，不难解得 $CJ= \frac{D \sin \frac{\theta}{2}}{cos(\frac{\theta}{2} + \alpha)}$，$JF= \frac{D \sin \frac{\theta}{2}}{cos(\frac{\theta}{2} - \alpha)}$

<img src="plots/plot1-2.png" alt="plot1-2" style="zoom:25%;" />

<img src="plots/plot3-1.PNG" style="zoom:25%;" />

当测量船沿着侧线移动时，不难发现两条波束与坡面相交形成的三角形总是相似的。

以本图所显示的初末位置为例，有$\Delta CPF \sim \Delta HKG$

过$M$点向$PJ$作垂线，$KM = D - d \tan \beta$，且显然有 $\alpha = \beta$

由相似比得出$GM = \frac{KM}{PJ} FJ$，$MH = \frac{KM}{PJ} CJ$

不难算出$HF = JF+MH-MJ = \frac{D \sin \frac{\theta}{2}}{\cos(\frac{\theta}{2}-\alpha)} + $$\frac{(D-d\tan \alpha) \sin \frac{\theta}{2}}{\cos(\frac{\theta}{2}+\alpha)}$ $-\frac{d}{\cos \alpha}$

我们定义，在有坡度的情况下，条带的覆盖宽度 $\eta = \frac{HF}{JF}$ 

联立上述各式，$\eta = 1 - \frac{d}{2D}(\frac{1}{\tan \frac{\theta}{2}}+\tan \alpha)$

也可以这样推导：

由于平行关系，不难发现$\angle IOP = \angle POR = \frac{\theta}{2}$，因此$P$是$IR$中点，梯形$KMJI$的中位线$PQ = \frac{1}{2}(2D-d\tan\alpha)$，

以及有$OQ = PQ-PO = D - \frac{1}{2}d\tan\alpha-\frac{d}{2\tan\frac{\theta}{2}} $。

以$F$为顶点的几组平行相似三角形给出关系：

$\frac{OQ}{IJ}= \frac{OF}{IF} = \frac{HF}{AF} = \eta$

联立同样得到 $\eta = 1 - \frac{d}{2D}(\frac{1}{\tan \frac{\theta}{2}}+\tan \alpha)$

基于这样的模型，我们用代码填充对应表格，得到填充结果如图：

![result1](plots/result1.png)

## 问题2

<img src="plots/plot2.png" alt="plot2" style="zoom:25%;" />

结合化归的思想，我们倾向于将问题2中的情形转化为问题1。

据分析，问题1中测线方向与海底坡面的法向在水平面上投影的夹角为 $\beta = \frac{\pi}{2}$。推广到一般的角度$\beta$，我们沿着侧线重新建立如图所示的坡面$\angle DBC$（记作$\alpha_0$)，以达到和问题1类似的效果。不难表示出坡度角$\tan \alpha_0 = \frac{DC}{BC}$。

我们引入辅助平面$CDH$ $\parallel$ 平面$NEG$：在$\Delta CDH$中，$CD = CH\tan\alpha$

过$C$点作$AI$的垂线，交$AN$于$B$点：在$\Delta CHB$中，$CH = BC \cos\angle CAB = \cos(\beta-\frac{\pi}{2})$

联立上述各式，有$\alpha_0 = \arctan (\sin\beta \tan\alpha)$

作为对问题1中$\eta = 1 - \frac{d}{2D}(\frac{1}{\tan \frac{\theta}{2}}+\tan \alpha)$的推广，我们有$\eta = 1 - \frac{d}{2D}(\frac{1}{\tan \frac{\theta}{2}}+\sin\beta \tan\alpha)$

基于这样的模型，我们用代码填充对应表格，得到填充结果如图：

## 问题三

<img src="plots/plot3-2.PNG" style="zoom:25%;" />

从模型的简洁性考虑，我们不妨规划一组平行于矩形边界的测线组。我们使用动态规划的思路，迭代的算法，从边界情况开始考虑，直至所有的区域已经被全部覆盖。

<img src="plots/plot3-3.png" alt="plot3-3" style="zoom:25%;" /><img src="plots/plot3-4.png" alt="plot3-4" style="zoom:25%;" /><img src="plots/plot3-5.png" alt="plot3-5" style="zoom:25%;" />

现在我们讨论应该是采用东西向的测线组，还是南北向的测线组。如图，如果采用

对于覆盖率 $\eta$，过大的覆盖率必然导致测线条数太多造成冗杂，过小的覆盖率又会导致某些地段不能完全测准。结合实际情况和实践经验，考虑设定 $\eta = 0.1$为宜。

我们知道，$C$点是覆盖面可以扫到的最左侧边界。如果从西侧开始，对于第1条侧线，我们将过$C$点垂直于纸面的直线与西侧边界重合，构建基本的几何图形结构，可以得到$I$点的位置，过$I$点垂直于纸面的直线即为第1条侧线。根据既定的覆盖率，我们又可以推到第2条测线所在位置$K$点……以此类推，直到第$n$条侧线时，覆盖面可以扫到的最右侧边界$G$点第一次超越东侧边界为止。

在第一问中我们得到 $\eta = 1 - \frac{d}{2D}(\frac{1}{\tan \frac{\theta}{2}}+\tan\alpha)$，

长度参数$CJ= \frac{D \sin \frac{\theta}{2}}{cos(\frac{\theta}{2} - \alpha)}$，$MG= \frac{D \sin \frac{\theta}{2}}{cos(\frac{\theta}{2} - \alpha)}$，分别与对应的$D$对应。

如果把第一条测线记作第$0$次，我们约定第$i$次测线到垂直下端坡面的距离为$D_i$，第$i$次与第$i+1$次的测线间隔为$d_i$，第$i$次的线段长度分别为$CJ_i$和$MG_i$。

最开始的边界情况是：$CJ_0 \sin\alpha + D_0 = h_{west}$，可以得到$D_0$的值。

对于每一个$i>1$的情形，总有$D_i = D_{i-1}-d_{i-1}\tan \alpha$ 来算出新的$D_i$。

这样直到第一次满足$MG_i +\sum_{i=1}^{n}d_i \geq L$，跳出循环。

如果从东侧开始，则与上述过程完全类似。由于不一定能够最后一次的边界情况可能取不到等号，所以可能会覆盖一个实际上稍大的区域，因此从西向东和从东向西理应不完全一致，但是不应该相差过大。

事实上我们用代码分别尝试由东向西和由西向东排布的可能性，最后得出：

可视化结果为：

## 附录代码

### 第一问：

**填充表格：**

```python
import math
import pandas as pd

def calculate_depths_coverages_overlaps():
    # 定义参数
    D_center = 70  # 中心点的深度
    theta = math.radians(120)  # 换能器开角转换为弧度
    alpha = math.radians(1.5)  # 坡度角转换为弧度
    d = 200  # 相邻两条测线的间距
    distances_from_center = [-800, -600, -400, -200, 0, 200, 400, 600, 800]

    # 根据距中心的距离计算每个位置的深度、覆盖宽度和与前一条测线的重叠率
    depths_list = []
    coverage_widths_list = []
    overlap_percentages_list = []

    previous_width = None
    for dist in distances_from_center:
        # 修改了此行，使其与距离的正负符号相反
        D = D_center - dist * math.tan(alpha)
        depths_list.append(D)

        W_prime = D * math.sin(theta/2) / math.cos(theta/2 - alpha) \
        					+ (D - d * math.tan(alpha) * math.sin(theta/2)) / math.cos(theta/2 + alpha) \
        					- d / math.cos(alpha)
        coverage_widths_list.append(W_prime)

        if previous_width is not None:
            overlap = 1 - d / (2 * D) * (1 / math.tan(theta / 2) + math.tan(alpha))
            overlap_percentages_list.append(overlap * 100)
        else:
            overlap_percentages_list.append(0)  # 第一个点的重叠率为0
        previous_width = W_prime

    # 结果存储到DataFrame
    df_result10 = pd.DataFrame({
        '测线距中心点处的距离/m': distances_from_center,
        '海水深度/m': depths_list,
        '覆盖宽度/m': coverage_widths_list,
        '与前一条测线的重叠率/%': overlap_percentages_list
    })
    # 保存结果到Excel文件
    file_path = "../data/result1.xlsx"
    df_result10.to_excel(file_path, index=False)
    print(f"结果已保存到 {file_path}")

if __name__ == "__main__":
    calculate_depths_coverages_overlaps()
```

### 第二问：

**填充表格：**



### 第三问：

**计算测线条数并可视化：**

```python
import numpy as np
from scipy.optimize import root
from numpy import pi as pi

# 定义中间、西、东的深度值
center_depth = 110
west_depth = 110 + 2 * 1852 * np.tan(np.deg2rad(1.5))
east_depth = 110 - 2 * 1852 * np.tan(np.deg2rad(1.5))

def measure_from_shallow_to_deep():
    """由浅入深测量方法"""

    # 获取第一条测量线的深度
    def first_depth(x):
        """计算与西深度之间的差值"""
        return ((x * np.sin(pi/3)) / np.cos(pi/3 + np.deg2rad(1.5))) * np.sin(np.deg2rad(1.5)) + x - west_depth

    def get_d(x):
        """获取检测线之间的距离"""
        yita = 1 - (x / (2 * D)) * (1/np.tan(pi/3) + np.tan(np.deg2rad(1.5))) - 0.1
        return yita

    def get_fj(x):
        """获取剩余的检测长度，用于检查任务是否完成"""
        return (x * np.sin(pi/3) / np.cos(pi/3 - np.deg2rad(1.5))) * np.cos(np.deg2rad(1.5))

    # 初始化值
    x0 = 110
    result = root(first_depth, x0)
    first_depth_west = result.x[0]
    first_d = first_depth_west * np.sin(pi/3) * np.cos(np.deg2rad(1.5)) / np.cos(pi/3 + np.deg2rad(1.5))
    d_sum = first_d
    D = first_depth_west
    n = 1

    # 循环迭代测量
    while d_sum + get_fj(D) <= 4 * 1852:
        result = root(get_d, 100)
        new_d = result.x[0]
        d_sum += new_d
        D = west_depth - d_sum * np.tan(np.deg2rad(1.5))
        n += 1

    return n, n * 2 * 1852, n * 2

def measure_from_deep_to_shallow():
    """由深入浅测量方法"""

    # 获取第一条测量线的深度
    def first_depth(x):
        """计算与东深度之间的差值"""
        return -(x * np.sin(pi/3)) / np.cos(pi/3 - np.deg2rad(1.5)) * np.sin(np.deg2rad(1.5)) + x - east_depth

    def get_d(x):
        """获取检测线之间的距离"""
        yita = 1 + (x / (2 * D)) * (-1/np.tan(pi/3) + np.tan(np.deg2rad(1.5))) - 0.1
        return yita

    def get_fj(x):
        """获取剩余的检测长度，用于检查任务是否完成"""
        return (x * np.sin(pi/3) / np.cos(pi/3 + np.deg2rad(1.5))) * np.cos(np.deg2rad(1.5))

    # 初始化值
    x0 = 110
    result = root(first_depth, x0)
    first_depth_west = result.x[0]
    first_d = first_depth_west * np.sin(pi/3) * np.cos(np.deg2rad(1.5)) / np.cos(pi/3 + np.deg2rad(1.5))
    d_sum = first_d
    D = first_depth_west
    n = 1

    # 循环迭代测量
    while d_sum + get_fj(D) <= 4 * 1852:
        result = root(get_d, 100)
        new_d = result.x[0]
        d_sum += new_d
        D = east_depth + d_sum * np.tan(np.deg2rad(1.5))
        n += 1

    return n, n * 2 * 1852, n * 2

# 调用由浅入深的测量函数
n_shallow, distance_shallow, miles_shallow = measure_from_shallow_to_deep()

# 调用由深入浅的测量函数
n_deep, distance_deep, miles_deep = measure_from_deep_to_shallow()

# 输出结果
print('由浅入深：一共有', n_shallow, '条测线，总路程为', distance_shallow, '米', miles_shallow, '海里')
print('由深入浅：一共有', n_deep, '条测线，总路程为', distance_deep, '米', miles_deep, '海里')
```

