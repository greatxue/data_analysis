import math
import pandas as pd

def calculate():
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
        					+ (D - d * math.tan(alpha) * math.sin(theta/2)) / math.cos(theta/2 + alpha)\
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
        '与前一条测线的重叠率/%':overlap_percentages_list
    })
    # 保存结果到Excel文件
    file_path = "../data/result1.xlsx"
    df_result10.to_excel(file_path, index=False)
    print(f"结果已保存到 {file_path}")

if __name__ == "__main__":
    calculate()