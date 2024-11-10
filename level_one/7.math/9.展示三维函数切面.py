import matplotlib.pyplot as plt
import numpy as np

# 创建 x, y 数据点
x = np.linspace(-5, 5, 100)  # x 值范围在 -5 到 5 之间，共 100 个点
y = np.linspace(-5, 5, 100)  # y 值范围在 -5 到 5 之间，共 100 个点
x, y = np.meshgrid(x, y)     # 创建网格，用于绘制 3D 曲面

# 定义三维函数 f(x, y)
def f(x, y):
    # 返回 x^2 + y^2
    return x**2 + y**2

# 计算 z 的值
z = f(x, y)

# 创建图形和轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 创建 3D 图形

# 绘制表面
surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.5)  # 使用颜色映射 'viridis'，透明度为 0.5

# 定义要突出显示的点
point_x, point_y = 1.0, 1.0  # 定义点的 x 和 y 坐标
point_z = f(point_x, point_y)  # 计算点的 z 坐标

# 绘制该点
ax.scatter(point_x, point_y, point_z, color='red', s=50)  # 用红色绘制该点，大小为 50

# 计算切平面的法线
normal = np.array([2 * point_x, 2 * point_y, -1])  # 法线向量

# 定义平面上的点 x_plane, y_plane, z_plane
x_plane = np.linspace(-5, 5, 10)
y_plane = np.linspace(-5, 5, 10)
x_plane, y_plane = np.meshgrid(x_plane, y_plane)
z_plane = (-normal[0] * (x_plane - point_x) - normal[1] * (y_plane - point_y)) / normal[2] + point_z

# 绘制切平面
ax.plot_surface(x_plane, y_plane, z_plane, color='yellow', alpha=0.5)  # 用黄色绘制切平面，透明度为 0.5

# 设置标签和标题
ax.set_xlabel('X axis')  # x 轴标签
ax.set_ylabel('Y axis')  # y 轴标签
ax.set_zlabel('Z axis')  # z 轴标签
ax.set_title("3D Surface Plot with Tangent Plane")  # 图标题

plt.show()  # 显示图形
