import matplotlib.pyplot as plt
import numpy as np

# 假设你的矩阵数据
matrix = [[11287, 557],
          [694, 11460]]

# 创建一个新的图像
plt.figure(figsize=(6, 6))

# 使用imshow函数绘制矩阵，viridis作为颜色映射
plt.imshow(matrix, cmap='viridis', interpolation='nearest')

# 添加颜色条
plt.colorbar(label='Value')

# 添加x轴和y轴标签
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 添加标题
plt.title('Confusion Matrix')

# 显示图像
plt.show()

