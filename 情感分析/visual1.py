import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载麒麟9000s评论数据
kiwi_data = pd.read_csv('data/kiwi_9000s_comments_with_labels.csv', encoding='utf-8')

# 创建一个映射字典，将数值标签映射到文本标签
label_mapping = {0: '积极', 1: '消极'}

# 使用映射字典转换标签
kiwi_data['label'] = kiwi_data['label'].map(label_mapping)

# 检查数据
print(kiwi_data.head())
# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 根据性别进行情感标签的分布分析
plt.figure(figsize=(10, 6))
sns.countplot(data=kiwi_data, x='label', hue='性别')
plt.title('情感标签分布（按性别）')
plt.xlabel('情感标签')
plt.ylabel('数量')
plt.legend(title='性别')
plt.show()

# 根据是否为SVIP进行情感标签的分布分析
plt.figure(figsize=(10, 6))
sns.countplot(data=kiwi_data, x='label', hue='svip')
plt.title('情感标签分布（按是否为SVIP）')
plt.xlabel('情感标签')
plt.ylabel('数量')
plt.legend(title='是否为SVIP')
plt.show()

# 根据性别和是否为SVIP进行情感标签的分布分析
plt.figure(figsize=(10, 6))
sns.countplot(data=kiwi_data, x='label', hue='性别')
plt.title('情感标签分布（按性别和是否为SVIP）')
plt.xlabel('情感标签')
plt.ylabel('数量')
plt.legend(title='性别')
plt.show()

# 根据IP归属地进行情感标签的分布分析
plt.figure(figsize=(10, 6))
sns.countplot(data=kiwi_data, x='label', hue='IP归属地')
plt.title('情感标签分布（按IP归属地）')
plt.xlabel('情感标签')
plt.ylabel('数量')
plt.legend(title='IP归属地')
plt.show()
