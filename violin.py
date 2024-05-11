import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 读取CSV文件
df = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_3.csv')

# 设置显示选项，使输出为非科学计数法
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 创建小提琴图
plt.figure(figsize=(10, 6))  # 设置图形大小
sns.violinplot(data=df.iloc[:, 0])  # 假设第三列的索引为0

# 添加标题和标签，并指定字体
font_path = "/home/visitor/Huang/Analytical-Method/simfang.ttf"  # 替换为你的字体文件路径
font_prop = fm.FontProperties(fname=font_path)
plt.title('第3列数据的小提琴图', fontproperties=font_prop)  # 使用指定字体
plt.xlabel('X轴标签', fontproperties=font_prop)  # 使用指定字体
plt.ylabel('第3列数据', fontproperties=font_prop)  # 使用指定字体

# 保存图像
plt.savefig('/home/visitor/Huang/Analytical-Method/violin.png')

# 显示图形
plt.show()
