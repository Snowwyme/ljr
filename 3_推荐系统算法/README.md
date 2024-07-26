# 说明

这段代码实现了一个基于矩阵分解的推荐系统，基于dmsc_v2数据集进行训练模型，最后对测试数据集进行测试

# 导入库

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dot, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
```
其中pandas 用于数据处理，sklearn 用于数据划分和特征编码，tensorflow 和 keras 用于构建和训练神经网络模型，matplotlib 用于数据可视化。

# 数据处理

##数据读取和可视化

'''python
# 读取数据
data_ratings = pd.read_csv('dmsc_v2/ratings.csv')

# 数据探索和可视化
print(data_ratings.head())
print(data_ratings.describe())
plt.hist(data_ratings['rating'], bins=5)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Ratings')
plt.show()

'''
从 CSV 文件中读取用户评分数据。打印数据的前几行和描述统计信息。绘制评分分布的直方图。

##特征编码

