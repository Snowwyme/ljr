import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dot, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

# 特征编码
le_user = LabelEncoder()
le_movie = LabelEncoder()
data_ratings['user_encoded'] = le_user.fit_transform(data_ratings['userId'])
data_ratings['movie_encoded'] = le_movie.fit_transform(data_ratings['movieId'])

# 选择特征和标签
X = data_ratings[['user_encoded', 'movie_encoded']]
y = data_ratings['rating']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型结构
class MatrixFactorizationModel(Model):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(MatrixFactorizationModel, self).__init__()
        self.user_embedding = Embedding(num_users, embedding_dim)
        self.movie_embedding = Embedding(num_movies, embedding_dim)
        self.dot = Dot(axes=1)
        self.prediction = Dense(1, activation='linear')

    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:, 0])
        user_vec = Flatten()(user_vec)
        movie_vec = self.movie_embedding(inputs[:, 1])
        movie_vec = Flatten()(movie_vec)
        dot_product = self.dot([user_vec, movie_vec])
        prediction = self.prediction(dot_product)
        return prediction

# 实例化模型
num_users = len(le_user.classes_)
num_movies = len(le_movie.classes_)
embedding_dim = 10
model = MatrixFactorizationModel(num_users, num_movies, embedding_dim)

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# 准备测试数据格式
X_test_reshaped = X_test.values.reshape(-1, 2)

# 训练模型
history = model.fit(X_train.values.reshape(-1, 2), y_train, epochs=1, batch_size=32, verbose=1, validation_split=0.2)

# 评估模型
predictions = model.predict(X_test_reshaped).flatten()
mse = tf.keras.losses.MSE(y_test, predictions).numpy()
print(f'Test MSE: {mse}')

# 计算更多的评估指标
rmse = tf.sqrt(mse).numpy()
mae = tf.keras.losses.MAE(y_test, predictions).numpy()
print(f'Test RMSE: {rmse}')
print(f'Test MAE: {mae}')

# 保存模型
model.save('matrix_factorization_model.keras')

# 加载模型并进行预测
from tensorflow.keras import load_model

loaded_model = load_model('matrix_factorization_model.keras', custom_objects={'MatrixFactorizationModel': MatrixFactorizationModel})
loaded_predictions = loaded_model.predict(X_test_reshaped).flatten()
loaded_mse = tf.keras.losses.MSE(y_test, loaded_predictions).numpy()
print(f'Loaded Model Test MSE: {loaded_mse}')


