from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv("./kmeans-master/data.csv" , encoding = 'gbk')


train_x = data.iloc[: , 1:]
kmeans = KMeans(n_clusters = 3)
#Min-Max规范化
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)

result = pd.concat((data , pd.DataFrame(predict_y)) , axis = 1)
result.rename({0 : u"聚类"} , axis = 1 , inplace = True)
result