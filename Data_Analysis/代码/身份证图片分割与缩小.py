from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
import PIL.Image as image


def load_data(file_path):
    f = open(file_path , 'rb')
    data = []
    img = image.open(f)
    width , height = img.size
    for x in range(width):
        for y in range(height):
            c1 , c2 , c3 = img.getpixel((x , y))
            #规范化处理，自定义Min_Max规范化
            data.append([(c1+1)/256.0, (c2+1)/256.0, (c3+1)/256.0])
    f.close()
    return np.mat(data) , width , height
    
    
img , width , height = load_data('./图像分割/人像侧.jpg')
kmeans = KMeans(n_clusters = 16)
label =  kmeans.fit_predict(img)
#将图像聚类结果转化成图像尺寸的矩阵
label = label.reshape([width , height])
#创建新图像，保存聚类压缩后的结果
pic_narrow = image.new('RGB' , (width , height))
for x in range(width):
    for y in range(height):
        #得到质心特征
        c1 = kmeans.cluster_centers_[label[x , y] , 0]
        c2 = kmeans.cluster_centers_[label[x , y] , 1]
        c3 = kmeans.cluster_centers_[label[x , y] , 2]
        #进行反变换
        pic_narrow.putpixel((x , y) , (int(c1 * 256) - 1 , int(c2 * 256) - 1 , int(c3 * 256) - 1))

#保存图像
pic_narrow.save('ID_narrow.jpg')