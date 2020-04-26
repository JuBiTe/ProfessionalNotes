from sklearn import svm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#数据加载
data = pd.read_csv("./breast_cancer_data-master/data.csv")
pd.set_option('display.max_columns' , None)

#数据探索

print(data.columns)

print(data.describe())


sns.countplot(data['diagnosis'] , label = 'Count')
plt.show()
corr = data[features_mean].corr()
plt.figure(figsize = (14 , 14))
# annot=True显示每个方格的数据
sns.heatmap(corr , annot = True)
plt.show()

# 特征选择
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean'] 

train , test = train_test_split(data , test_size = 0.3)
train_X = train[features_remain]
train_y = train['diagnosis']
test_X  = test[features_remain]
test_y  = test['diagnosis']

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
from sklearn import preprocessing

ss = preprocessing.StandardScaler()
train_X = ss.fit_transform(train_X)
test_X  = ss.transform(test_X)

model = svm.SVC()
model.fit(train_X , train_y)
prediction = model.predict(test_X)
print("The accuracy score is %s" % metrics.accuracy_score(test_y , prediction))

from sklearn.model_selection import cross_val_score
print("The k cross_val_score is %s" % np.mean(cross_val_score(model , train_X , train_y , cv = 10)))



#利用LinearSVC训练全部特征
#准备训练和测试数据
linear_train , linear_test = train_test_split(data , test_size = 0.3)
linear_train_X = linear_train[features_mean]
linear_test_X  = linear_test[features_mean]
linear_train_y = linear_train['diagnosis']
linear_test_y  = linear_test['diagnosis']

#规范化数据
linear_train_X = ss.fit_transform(linear_train_X)
linear_test_X = ss.transform(linear_test_X)

lsvc = svm.LinearSVC()
lsvc.fit(linear_train_X , linear_train_y)
lsvc_prediction = lsvc.predict(linear_test_X)
print("The accuracy score is %s" % metrics.accuracy_score(linear_test_y , lsvc_prediction))


print("The k cross_val_score is %s" % np.mean(cross_val_score(lsvc , linear_train_X , linear_train_y , cv = 10)))


