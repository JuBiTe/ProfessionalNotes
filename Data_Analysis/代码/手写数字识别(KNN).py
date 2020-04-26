from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

#载入数据
digits = load_digits()
data = digits.data

#显示图像
plt.gray()
plt.imshow(digits.images[0])
plt.show()

train_X , test_X , train_y , test_y = train_test_split(data , digits.target , test_size = 0.3 , random_state = 42)
#采用Z-score规范化
ss = preprocessing.StandardScaler()
train_ss_X = ss.fit_transform(train_X)
test_ss_X  = ss.transform(test_X)

#创建KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_X , train_y)
predict_y = knn.predict(test_ss_X)
print("KNN's accuracy score is %.4lf" % accuracy_score(test_y , predict_y))



#创建SVM分类器
svm = SVC()
svm.fit(train_ss_X , train_y)
predict_y = svm.predict(test_ss_X)
print("SVM 's accuracy score is %.4lf" % accuracy_score(test_y , predict_y))
#采用Min-Max规范化
mm = preprocessing.MinMaxScaler()
train_mm_X = mm.fit_transform(train_X)
test_mm_X  = mm.transform(test_X)
#创建Bayes分类器
mnb = MultinomialNB()
mnb.fit(train_mm_X , train_y)
predict_y = mnb.predict(test_mm_X)
print("Bayes ' s accuracy score is %.4lf" accuracy_score(test_y , predict_y))
#创建决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(train_mm_X , train_y)
predict_y = dtc.predict(test_mm_X)
print("CART's accuracy score is %.4lf" % accuracy_s)
