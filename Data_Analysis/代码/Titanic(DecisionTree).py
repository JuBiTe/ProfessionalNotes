import pandas as pd
import numpy as np
from pandas import DataFrame
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score

#数据加载
train_data  = pd.read_csv("./Titanic_Data-master/train.csv")
test_data   = pd.read_csv("./Titanic_Data-master/test.csv")

#数据探索
print(train_data.info())
print("-" * 30)
print(train_data.describe())
print("-" * 30)
print(train_data.describe(include = ['O']))
print("-" * 30)
print(train_data.head())
print("-" * 30)
print(train_data.tail())


train_data['Age'].fillna(train_data['Age'].mean() , inplace = True)
test_data['Age'].fillna(test_data['Age'].mean() , inplace = True)

train_data['Fare'].fillna(train_data['Fare'].mean() , inplace = True)
test_data['Fare'].fillna(test_data['Fare'].mean() , inplace = True)



train_data['Embarked'].fillna('S' , inplace = True)
test_data['Embarked'].fillna('S', inplace = True)


#特征选择
features = ['Pclass' , 'Sex' , 'Age' , 'SibSp' , 'Parch' , 'Fare' , 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]


#转换字符串为数字

dvec = DictVectorizer(sparse = False)
train_features = pd.DataFrame(train_features)
train_features = dvec.fit_transform(train_features.to_dict(orient = 'record'))



clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(train_features , train_labels)

test_features = dvec.fit_transform(test_features.to_dict(orient = 'record'))
pred_labels = clf.predict(test_features)

acc_decision_tree = round(clf.score(train_features , train_labels) , 6)
print(u'The score accuracy is %s' % acc_decision_tree)






print("cross_val_score is %s" % np.mean(cross_val_score(clf , train_features , train_labels , cv = 10)))



export_graphviz(clf , out_file = 'Tianic.dot')













