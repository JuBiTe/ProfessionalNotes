import jieba
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score 


warnings.filterwarnings('ignore')

def cut_words(file_path):
    """
    对文本进行分词
    :param file_path 文本路径
    :return 用空格分词的字符串
    """
    text_with_spaces = ""
    text = open(file_path , 'r' , encoding = 'gb18030').read()
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

#读取一个文件夹下的文本
def loadfile(file_dir , label):
    
    """
    将路径下的所有文件加载
    :param file_dir:保存txt文件目录
    :param label: 保存标签
    :return 分词后的文档列表和标签 
    """
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
        file_path = file_dir +  '/' +  file
        words_list.append(cut_words(file_path))
        labels_list.append(label)
    
    return words_list , labels_list



#收集训练数据
train_words_list1 , train_labels1 = loadfile("./text_classification-master/text_classification/train/女性" , "女性")
train_words_list2 , train_labels2 = loadfile("./text_classification-master/text_classification/train/体育" , "体育")
train_words_list3 , train_labels3 = loadfile("./text_classification-master/text_classification/train/文学" , "文学")
train_words_list4 , train_labels4 = loadfile("./text_classification-master/text_classification/train/校园" , "校园")

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

#收集测试数据
test_words_list1 , test_labels1 = loadfile("./text_classification-master/text_classification/test/女性" , "女性")
test_words_list2 , test_labels2 = loadfile("./text_classification-master/text_classification/test/体育" , "体育")
test_words_list3 , test_labels3 = loadfile("./text_classification-master/text_classification/test/文学" , "文学")
test_words_list4 , test_labels4 = loadfile("./text_classification-master/text_classification/test/校园" , "校园")

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

#获得停词表
stop_words = open("./text_classification-master/text_classification/stop/stopword.txt" , 'r' , encoding = 'utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig')   # 列表头部\ufeff处理
stop_words = stop_words.split('\n')   # 根据分隔符分隔

#计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)


#获得训练集和测试集特征
train_features = tf.fit_transform(train_words_list)
test_features = tf.transform(test_words_list)


#多项式贝叶斯分类
clf = MultinomialNB(alpha = 0.001).fit(train_features , train_labels)
predicted_labels = clf.predict(test_features)

#计算准确率
print("The NaiveBayes's score is %s" % metrics.accuracy_score(test_labels , predicted_labels))
# The NaiveBayes's score is 0.91

#交叉验证
import numpy as np
from sklearn.model_selection import cross_val_score 
print("The cross_val_score is %s" % np.mean(cross_val_score(clf , train_features , train_labels , cv = 10)))

# The cross_val_score is 0.8862647624279043


