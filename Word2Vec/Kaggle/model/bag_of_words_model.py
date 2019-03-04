# -*- coding=utf-8 -*-

import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup # 解析网页tag
from sklearn.feature_extraction.text import CountVectorizer #文本统计
from sklearn.model_selection import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import nltk
#nltk.download()
from nltk.corpus import stopwords

# 用pandas读入训练数据
datafile = os.path.join("..","data","labeledTrainData.tsv") #文件路径字符串拼接
#.tsv后缀可以使用.read_csv()接口读取
df = pd.read_csv(datafile, sep="\t", escapechar='\\')
print("Number of reviews: {}".format(len(df)))
print(df.head(2))

# 对影评数据进行预处理, 大概有以下几个环节:
# 1. 去掉html标签
# 2. 移除标点
# 3. 切分成词/token
# 4. 去掉停用词
# 5. 重组为新的句子

"""
def display(text, title):
    print(title)
    print("\n----------------我 是 分 割 线---------------")
    print(text)
    
raw_example = df["review"][0]
display(raw_example, "原始数据")

# 去掉网页相关的一些tag,如换行符<br /><br />等
example = BeautifulSoup(raw_example, "html.parser").get_text()
display(example,"去掉HTML标签的数据")

# 使用正则时提取只包含字母的单词
example_letters = re.sub(r'[^a-zA-Z]', ' ', example)
display(example_letters, "去掉标点的数据")

# 将字符全部转换成小写
words = example_letters.lower().split()
display(words, "纯词列表数据")

# 去掉停用词数据
# 使用nltk自带的停用词数据
words_nostop = [w for w in words if w not in stopwords.words("english")]
# 使用自定义的停用词语料
stopwords = {}.fromkeys([line.rstrip() for line in open("../stopwords.txt")])
words_nostop = [w for w in words if w not in stopwords]
display(words_nostop, "去掉停用词数据")
"""

eng_stopwords = set(stopwords.words("english"))
# eng_stopwords = set(stopwords)

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower(text).split()
    words = [w for w in words if w not in stopwords]
    return " ".join(words)

# 清洗数据添加到dataframe里
df["clean_review"] = df.review.apply(clean_text)
df.head()

# 抽取bag of words 特征(使用sklearn的CountVectorizer)
vectorizer = CountVectorizer(max_features = 5000)
train_data_features = vextorizer.fit_transform(df.clean_review).toarray()
print(train_data_features.shape)

# 训练分类器
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, df.sentiment)

# 使用混淆矩阵查看效果
confusion = confusion_matirx(df.sentiment,forest.predict(train_data_features))
print(confusion)

# 删除不占用的变量
del df
del train_data_features

# 读取测试数据进行预测
testfile = os.path.join("..", "data", "testData.tsv")
df = pd.read_csv(datafile, sep="\t", "\\")
print("Number of reviews:{}".format(len(df)))
df["clean_reviews"] = df.reviews.apply(clean_text)
print(df.head(2))
test_data_features = vextorizer.fit_transform(df.clean_review).toarray()
print(test_data_features.shape)
result = forest.predict(test_data_features)
output = pd.DataFrame({"id":df.id,"sentiment":result})
print(output.head(2))

# 保存训练模型
output.to_csv(os.path.join("..","data","Bag_of_Words_model.csv"), index=False)
del df
del test_data_features
