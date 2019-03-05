# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

def load_dataset(name, nrows=None):
    datasets = {"unlabeled_train":"unlabeledTrainData.tsv",
                'labeled_train': 'labeledTrainData.tsv',
                'test': 'testData.tsv'}
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join("..","data",datasets[name])
    df = pd.read_csv(data_file, sep="\t", escapechar='\\', nrows=nrows)
    print("Number of reviews {}".format(len(df)))
    return df

eng_stopwords = set(stopwords.words("english"))

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

# 读入之前训练好的Word2Vec模型
model_name = '300features_40minwords_10context.model'
model = Word2Vec.load(os.path.join(".",model_name))

# 可以根据Word2Vec的结果去对影评文本进行编码
# 编码方式有一点粗暴,简单来说就是把这句话中的词向量做平均
df = load_dataset('labeled_train')
print(df.head(2))

def to_review_vector(review):
    words = clean_text(review, remove_stopwords=True)
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0))

train_data_features = df.review.apply(to_review_vector)
print(train_data_features.head(2))

# 使用随机森林RFC构建分类器
forest = RandomForestClassifier(n_estimators=100, random_state=8)
forest = forest.fit(train_data_features, df.sentiment)

# 同样在训练集上试试, 确保模型能正常work
confusion_M = confusion_matrix(df.sentiment, forest.predict(train_data_features))
print(confusion_M)

del df
del train_data_features

df = load_dataset("test")
df.head(2)

test_data_features = df.review.apply(to_review_vector)
test_data_features.head(2)

result = forest.predict(test_data_features)
output = pd.DataFrame({"id":df.id,"sentiment":result})
output.to_csv(os.path.join("..","data","Word2Vec_model.csv"),index=False)
print(output.head(2))

