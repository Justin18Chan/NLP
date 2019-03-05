# -*- coding:utf-8 -*-

"""  使用word2vec训练词向量   """

import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import nltk.data
#nltk.download()
from nltk.corpus import stopwords

from gensim.models.word2vec import Word2Vec

def load_dataset(name, nrows=None):
    datasets = {"unlabeled_train":"unlabeledTrainData.tsv",
                "labeled_train":"labeledTrainData.tsv",
                "test":"testData.tsv"
            }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join("..","data",datasets[name])
    print(data_file)
    df = pd.read_csv(data_file, sep='\t', escapechar='\\', nrows=nrows)
    print("Number of reviews {}".format(len(df)))
    return df

# 读入无标签数据 用于训练word2vec 词向量模型,也可以把有标签数据一起加入训练,准确度更高
df = load_dataset("unlabeled_train")
print(df.head(2))

# 清洗数据,做数据预处理
# 参数remove_stopwords=True配置去除停用词
eng_stopwords = set(stopwords.words("english"))
#eng_stopwords = {}.fromkeys([line.rstrip for line in open("../stopwords.txt")])

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

# from nltk.tokenize import sent_tokenize 分句
# 使用nltk自带的分句sentence tokenize对文本进行分句处理.
# from nltk.tokenize import word_tokenize 分词
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def print_call_counts(f):
    n = 0
    def wrapped(*args, **kwargs):
        nonlocal n
        n += 1
        if n % 1000 == 1:
            print('method {} called {} times'.format(f.__name__, n))
        return f(*args, **kwargs)
    return wrapped

@print_call_counts
def split_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences

sentences = sum(df.review.apply(split_sentences),[])
print('{} reviews -> {} sentences'.format(len(df), len(sentences)))

# 用gensim 训练word2vec模型
import logging
logging.basicConfig(format='%(astime)s:%(levelname)s:%(message)s',level=logging.INFO)

# 设定词向量训练的参数
num_features = 300 # Word vector dimensionality,300-500 ,词向量维数
min_word_count = 40 # Minnum word count 
num_workers = 4  #Number of threads to run in parallel 同时工作线程个数
context = 10 #Context window size 文本滑窗大小
downsampling = 1e-3 # Downsample setting for frequent words
model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)

print('Training model ...')
model = Word2Vec(sentences, 
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count, 
                          window=context,
                          sample=downsampling)
# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model.save(os.path.join('..', 'notebooks', model_name))

# 查看训练的词向量结果
# model.doesnt_match从一堆词中找到不匹配的词
print(model.doesnt_match("man woman child kitchen".split()))
print(model.doesnt_match('france england germany berlin'.split()))
# model.most_similar(word)找到与word相关性最大的词
print(model.most_similar("man"))


