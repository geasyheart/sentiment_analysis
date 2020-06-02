"""
将正负面评论转为矩阵
"""
import os
import re
from typing import List

import gensim
import jieba
import numpy as np
import pandas as pd

stop_words = []
with open('data/baidu_stopwords.txt', 'r') as f:
    for line in f:
        stop_words.append(line.replace("\n", ''))

model = gensim.models.Word2Vec.load('data/simple.zh.text.model')


def get_wordvec(words: List) -> np.ndarray:
    vecs = []
    for word in words:
        try:
            vec = model[word]
            vecs.append(vec)
        except KeyError:
            continue
    return np.array(vecs, dtype='float')


def build_vecs(d):
    file_vec_avg = []
    for file in os.listdir(d):
        filename = os.path.join(d, file)
        with open(filename, "rb") as f:
            line = f.read()
            try:
                line = line.decode("GB18030").encode("utf-8").decode("utf-8")
            except UnicodeDecodeError:
                continue
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）《×]+", "", line)
            line = [i for i in jieba.cut(line) if i not in stop_words]

            vecs = get_wordvec(line)
            if len(vecs) <= 0:
                continue
            file_sum = sum(vecs) / len(vecs)
            file_vec_avg.append(file_sum)

            # print(filevecs)
    return file_vec_avg


def main():
    pos_vecs = build_vecs("data/ChnSentiCorp_htl_ba_2000/pos")
    neg_vecs = build_vecs("data/ChnSentiCorp_htl_ba_2000/neg")

    Y = np.concatenate((np.ones(len(pos_vecs)), np.zeros(len(neg_vecs))))

    X = pos_vecs[:]

    for neg in neg_vecs:
        X.append(neg)

    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)

    data = pd.concat([df_y, df_x], axis=1)
    data.to_csv("data/2000_data.csv")


if __name__ == '__main__':
    main()
