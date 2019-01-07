import pymysql
import os
from os import listdir
from os.path import isfile, join
import numpy as np


def get_filename(filefolderpath):
    """读取文件夹"""
    Files = [filefolderpath + f for f in listdir(filefolderpath) if isfile(join(filefolderpath, f))]
    return Files


def get_files(filepathlist):
    """读取文件夹中数据"""
    data = []
    for filepath in filepathlist:
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
            f.close()
        data.append(line)


def get_wordlist(wordlistpath):
    """加载词表"""
    wordList = np.load(wordlistpath)
    wordList = wordList.tolist()
    wordList = [word.decode('UTF-8') for word in wordList]
    return wordList


def get_wordvectors(vectorspath):
    """获取词向量矩阵"""
    wordVectors = np.load(vectorspath)
    return wordVectors


def get_ids(idspath):
    """获取句子中每个单词的id列表"""
    ids = np.load(idspath)
    return ids
def get_labels():
    left = [[1,0]]*12500
    right = [[0,1]]*12500
    left.extend(right)
    return np.array(left)



def batch_iter(x, y, batch_size):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = (i + 1) * batch_size
        if end_id <= data_len:
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        else:
            break


def get_train(ids, labels):
    x_train = np.concatenate((ids[:7500], ids[17500:]), axis=0)
    y_train = np.concatenate((labels[:7500], labels[17500:]), axis=0)
    return x_train, y_train


def get_val(ids, labels):
    x_val = np.concatenate((ids[7500:10000], ids[15000:17500]), axis=0)
    y_val = np.concatenate((labels[7500:10000], labels[15000:17500]), axis=0)
    return x_val, y_val


def get_test(ids, labels):
    x_test = np.concatenate((ids[10000:12500], ids[12500:15000]), axis=0)
    y_test = np.concatenate((labels[10000:12500], labels[12500:15000]), axis=0)
    return x_test, y_test


