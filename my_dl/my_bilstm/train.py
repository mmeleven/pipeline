#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import numpy as np


def load_embedding(model_path='data\\clean\\word2vec.model'):
    """
    加载词向量model
    :param model_path:
    :return:
    """
    import gensim
    model = gensim.models.Word2Vec.load(model_path)
    # print(model['我'])
    return model


def load_data(filename, model_path='data\\clean\\word2vec.model', sequence_length=20, mode='train'):
    """
    加载训练数据，返回np.array格式X,Y
    :param filename:
    :param model_path:
    :param sequence_length:
    :param mode:
    :return:
    """
    embedding_model = load_embedding(model_path)
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    X = np.zeros((len(lines), sequence_length, 100), dtype=np.float)
    Y = np.zeros(len(lines))
    for index, line in enumerate(lines):
        split_list = line.strip('\r\n').split('\t')
        if mode == 'train' and len(split_list) > 1:
            seg_list = split_list[:-1]
            if split_list[-1] in ['0', '1', '2', '3']:
                Y[index] = float(split_list[-1])
        else:
            seg_list = split_list
        for seg_index, seg in enumerate(seg_list):
            if seg_index < sequence_length and seg in embedding_model:
                X[index, seg_index, :] = embedding_model[seg]
    if mode == 'train':
        return X, Y
    else:
        return X


if __name__ == '__main__':
    # load data
    train_X, train_Y = load_data('data\\clean\\train_seg.txt')
    print(np.shape(train_X))  # (3311, 20, 100)
    print(np.shape(train_Y))  # (3311,)

    test_X = load_data('data\\clean\\test_seg.txt', mode='test')
    print(np.shape(test_X))  # (200, 20, 100)
