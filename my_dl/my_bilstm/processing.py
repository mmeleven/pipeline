#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# import numpy as np
import os

# tag字典
TAG = {
    '体育': 0,
    '女性': 1,
    '文学出版': 2,
    '校园': 3
}


def pre_processing(dir_path, out_path, mode='train'):
    """
    将文件夹的文件整理在一个txt中
    :param dir_path:
    :param out_path:
    :param mode:
    :return:
    """
    tag_list = os.listdir(dir_path)
    fw = open(out_path, 'w', encoding='utf-8')
    for tag in tag_list:
        tag_path = os.path.join(dir_path, tag)
        txt_list = os.listdir(tag_path)
        for txt_name in txt_list:
            filename = os.path.join(tag_path, txt_name)
            # print(filename)
            with open(filename, 'r', errors='ignore') as fr:
                content = fr.readline()
                if not content.split():
                    continue
                # print(content)
                if mode == 'train':
                    fw.write(content + '\t' + str(TAG[tag]) + '\n')
                else:
                    fw.write(content + '\n')
    fw.close()


def cut_for_embedding(filename, out_path, stopword_path='data\\stop\\stopword.txt', mode='train'):
    """
    将原始预料去除停用词分词，用于词向量训练
    :param filename:
    :param out_path:
    :param stopword_path:
    :param mode:
    :return:
    """
    import jieba
    with open(stopword_path, 'r', encoding='utf-8') as fr:
        stopwords = fr.readlines()
    stopwords = [word.strip('\n') for word in stopwords]
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    fw = open(out_path, 'w', encoding='utf-8')
    for line in lines:
        if mode == 'train':
            if len(line.strip('\n').split('\t')) > 1:
                content = line.strip('\n').split('\t')[0]
                # print(content)
                tag = line.strip('\n').split('\t')[1]
        else:
            content = line.strip('\n')
        seg_list = list(jieba.cut(content))
        for _ in seg_list[:]:
            if _ in stopwords or len(_.strip()) == 0:
                seg_list.remove(_)
        if mode == 'train':
            seg_list.append(tag)
        fw.write('\t'.join(seg_list) + '\n')
    fw.close()


if __name__ == '__main__':
    # 预处理
    pre_processing('data\\train_data', 'data\\clean\\train.txt')
    pre_processing('data\\test_data', 'data\\clean\\test_true.txt')
    pre_processing('data\\test_data', 'data\\clean\\test.txt', mode='test')
    # 分词
    cut_for_embedding('data\\clean\\train.txt', 'data\\clean\\train_seg.txt')
    cut_for_embedding('data\\clean\\test.txt', 'data\\clean\\test_seg.txt', mode='test')
