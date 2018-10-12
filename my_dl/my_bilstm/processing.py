#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# import numpy as np
import os

TAG = {
    '体育': 0,
    '女性': 1,
    '文学出版': 2,
    '校园': 3
}


def pre_processing(dir_path, out_path, mode='train'):
    tag_list = os.listdir(dir_path)
    fw = open(out_path, 'w', encoding='utf-8')
    for tag in tag_list:
        tagpath = os.path.join(dir_path, tag)
        txtlist = os.listdir(tagpath)
        for txtname in txtlist:
            filename = os.path.join(tagpath, txtname)
            # print(filename)
            with open(filename, 'r', errors='ignore') as fr:
                content = fr.readline()
                # print(content)
                if mode == 'train':
                    fw.write(content + '\t' + str(TAG[tag]) + '\n')
                else:
                    fw.write(content + '\n')
    fw.close()


if __name__ == '__main__':
    pre_processing('data\\train_data', 'data\\clean\\train.txt')
    pre_processing('data\\test_data', 'data\\clean\\test_true.txt')
    pre_processing('data\\test_data', 'data\\clean\\test.txt', mode='test')
