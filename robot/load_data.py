# -*- coding: utf-8 -*-
# Version: June 2 14:04:18 CEST 2017

import numpy as np
# import csv


def prepare_sets(path, header):
    '''
    Read the feature matrix from csv file
    '''
    with open(path) as csvfile:
        if header:
            next(csvfile) # ignore header
        data = [row.strip().split(',') for row in csvfile]
    return data


def prepare_response(path, header):
    '''
    Reads the response vector from csv file
    '''
    with open(path) as csvfile:
        if header:
            next(csvfile) # ignore the header
        data = []
        for row in csvfile:
            data.append(int(row.strip().split(',')[0]))
    return data


def load_weights():
    path_train_x = './data/train_weights_example.csv'
    path_valid_x = './data/valid_weights_example.csv'
    path_test_x = './data/test_weights_example.csv'

    train_set_x = np.asarray(prepare_sets(path_train_x, False), dtype=np.float32)
    valid_set_x = np.asarray(prepare_sets(path_valid_x, False), dtype=np.float32)
    test_set_x = np.asarray(prepare_sets(path_test_x, False), dtype= np.float32)

    return train_set_x, test_set_x, valid_set_x


if __name__ == '__main__':
    train, valid, test = load_weights()
    
    print(train.shape)
    print(valid.shape)
    print(test.shape)