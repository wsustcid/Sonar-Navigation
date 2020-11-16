# -*- coding: utf-8 -*-
'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2020-01-10 10:08:25
@LastEditTime: 2020-01-17 10:35:13
'''

""""
版本1： 
1. 不使用数据生成器，一次读入所有数据集
2. 希望减少硬盘与GPU之间的数据传输，提升GPU的利用率
3. 之前的逻辑是：先读入所有数据ID,然后划分训练与验证，训练时根据ID表示的地址去硬盘中读图片，每次只读一个batch-->及时使用了多线程，GPU的利用率也上不去；
4. 在内存足够(8G)的情况下，完全可以直接将所有数据一次读入，使用model.fit()进行训练

"""

import csv
import numpy as np
from math import ceil, floor

from sklearn.model_selection import train_test_split


def read_data(csv_path, length, dim=2, target_height=None, target_width=None):
    """ Reading ultrasonic data from csv file
    Args: 
     - csv_path: the path of the csv file
     - length: the length of time-ordered sequence data to be read for each sample

    Return: 
     - train_X, train_Y, valid_X, valid_Y
    """
    
    height = length
    width = 16
    
    if dim == 3:
        # for a tensor representation 
        length=3*length

    X = []
    Y = []
    with open(csv_path, 'r') as f:
        f_reader = csv.reader(f)
        for line in f_reader:
            distance = [float(line[j+i*18]) for i in range(length) for j in range(16)]
            # use the last angular value in sequence data as label
            angle  = float(line[17+(length-1)*18])
            #[float(line[17+i*18]) for i in range(length)]
            X.append(distance)
            Y.append(angle)
    
    print("{} samples with length {} have been read from {}.".format(len(X), 
                                                                     length,
                                                                     csv_path))
    n_samples = len(X)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)


    ## Date representation
    if target_height == None: target_height = height
    if target_width  == None: target_width  = width 
        
    h_repeats = int(ceil(target_height/float(height)))
    w_repeats = int(ceil(target_width/float(width)))
    
    h_delta = h_repeats*height-target_height
    w_delta = w_repeats*width-target_width
    
    h_start = int(floor(h_delta/2.0))
    h_end = h_repeats*height-(h_delta-h_start)
    
    w_start = int(floor(w_delta/2.0))
    w_end = w_repeats*width-(w_delta-w_start)
    
    if dim == 1:
        # shape (n, 16xlength)
        pass
      
    elif dim == 2:
        # reshape
        X = X.reshape(n_samples, 1, height, width)
        
        X = X.repeat(h_repeats, axis=2)
        X = X.repeat(w_repeats, axis=3)
        
        X = X[:,:,h_start:h_end, w_start:w_end]
                
    elif dim == 3:
        # 之前是直接把二维图片重复3次，形成第三维
        # dist = dist.repeat(dim, axis=0)
        X = X.reshape(n_samples, 3, height, width)
        
        X = X.repeat(h_repeats, axis=2)
        X = X.repeat(w_repeats, axis=3)
        
        X = X[:,:,h_start:h_end, w_start:w_end]
                
    else:
        raise IOError("Unidentified dim value: use 1, 2 or 3")

    print("X shape: {}, Y shape: {}".format(X.shape, Y.shape))
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y,
                                                          test_size=0.2,
                                                          random_state=10,
                                                          shuffle = True)

    return X_train, Y_train, X_valid, Y_valid 