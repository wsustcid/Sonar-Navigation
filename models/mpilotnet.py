# -*- coding: utf-8 -*-
'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2020-01-10 09:49:43
@LastEditTime: 2020-01-17 09:28:04
'''


from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Flatten, Dense, Dropout

from keras import backend as K
# set channels_first for all models
K.set_image_data_format("channels_first")



def mpilotnet(channels, height, width):
    """
    baseline 建立思路：
    1. 五层卷积不动，根据卷积确定合适的输入尺寸
    2. 前期实验表明(确定最佳时序时)，多个全连接层效果并不如单层的好，因此我们仅保留一层，并添加dropout
    3. 隐藏层大小根据最佳时序对应的确定？
    
    后期提升思路：
    1. 修改合适的卷积核大小
    2. 最后再加个隐藏层？
    
    """
    # (66,200,3)

    # Input
    inputs = Input(shape=(channels, height, width))
    
    # Normalize to [-0.5, 0.5]
    inputs_norm = Lambda(lambda x: (x-2.5)/5.0)(inputs)
    
    conv1 = Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', 
                   activation='relu')(inputs_norm)

    conv2 = Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', 
                   activation='relu')(conv1)

    conv3 = Conv2D(48, kernel_size=(5,5), strides=(2,2), padding='valid',
                   activation='relu')(conv2)

    conv4 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid',
                   activation='relu')(conv3)

    conv5 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid',
                   activation='relu')(conv4)

    flatten = Flatten()(conv5)
    
    x = Dropout(0.5)(flatten)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    steer = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=steer)

    return model

