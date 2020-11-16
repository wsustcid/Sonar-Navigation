# -*- coding: utf-8 -*-
'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2020-01-10 10:08:49
@LastEditTime: 2020-01-17 11:05:37
'''

"""
V1: 见version1.0
V2: 采用modified PilotNet, 验证时空信息重要性；建立baseline

"""


import os 

from data_gen import read_data
from models.mpilotnet import mpilotnet
from evaluate import rmse, r_square

from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K

K.set_image_data_format('channels_first')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()   
config.gpu_options.allow_growth = True      
set_session(tf.Session(config=config))


def train_model(FLAGS):
    
    # data 
    X_train, Y_train, X_valid, Y_valid = read_data(csv_path=FLAGS.data_path, 
                                                   length=FLAGS.length, 
                                                   dim=FLAGS.dim, 
                                                   target_height=FLAGS.height,
                                                   target_width=FLAGS.width)


    # model
    model = mpilotnet(FLAGS.channels, FLAGS.height, FLAGS.width)
    
    if FLAGS.optimizer == 'adam':
      optimizer = Adam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'sgd':
      optimizer = SGD(lr=FLAGS.learning_rate) 
    else:
      print("optimizer error!")
    print("{} - Learning rate: {}".format(FLAGS.optimizer, FLAGS.learning_rate))

    #eval.w1 = FLAGS.steer_weight
    #eval.w2 = FLAGS.speed_weight
    #print("Steer Trainable: {}, Speed Trainable: {}".format(eval.w1, eval.w2))

    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=[rmse, 'mae', r_square])
    model.summary()

    # Restart training from previous best weights by hands.
    # rdlr will continue to train from bad results.
    if FLAGS.restore_weights_path !='' and os.path.exists(FLAGS.restore_weights_path):
      model.load_weights(FLAGS.restore_weights_path)
      print("Training model from {}".format(FLAGS.restore_weights_path))
    else:
      print("Training model from scrath")
    
    # callbacks
    es = EarlyStopping(monitor='val_r_square',
                       mode='max',
                       patience=10,
                       verbose=1)
    
    #rdlr = ReduceLROnPlateau(monitor='val_mean_squared_error',
    #                         factor=0.7,
    #                         patience=10,
    #                         verbose=1,
    #                         mode='min')
    
    best_weights_path = os.path.join(FLAGS.output_path, 
                                     'weights/weights_{epoch:02d}_{val_r_square:.4f}.h5')
    if not os.path.exists(os.path.split(best_weights_path)[0]):
        os.makedirs(os.path.split(best_weights_path)[0])
    mc = ModelCheckpoint(best_weights_path,
                         monitor='val_r_square',
                         verbose=1,
                         save_best_only=True,
                         mode='max',
                         save_weights_only=True) 

    train_log_path = os.path.join(FLAGS.output_path, 'log/train_log.csv')
    if not os.path.exists(os.path.split(train_log_path)[0]):
        os.makedirs(os.path.split(train_log_path)[0])
    logger = CSVLogger(train_log_path)
    
    # fit
    model.fit(X_train, Y_train,
              batch_size = FLAGS.batch_size,
              epochs=FLAGS.max_epochs,
              verbose=1,
              callbacks=[es, mc, logger],
              validation_data=(X_valid, Y_valid))

    print("Training is done!")