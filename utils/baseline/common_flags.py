'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2019-08-12 10:38:10
@LastEditTime: 2020-01-09 09:38:39
'''

import gflags

""" Usage: In command line, typing
    python train.py --csv_path=/media/... --length=1 ...
    Remark: '' should not be used when assgin value to the string type data 
"""

# instantiation
Flags = gflags.FLAGS

gflags.DEFINE_string('date_prefix', '2019_08_07_', 'date prefix')

# Loading Data
gflags.DEFINE_string('csv_path', '/gdata/wangshuai/sonarset/sim/track-I-II/2019-07-03-48.csv', 'dataset path')
gflags.DEFINE_integer('length', 1, 'time length')
gflags.DEFINE_integer('k_folds', 5, 'k fold cross validation')
gflags.DEFINE_integer('dim', 1, 'dimension of input data, 1, 2 or 3')
gflags.DEFINE_integer('target_height', 224, 'height of input image')
gflags.DEFINE_integer('target_width', 224, 'width of input image')
gflags.DEFINE_integer('batch_size', 128, 'batch size')

# choose model
gflags.DEFINE_string('model_type', 'mlp_models', 'Model type to be train') 
gflags.DEFINE_string('model_name', 'mlp', 'Model to be import')

# hidden neurons for mlp 
gflags.DEFINE_integer('initial_n_hidden', 64, 'Initials number of neurons')
gflags.DEFINE_integer('increase_n_hidden', 64, 'Increased number of neurons')

# Training
gflags.DEFINE_integer('epochs', 10, 'Maximum number of epochs for training')

# multiple gpus
gflags.DEFINE_integer('num_gpu', 4, "The number of gpus used in training")