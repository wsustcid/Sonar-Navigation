'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2019-08-12 10:38:23
@LastEditTime: 2019-08-12 10:38:24
'''


from datetime import datetime
import numpy as np
import os, sys
import csv
import gflags

from data_processing import csv_read, data_generator
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

from mlp_models import mlp
from cnn_models import pilotnet, dronet, vgg16, DenseNet121

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from evaluate import rmse, r_square, evaluate

from common_flags import Flags



def getModel(name, input_dim, n_hidden, channels, height, width):
    ''' Initialize model
    Args:
    name: the model name to be initialized
    input_dim: dims of input tensor for mlp model
    n_hidden: number of hidden layer neurons for mlp model
    channels, height, width: the shape of input tensor for cnn models 

    Return:
    a model instance
    '''
    # ensure we have corresponding model
    assert name in ['mlp', 'pilotnet', 'dronet', 'vgg16', 'DenseNet121']

    if name == 'mlp':
        model = mlp(input_dim, n_hidden)
    elif name == 'pilotnet':
        model = pilotnet(channels, height, width)
    elif name == 'dronet':
        model = dronet(channels, height, width)
    elif name == 'vgg16':
        model = vgg16(channels, height, width)
    elif name == 'DenseNet121':
        model = DenseNet121(channels, height, width)
    else:
        pass
    
    return model


def getGenerator(csv_path, length, k_folds, dim,
                 target_height, target_width, batch_size):
    """ Reading data from csv file and constructing data generators
    Args:
    csv_path: path of csv file
    length: length of sequence data to be read
    k_folds: number of folds for training and validation
    dim: the dimension of samples used for training or validation. Selectable
         value is 1, 2 or 3.
    target_height, target_width: target height and width of input samples
    batch_size: the size of data batch

    Retrun:
    two lists which save train_set and valid_set respectively
    two lists which save train_steps_per_epoch and valid_steps_per_epoch
    
    Remark:
    - random_state was set at every shuffle operaton to ensure that 
      all the models use the same training set and validation set.
    """
    # loading data
    dataset = csv_read(csv_path, length)
    dataset = shuffle(dataset, random_state=0)

    train_data, test_data = train_test_split(dataset,
                                           test_size=0.2, 
                                           shuffle=True,
                                           random_state=10)
    
    test_generator = data_generator(Data=test_data, 
                                    dim=dim, 
                                    target_height=target_height, 
                                    target_width=target_width, 
                                    batch_size=len(test_data),
                                    random_state=20)
    
    train_steps, valid_steps = [], []
    train_generators, valid_generators = [], []
    # get data generator
    if k_folds==1:
        train_set, valid_set = train_test_split(train_data,
                                                test_size=0.2, 
                                                shuffle=True,
                                                random_state=30)
        train_steps_per_epoch = len(train_set)//batch_size
        valid_steps_per_epoch = len(valid_set)//batch_size

        train_steps.append(train_steps_per_epoch)
        valid_steps.append(valid_steps_per_epoch)
        
        train_generator = data_generator(Data=train_set, 
                                         dim=dim, 
                                         target_height=target_height, 
                                         target_width=target_width, 
                                         batch_size=batch_size)
                                         
        valid_generator = data_generator(Data=valid_set, 
                                         dim=dim, 
                                         target_height=target_height, 
                                         target_width=target_width, 
                                         batch_size=batch_size)
        train_generators.append(train_generator)
        valid_generators.append(valid_generator)

    elif k_folds >=2:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=40)
        
        for train_index, valid_index in kfold.split(dataset):
            train_set = np.array(dataset)[train_index]
            valid_set = np.array(dataset)[valid_index]

            train_steps_per_epoch = len(train_set)//batch_size
            valid_steps_per_epoch = len(valid_set)//batch_size

            train_steps.append(train_steps_per_epoch)
            valid_steps.append(valid_steps_per_epoch)
        
            train_generator = data_generator(Data=train_set, 
                                             dim=dim, 
                                             target_height=target_height, 
                                             target_width=target_width, 
                                             batch_size=batch_size)
                                         
            valid_generator = data_generator(Data=valid_set, 
                                             dim=dim, 
                                             target_height=target_height, 
                                             target_width=target_width, 
                                             batch_size=batch_size)

            # I have verified that the object of data_generator is an immutable object
            train_generators.append(train_generator)
            valid_generators.append(valid_generator)
    else:
        raise IOError("Undefined k folds value!")
            
    return (train_generators, train_steps, valid_generators, valid_steps, test_generator)


def trainModel(model, train_generator, train_step,
               valid_generator, valid_step, epochs, model_prefix):
    """ training model and saving best model
    Args:
      model: a model instance
      train/valid_generator: data generator for training/validation
      train/valid_step: steps per epoch while training or validation
      epochs: training epochs

    """
    # compile model
    optimizer = optimizers.Adam(decay=1e-5)
    model.compile(loss='mse', optimizer=optimizer,
                  metrics=['mse', rmse, 'mae', r_square])
    
    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', 
                                   patience=200, min_delta=0, verbose=1)
    
    train_path = sys.path[0]
    model_path = os.path.join(train_path,'h5/', model_prefix+'_best_model.h5')
    if not os.path.exists(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0])
    
    model_checkpoint = ModelCheckpoint(model_path,
                                       monitor="val_r_square",
                                       mode='max',
                                       save_best_only=True,
                                       verbose=1)
                                       
    train_log_path= os.path.join(train_path,'log/', model_prefix+'_train_log.csv')
    if not os.path.exists(os.path.split(train_log_path)[0]):
        os.makedirs(os.path.split(train_log_path)[0])
    
    csv_logger = CSVLogger(train_log_path)

    # 
    callbacks = [early_stopping, model_checkpoint, csv_logger]
    
    # fit model
    History = model.fit_generator(generator=train_generator,
                                  steps_per_epoch = train_step,
                                  epochs = epochs,
                                  validation_data = valid_generator,
                                  validation_steps = valid_step,
                                  verbose = 1,
                                  callbacks=callbacks)
  


def main(n_hidden=None):
    """ train models with k fold cross validation
    """
    prefix = date_prefix+Flags.model_name +'_'+str(Flags.length)
    if Flags.model_type == 'mlp_models':
        model_prefix = prefix+'_'+str(n_hidden)

    elif Flags.model_type == 'cnn_models':
        model_prefix = prefix
    else:
        pass
    # get data generators 
    generators = getGenerator(csv_path = Flags.csv_path, 
                              length   = Flags.length,
                              k_folds  = Flags.k_folds, 
                              dim = Flags.dim, 
                              target_height = Flags.target_height,
                              target_width  = Flags.target_width,
                              batch_size    = Flags.batch_size)
    train_generators, train_steps, valid_generators, valid_steps, test_generator = generators
    
    # k fold training
    count =0
    mse_fold, rmse_fold, mae_fold, r2_fold = [], [], [], []
    for train_generator,train_step,valid_generator,valid_step in zip(train_generators,
                                                                     train_steps,
                                                                     valid_generators,
                                                                     valid_steps):
        if Flags.dim == 1:
            input_dim = Flags.length*16
            channels = None
        elif Flags.dim == 2:
            input_dim = None
            channels = 1
        elif Flags.dim == 3:
            input_dim = None
            channels = 3
        else:
            raise IOError('Wong dim value!')

        # get model
        model = getModel(name=Flags.model_name, input_dim=input_dim,
                         n_hidden=n_hidden, channels=channels, 
                         height=Flags.target_height,
                         width=Flags.target_width)
        count +=1
        if count==1:
            model.summary()
        
        
        # train
        trainModel(model=model, train_generator=train_generator, 
                   train_step=train_step, valid_generator=valid_generator, 
                   valid_step=valid_step, epochs=Flags.epochs, 
                   model_prefix=model_prefix+'_'+str(count))

        
        val_results = evaluate(model_prefix=model_prefix+'_'+str(count), 
                               test_generator=test_generator,
                               test_step=1)
        val_mse, val_rmse, val_mae, val_r2 = val_results

        #save results to lists
        mse_fold.append(val_mse)
        rmse_fold.append(val_rmse)
        mae_fold.append(val_mae)
        r2_fold.append(val_r2)
    
    # average
    mse_ave  = np.mean(mse_fold)
    rmse_ave = np.mean(rmse_fold)
    mae_ave  = np.mean(mae_fold)
    r2_ave   = np.mean(r2_fold)
    
    # save all the results with various length value to one file
    kfold_log_path = os.path.join(sys.path[0],'log/', prefix+'_kfold_log.csv')
    if not os.path.exists(os.path.split(kfold_log_path)[0]):
        os.makedirs(os.path.split(kfold_log_path)[0])
    with open(kfold_log_path, 'a') as f:
        csv_w = csv.writer(f)
        #csv_w.writerow(['mse_ave', 'rmse_ave', 'mae_ave', 'r2_ave'])
        csv_w.writerow([mse_ave, rmse_ave, mae_ave, r2_ave])

    return r2_ave




def _main():
    """ Training of mlp models
    - number of hidden layer neurons will be increased until
      the value of r2 score stop increasing. 
    """

    r2_best = -100
    for i in range(50):
        n_hidden = Flags.initial_n_hidden + i*Flags.increase_n_hidden

        # train model with specific number of hidden neurons
        r2_ave = main(n_hidden)
        
        if r2_ave > r2_best:
            r2_best = r2_ave
        else:
            # stop increase n_hidden
            break
    

if __name__ == "__main__":
    try:
        # parse all arguments
        Flags(sys.argv)
    except:
        # use --help to show all flags
        print("Usage: {} Args \n {}".format(sys.argv[0], Flags))
        sys.exit(1)

    date_prefix = Flags.date_prefix
    
    if Flags.model_type == 'mlp_models':
        _main()
    
    elif Flags.model_type == 'cnn_models':
        main()
    else:
        # for test other model types in future
        pass