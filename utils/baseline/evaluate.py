import os, sys
import csv
from keras.models import load_model
from keras import backend as K

def rmse(y_true, y_pred):
    """ Root Mean Squared Error 
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r_square(y_true, y_pred):
    """ R Square Score
    """
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/K.clip(SS_tot,K.epsilon(),None)

dependencies = {'rmse': rmse, 'val_rmse': rmse, 
                'r_square': r_square, 'val_r_square': r_square}

def evaluate(model_prefix, test_generator, test_step):
    '''
    Args:
    model_prefix: name prefix of the model
    test_generator: data generator for testing
    test_steps: steps per epoch for testing

    Return: 
    Metrics values on test dataset
    '''
    test_path = sys.path[0]

    model_path = os.path.join(test_path, 'h5/', model_prefix+'_best_model.h5')
    model = load_model(model_path, custom_objects=dependencies)
        
    test_loss, test_mse, test_rmse, test_mae, test_r2 = model.evaluate_generator(
                                                        test_generator,
                                                        steps=test_step)
    # save results to file
    test_log_path= os.path.join(test_path,'log/', model_prefix+'_test_log.csv')
    if not os.path.exists(os.path.split(test_log_path)[0]):
        os.makedirs(os.path.split(test_log_path)[0])
    
    # python3 use open()
    with open(test_log_path, 'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerow(['test_loss', 'test_mse', 'test_rmse', 'test_mae', 'test_r2'])
        csv_w.writerow([test_loss, test_mse, test_rmse, test_mae, test_r2])

    return (test_mse, test_rmse, test_mae, test_r2)