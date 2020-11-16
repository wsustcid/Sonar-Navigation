'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2020-01-10 16:36:03
@LastEditTime: 2020-01-10 16:37:28
'''


from keras import backend as K

def rmse(y_true, y_pred):
    """ Root Mean Squared Error 
    """
    return K.sqrt(K.mean(K.square(y_true-y_pred)))

def r_square(y_true, y_pred):
    """ R Square Score
    """
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    
    return 1 - SS_res/K.clip(SS_tot,K.epsilon(),None)