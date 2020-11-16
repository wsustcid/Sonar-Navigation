from keras.models import Model 
from keras.layers import Input, Dense, Dropout

def mlp(input_dim, n_hidden):
    """ Define a mlp model with one hidden layer
    Args:
    - input_dim: dimensions of input tensor
    - n_hidden: number of hidden layer neurons

    return:
    - A model instance
    """

    # Input
    x = Input(shape=(input_dim, ))

    # hidden layer
    h1 = Dense(n_hidden, activation='relu')(x)
    h1 = Dropout(0.5)(h1)

    # output layer
    y = Dense(1)(h1)
        
    # model
    model = Model(inputs=x, outputs=y)

    return model