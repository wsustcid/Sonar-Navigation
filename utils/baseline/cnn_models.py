
""" The models defined in this file
1. PilotNet
2. DroNet
3. VGG16
4. DenseNet
"""

from keras.models import Model
from keras.layers import Input, Flatten, Dense
# modules used in pilotnet
from keras.layers import Lambda, Conv2D

# modules used in dronet
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.merge import add

# modules used in densenet
from keras.layers import Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Concatenate


from keras import backend as K

# set channels_first for all models
K.set_image_data_format("channels_first")



def pilotnet(channels, height, width):
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

    fc1 = Dense(100, activation='relu')(flatten)

    fc2 = Dense(50, activation='relu')(fc1)

    fc3 = Dense(10, activation='relu')(fc2)

    steer = Dense(1)(fc3)
    
    model = Model(inputs=inputs, outputs=steer)

    return model


def dronet(channels, height, width):
    # (200,200,1)

    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3

    # Input
    inputs = Input(shape=(channels, height, width))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(inputs)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x2 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x4 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x6)

    x6 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    steer = Dense(1)(x)

    # model
    model = Model(inputs=inputs, outputs=steer)
    
    return model




def vgg16(channels, height, width):
    # (224,224,3)

    inputs = Input(shape=(channels, height, width))

    # bolck 1
    x1    = Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x2    = Conv2D(64, (3,3), padding='same', activation='relu')(x1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(x2)
    
    # bolck 2
    x3    = Conv2D(128, (3,3), padding='same', activation='relu')(pool1)
    x4    = Conv2D(128, (3,3), padding='same', activation='relu')(x3)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(x4)

    # bolck 3
    x5    = Conv2D(256, (3,3), padding='same', activation='relu')(pool2)
    x6    = Conv2D(256, (3,3), padding='same', activation='relu')(x5)
    x7    = Conv2D(256, (3,3), padding='same', activation='relu')(x6)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(x7)

    # bolck 4
    x8    = Conv2D(512, (3,3), padding='same', activation='relu')(pool3)
    x9    = Conv2D(512, (3,3), padding='same', activation='relu')(x8)
    x10   = Conv2D(512, (3,3), padding='same', activation='relu')(x9)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(x10)

    # bolck 5
    x11 = Conv2D(512, (3,3), padding='same', activation='relu')(pool4)
    x12 = Conv2D(512, (3,3), padding='same', activation='relu')(x11)
    x13 = Conv2D(512, (3,3), padding='same', activation='relu')(x12)
    pool5 = MaxPooling2D((2,2), strides=(2,2))(x13)

    # flatten
    flatten = Flatten()(pool5)

    # fc
    x14 = Dense(4096, activation='relu')(flatten)
    x15 = Dense(4096, activation='relu')(x14)
    
    # output
    steer = Dense(1)(x15)

    model = Model(inputs=inputs, outputs=steer)

    return model


## DenseNet

def conv_block(x, growth_rate, name):
    """ A building block for a dense block
    Args:
        x: input tensor
        growth_rate: int, number of filters in 3x3 conv
        name: string, conv block label

    Return:
        the concatenated output tensor of one conv block
    """

    concat_axis = 1 if K.image_data_format() == 'channels_first' else 3
        
    # bottlenet block
    x1 = BatchNormalization(axis=concat_axis, 
                            epsilon=1.001e-5, 
                            name=name+'_0_bn')(x)
    x1 = Activation('relu', name=name+'_0_relu')(x1)
    x1 = Conv2D(4*growth_rate, (1,1), padding='same', 
                kernel_initializer='he_normal', 
                kernel_regularizer=l2(1e-4),
                use_bias=False, 
                name=name+'_0_conv')(x1)
        
    x1 = BatchNormalization(axis=concat_axis, 
                            epsilon=1.001e-5, 
                            name=name+'_1_bn')(x1)
    x1 = Activation('relu', name=name+'_1_relu')(x1)
    x1 = Conv2D(growth_rate, (3,3), padding='same', 
                kernel_initializer='he_normal', 
                kernel_regularizer=l2(1e-4),
                use_bias=False, 
                name=name+'_1_conv')(x1)

    # concatate layers
    x = Concatenate(axis=concat_axis, name=name+'_concat')([x, x1])

    return x

def dense_block(x, growth_rate, n_conv_blocks, name):
    """ A dense block
    Args:
        x: input tensor of a dense block
        growth_rate: number of filters in 3x3 conv
        n_conv_blocks: number of conv blocks in one dense block
        name: dense block label
    Return:
        output tensor of a dense block
    """

    for i in range(n_conv_blocks):
        x = conv_block(x, growth_rate, name=name+'_'+str(i+1))
    
    return x

def transition_block(x, reduction, name):
    """ A transition block for reduce dimensions of channel
    Args:
        x: input tensor
        reduction: float, compression rate at transition layers
        name: string, transition block label

    Return:
        output tensor
    """

    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        
    x = BatchNormalization(axis=bn_axis, 
                           epsilon=1.001e-5, 
                           name=name+'_bn')(x)
    x = Activation('relu', name=name+'_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis]*reduction), (1,1), padding='same',
               kernel_initializer='he_normal', 
               kernel_regularizer=l2(1e-4),
               use_bias=False, 
               name=name+'_conv')(x)
    x = AveragePooling2D((2,2), strides=(2,2), name=name+'_pool')(x)

    return x

def DenseNet(blocks, channels, height, width):
    """ Instantiates the original DenseNet architecture
        - include first 7x7 conv block and 3x3 max pool
        - 4 dense blocks and 3 transition blocks
        - the softmax layer was replaced with a fc layer which outputs 1 prediction
    Args:
        block: list, number of conv blocks in 4 dense blocks respectively.
        input_shape: shape of input image

    Return:
        a model of DenseNet
    """

    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3

    img_input = Input(shape=(channels, height, width))

    # first conv and pool
    x = ZeroPadding2D(padding=((3,3),(3,3)))(img_input)
    x = Conv2D(64, (7,7), strides=(2,2),
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4),
               use_bias = False,
               name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = ZeroPadding2D(padding=((1,1),(1,1)))(x)
    x = MaxPooling2D((3,3), strides=(2,2), name='pool1')(x)

    x = dense_block(x, growth_rate=32, n_conv_blocks=blocks[0], name='block1')
    x = transition_block(x, reduction=0.5, name='trans1')

    x = dense_block(x, growth_rate=32, n_conv_blocks=blocks[1], name='block2')
    x = transition_block(x, reduction=0.5, name='trans2')

    x = dense_block(x, growth_rate=32, n_conv_blocks=blocks[2], name='block3')
    x = transition_block(x, reduction=0.5, name='trans3')

    x = dense_block(x, growth_rate=32, n_conv_blocks=blocks[3], name='block4')
    
    # 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    
    steer = Dense(1, name='pred')(x)

    model = Model(inputs=img_input, outputs=steer, name='densenet')

    return model


def DenseNet121(channels, height, width):

    return DenseNet(blocks=[6, 12, 24, 16],
                    channels=channels,
                    height=height,
                    width=width)
    

    



    

        

