import numpy as np

from keras.models import Sequential
from keras.constraints import max_norm
from keras import regularizers, initializers
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

np.random.seed(14)


def cnn_2convs(input_img):
    model = Sequential()
    model.add(Conv2D(4, (3, 3), strides = 1,
                    input_shape = input_img, activation = 'relu',
                    padding = 'same', kernel_regularizer = regularizers.l2(1e-3)))
    model.add(MaxPool2D(pool_size = (2, 2))) 
    model.add(Dropout(0.3))
    model.add(Conv2D(8, (3, 3), strides = 1, activation = 'relu',
                    padding = 'same', kernel_regularizer = regularizers.l2(1e-3)))
    model.add(MaxPool2D(pool_size = (2, 2))) 
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(26, activation = 'relu', bias_initializer = initializers.Constant(0.1),
                    kernel_regularizer = regularizers.l2(1e-3)))
    model.add(Dropout(0.3))
    model.add(Dense(13, activation = 'sigmoid'))

    return model

def cnn_3convs(input_img):
    model = Sequential()
    model.add(Conv2D(4, (3, 3), strides = 1,
                    input_shape = input_img, activation = 'relu',
                    padding = 'same', kernel_regularizer = regularizers.l2(1e-3)))
    model.add(MaxPool2D(pool_size = (2, 2))) 
    model.add(Dropout(0.3))
    model.add(Conv2D(8, (3, 3), strides = 1, activation = 'relu',
                    padding = 'same', kernel_regularizer = regularizers.l2(1e-3)))
    model.add(MaxPool2D(pool_size = (2, 2))) 
    model.add(Dropout(0.3))
    model.add(Conv2D(16, (3, 3), strides = 2, activation = 'relu',
                    padding = 'same', kernel_regularizer = regularizers.l2(1e-3)))
    model.add(MaxPool2D(pool_size = (2, 2))) 
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(26, activation = 'relu', bias_initializer = initializers.Constant(0.1),
                    kernel_regularizer = regularizers.l2(1e-3)))
    model.add(Dropout(0.3))
    model.add(Dense(13, activation = 'sigmoid'))

    return model
