import numpy as np

from keras.models import Sequential
from keras.constraints import max_norm
from keras import regularizers, initializers
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

np.random.seed(7)

def cnn(input_img):
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape = input_img, activation = 'relu', padding = 'same', kernel_regularizer = regularizers.l2(1e-7)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(13, (2, 2), activation = 'relu', padding = 'same', kernel_regularizer = regularizers.l2(1e-7)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(13, activation = 'sigmoid', kernel_regularizer = regularizers.l2(1e-7)))

    return model
