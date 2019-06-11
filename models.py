import numpy as np
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, BatchNormalization
np.random.seed(7)

def model_1(input_img):
    conv1 = Conv2D(2, (3, 3), activation = 'relu', padding = 'same')(input_img)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(drop1)

    conv2 = Conv2D(6, (3, 3), activation = 'relu', padding = 'same')(pool1)
    drop2 = Dropout(0.65)(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(drop2)

    conv3 = Conv2D(6, (3, 3), activation = 'relu', padding = 'same')(pool2)
    drop3 = Dropout(0.6)(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(drop3)

    conv6 = Conv2D(6, (3, 3), activation = 'relu', padding = 'same')(pool3)
    drop6 = Dropout(0.35)(conv6)
    pool6 = MaxPooling2D(pool_size = (2, 2))(drop6)

    model_flatten = Flatten()(pool6)
    dense1 = Dense(256, activation = 'sigmoid')(model_flatten)
    drop5 = Dropout(0.35)(dense1)
    dense2 = Dense(13, activation = 'sigmoid')(drop5)

    return dense2

def model_2(input_img):
    x = Conv2D(6, (3, 3), activation = 'relu', padding = 'same')(input_img)
    x = Dropout(0.3)(x)
    x = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)

    x = Conv2D(6, (3, 3), activation = 'relu', padding = 'same', strides = (2, 2))(x)
    x = Dropout(0.3)(x)
    #x = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)

    x = Conv2D(6, (3, 3), activation = 'relu', padding = 'same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(6, (3, 3), activation = 'relu', padding = 'same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(6, (3, 3), activation = 'relu', padding = 'same')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(12, (3, 3), activation = 'relu', padding = 'same')(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(13, activation = 'sigmoid')(x)

    return output
