import numpy as np
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, BatchNormalization

def model_1(input_img):
    conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(input_img)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    drop1 = Dropout(0.39)(pool1)

    conv2 = Conv2D(2, (3, 3), activation = 'relu', padding = 'same')(drop1)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
    drop2 = Dropout(0.49)(pool2)

    model_flatten = Flatten()(drop2)
    dense1 = Dense(128, activation = 'relu')(model_flatten) 
    dense2 = Dense(13, activation = 'sigmoid')(dense1)

    return dense2
