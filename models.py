import numpy as np
from keras import regularizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, BatchNormalization
np.random.seed(7)

def model_1(input_img):
    input_layer = Conv2D(2, (3, 3), activation = 'relu', padding = 'same')(input_img)
    x = Dropout(0.3)(input_layer)
    x = Conv2D(4, (3, 3), activation = 'relu', padding = 'same')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(4, (3, 3), activation = 'relu', padding = 'same')(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.45)(x)
    output = Dense(13, activation = 'sigmoid')(x)

    return output
