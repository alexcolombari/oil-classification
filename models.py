import numpy as np
from keras.constraints import max_norm
from keras import regularizers, initializers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, BatchNormalization
np.random.seed(7)

def cnn(input_img):
    input_layer = Conv2D(2, (3, 3), activation = 'relu', padding = 'same',
                        bias_initializer=initializers.RandomNormal(0.1),
                        kernel_constraint=max_norm(2.))(input_img)
    x = Dropout(0.3)(input_layer)

    x = Flatten()(x)
    output = Dense(13, activation = 'sigmoid', bias_initializer=initializers.RandomNormal(0.1),
                    kernel_constraint=max_norm(2.))(x)

    return output
