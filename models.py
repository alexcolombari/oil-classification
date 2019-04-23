import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D
np.random.seed(42)

def createModel(IMG_DIMS):
    # Encoder
    model = Sequential()
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same', input_shape = IMG_DIMS))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(6, (5, 5), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(13, activation='sigmoid'))

    return model
