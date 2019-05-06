import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, BatchNormalization
np.random.seed(7)

def createModel(IMG_DIMS):
    model = Sequential()
    model.add(Conv2D(6, (3, 3), activation = 'relu', padding = 'same', input_shape = IMG_DIMS))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(6, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(12, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(12, (3, 3), activation = 'relu', padding = 'same'))   
    model.add(Dropout(0.3))

    model.add(Conv2D(12, (3, 3), activation = 'relu', padding = 'same'))   
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(13, activation = 'sigmoid'))

    return model

def encoder_model(input_img):
    # ENCODER
    #input = 100 x 200 x 1 (wide and thin)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # 100 x 200 x 16
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 10 x 200 x 16
    
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1) # 50 x 100 x 8
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 50 x 100 x 8
    
    conv3 = Conv2D(4, (3, 3), activation='relu', padding='same')(pool2) # 25 x 50 x 4 (small and thick)

    # DECODER
    conv4 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv3) # 25 x 50 x 4
    up1 = UpSampling2D((2,2))(conv4) # 25 x 50 x 128
    
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1) # 50 x 100 x 8
    up2 = UpSampling2D((2,2))(conv5) # 50 x 100 x 64
    
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(up2) # 100 x 200 x 16
    
    return decoded
