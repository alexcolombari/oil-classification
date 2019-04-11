from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D

def createModel(IMG_DIMS):
    # Encoder
    model = Sequential()
    model.add(Conv2D(12, (3, 3), activation='relu', padding='same', input_shape=IMG_DIMS))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(24, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(24, (3, 3), activation='relu'))
    model.add(Dropout(0.6))

    model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='sigmoid'))

    return model
