from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D

def createModel(IMG_DIMS):
    # Encoder
    #input = 28 x 28 x 1 (wide and thin)
    model = Sequential()
    model.add(Conv2D(2, (3, 3), activation='relu', padding='same', input_shape=IMG_DIMS))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(2, (3, 3), activation='relu')) #strides=(2, 2)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(4, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))    

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='sigmoid'))

    return model
   
'''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape = IMG_DIMS))
    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='sigmoid'))

    return model
    '''
