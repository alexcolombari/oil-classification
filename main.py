#! /usr/bin/env python3
import numpy as np
from models import createModel
from keras import backend as K
import matplotlib.pyplot as plt
from dataset import load_dataset
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


def train(img_array, labels_array):
    # Spliting train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(img_array, labels_array, random_state = 42)
    
    print("X_train shape: {}\nY_train shape: {}".format(X_train.shape, Y_train.shape))
    print("X_test shape: {}\nY_test shape: {}".format(X_test.shape, Y_test.shape))

    # Normalization
    X_train = X_train / 255
    X_test = X_test / 255

    '''
    # Class Weight
    class_weight_list = compute_class_weight('balanced', np.unique(Y_train), Y_train)
    class_weight = dict(zip(np.unique(Y_train), class_weight_list))
    Y_train = to_categorical(Y_train, num_classes = len(np.unique(Y_train)))
    '''

    # Data augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.15,
        height_shift_range=0.15, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    
    # INPUT SHAPE AND MODEL SETTINGS
    IMG_DIMS = (10, 10, 3)

    trainmodel = createModel(IMG_DIMS)
    BATCH_SIZE = 5
    EPOCHS = 50
    trainmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(trainmodel.summary())

    '''
    # CHECKPOINT
    directory = "model-save/"
    #filepath = directory + "acc-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = directory + "acc_model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
        mode='max')
    callbacks_list = [checkpoint]
    '''

    class_weight = { 0: 7.6923,
                     1: 7.6923,
                     2: 7.6923,
                     3: 7.6923,
                     4: 7.6923,
                     5: 7.6923,
                     6: 7.6923,
                     7: 7.6923,
                     8: 7.6923,
                     9: 7.6923,
                    10: 7.6923,
                    11: 7.6923,
                    12: 7.6923 }
    
    # MODEL FIT WITH DATA AUG
    history = trainmodel.fit_generator(
        aug.flow(X_train, Y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS, verbose=1, steps_per_epoch=len(X_train)//BATCH_SIZE,
        validation_data=(X_test, Y_test), class_weight = class_weight)
    
    scores = trainmodel.evaluate(X_test, Y_test)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

    preds = trainmodel.predict(X_test)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0


    print("\nY predicted 1: {}".format(preds[1]))
    print("\nY test 1: {}".format(Y_test[1]))       
    print("\nY predicted 15: {}".format(preds[15]))
    print("\nY predicted 27: {}".format(preds[27]))
    print("\nY predicted 55: {}".format(preds[55]))
    print("\nY predicted 13: {}".format(preds[13]))


'''
    # PLOT
    # Accuracy curve
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Loss curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
'''

dataset, labels = load_dataset()
train(dataset, labels)
