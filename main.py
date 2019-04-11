#! /usr/bin/env python3
import numpy as np
#from statistics import median
from models import createModel
from keras import backend as K
import matplotlib.pyplot as plt
from dataset import load_dataset
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(42)

def train(img_array, labels_array, class_weights):
    # Spliting train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(img_array, labels_array)
    
    print("X_train shape: {}\nY_train shape: {}".format(X_train.shape, Y_train.shape))
    print("X_test shape: {}\nY_test shape: {}".format(X_test.shape, Y_test.shape))


    # NORMALIZATION
    X_train = X_train / 255
    X_test = X_test / 255

    # DATA AUGMENTATION
    aug = ImageDataGenerator(rotation_range=70, width_shift_range=0.62, 
            height_shift_range=0.66, shear_range=0.65, zoom_range=0.55,
                horizontal_flip=True, fill_mode="nearest")


    # INPUT SHAPE AND MODEL SETTINGS
    IMG_DIMS = (10, 10, 3)
    model = createModel(IMG_DIMS)
    BATCH_SIZE = 5
    EPOCHS = 30
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    
    # CHECKPOINT
    directory = "model-save/"
    #filepath = directory + "acc-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = directory + "val_loss_model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
        mode='min')
    callbacks_list = [checkpoint]
    

    # MODEL FIT WITH DATA AUG                   steps_per_epoch = len(X_train)//BATCH_SIZE
    history = model.fit_generator(
        aug.flow(X_train, Y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS, verbose=1, steps_per_epoch=2,  
        validation_data=(X_test, Y_test), class_weight = class_weights)


    scores = model.evaluate(X_test, Y_test)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

    preds = model.predict(X_test)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0


    # PREDICT VS Y_TEST
    print("\nY pred         1: {}".format(preds[1]))
    print("\nY test         1: {}".format(Y_test[1]))       
    print("\nY pred        15: {}".format(preds[15]))
    print("\nY test        15: {}".format(Y_test[15])) 
    print("\nY predicted   55: {}".format(preds[55]))
    print("\nY test        55: {}".format(Y_test[55]))
    #print("\nY predicted  231: {}".format(preds[231]))
    #print("\nY test       231: {}".format(Y_test[231]))
    #print("\nY predicted  456: {}".format(preds[456]))
    #print("\nY test       456: {}".format(Y_test[456]))
    

    '''    
    # PLOT
    # Accuracy curve
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.savefig('accuracy_curve.png')
    plt.show()

    # Loss curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.savefig('loss_curve.png')
    plt.show()
    '''


dataset, labels, class_weights = load_dataset()

train(dataset, labels, class_weights)
