#! /usr/bin/env python3
import numpy as np
import math
from models import createModel
from keras import backend as K
import matplotlib.pyplot as plt
from dataset import load_dataset
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
np.random.seed(42)

def train(img_array, labels_array, class_weight):
    # SPLIT TRAIN AND TEST DATA
    train_images, test_images, train_labels, test_labels = train_test_split(img_array, labels_array)
    
    print("X_train shape: {}\nY_train shape: {}".format(train_images.shape, train_labels.shape))
    print("X_test shape: {}\nY_test shape: {}".format(test_images.shape, test_labels.shape))


    # NORMALIZATION
    train_images = train_images / 255
    test_images = test_images / 255

    # DATA AUGMENTATION
    aug = ImageDataGenerator(rotation_range=50, width_shift_range=0.5, 
            height_shift_range=0.66, shear_range=0.65, zoom_range=0.55,
                horizontal_flip=True, fill_mode="nearest")



    # INPUT SHAPE AND MODEL SETTINGS
    IMG_DIMS = (100, 200, 3)
    directory = "model-save/"
    filepath = directory + "2_val_loss_model.h5"
    model = createModel(IMG_DIMS)
    #model = load_model(filepath)
    BATCH_SIZE = 64
    EPOCHS = 6000

    # LEARNING RATE
    learning_rate = 0.001
    decay_rate = learning_rate / EPOCHS

    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,
                math.floor((1 + epoch) / epochs_drop))
        return lrate

    # MODEL COMPILE
    adam_opt = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = decay_rate)
    model.compile(optimizer = adam_opt, loss = 'binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    
    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True,
        mode='min')
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 500, verbose = 0,
        mode = 'min', restore_best_weights = True)
    lrate = LearningRateScheduler(step_decay)    
    callbacks_list = [checkpoint, early, lrate]
    

    # MODEL FIT WITH DATA AUG                   steps_per_epoch = len(X_train)//BATCH_SIZE
    history = model.fit_generator(
        aug.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        epochs=EPOCHS, verbose=1, validation_data=(test_images, test_labels),
        class_weight = class_weight, callbacks = callbacks_list)

    # LOAD BEST SAVED MODEL
    model = load_model(filepath)

    # MODEL EVALUATE
    scores = model.evaluate(test_images, test_labels)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

    '''
    preds = model.predict(test_images)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    '''
    # PREDICTION
    Y_pred_probabilities = model.predict(test_images)
    Y_pred = np.round(Y_pred_probabilities)
    Y_pred[Y_pred>=0.5] = 1
    Y_pred[Y_pred<0.5] = 0

    '''
    errors = []
    for i in range(len(Y_pred)):
        if Y_pred[i][1] != test_labels[i][1]:
            errors.append([i, Y_pred_probabilities[i][0]])
    sorted_errors = sorted(errors, key=lambda rec: rec[1], reverse=True)

    print("\nNumber of errors = ", len(sorted_errors))
    '''
    
    # PREDICT VS Y_TEST
    print("\nY pred         3: {}".format(Y_pred[3]))
    print("\nY test         3: {}".format(test_labels[3]))       
    print("\nY pred        15: {}".format(Y_pred[15]))
    print("\nY test        15: {}".format(test_labels[15])) 
    print("\nY predicted   55: {}".format(Y_pred[55]))
    print("\nY test        55: {}".format(test_labels[55]))
    print("\nY predicted  231: {}".format(Y_pred[231]))
    print("\nY test       231: {}".format(test_labels[231]))
    print("\nY predicted  456: {}".format(Y_pred[456]))
    print("\nY test       456: {}".format(test_labels[456]))
    print("\nY predicted  755: {}".format(Y_pred[755]))
    print("\nY test       755: {}".format(test_labels[755]))    
    print("\nY predicted  932: {}".format(Y_pred[932]))
    print("\nY test       932: {}".format(test_labels[932]))    
    
       
    # PLOT
    # Accuracy curve
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('accuracy_curve.png')
    #plt.show()

    # Loss curve
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss_curve.png')
    #plt.show()
    


dataset, labels, class_weight = load_dataset()

train(dataset, labels, class_weight)
