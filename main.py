#! /usr/bin/env python3
import os
import math
import numpy as np
from models import *
from metrics import *
import matplotlib.pyplot as plt
from dataset import load_dataset
from keras.optimizers import Adam, Adadelta
from keras.utils import plot_model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
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
            height_shift_range=0.5, shear_range=0.60, zoom_range=0.55,
                horizontal_flip=True, fill_mode="nearest")

    # INPUT SHAPE AND MODEL SETTINGS
    IMG_DIMS = (100, 200, 3)
    directory = "model-save/"
    filepath = directory + "5_val_loss_model.hdf5"
    #model = createModel_4(IMG_DIMS)
    model = load_model(filepath)
    BATCH_SIZE = 128
    EPOCHS = 2

    '''
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
    '''

    # MODEL COMPILE
    #adam_opt = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = decay_rate)
    #adadelta_opt = Adadelta(lr = 1.0, rho = 0.95, epsilon = None, decay = 0.0)
    
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes = True)

    '''
    # CLEAR TENSOBOARD DIRECTORY LOG FILES
    mydir = "./logs/"
    filelist = [ f for f in os.listdir(mydir) if f.endswith(".AIserver") ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))
    '''

    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True,
        mode='min')
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 950, verbose = 1,
        mode = 'min', restore_best_weights = True)
    #lrate = LearningRateScheduler(step_decay)
    tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 2, batch_size = BATCH_SIZE,
        write_graph = False, write_images = False, embeddings_layer_names = None, update_freq = 'epoch')    
    callbacks_list = [checkpoint, tensor_board]
    

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

    
    # PREDICT VS Y_TEST
    print("\nY pred          3: {}".format(Y_pred[3]))
    print("\nY test          3: {}".format(test_labels[3]))       
    print("\nY pred         26: {}".format(Y_pred[26]))
    print("\nY test         26: {}".format(test_labels[26])) 
    print("\nY predicted    55: {}".format(Y_pred[55]))
    print("\nY test         55: {}".format(test_labels[55]))
    print("\nY predicted   231: {}".format(Y_pred[231]))
    print("\nY test        231: {}".format(test_labels[231]))
    print("\nY predicted   456: {}".format(Y_pred[456]))
    print("\nY test        456: {}".format(test_labels[456]))
    print("\nY predicted   755: {}".format(Y_pred[755]))
    print("\nY test        755: {}".format(test_labels[755]))    
    print("\nY predicted   932: {}".format(Y_pred[932]))
    print("\nY test        932: {}".format(test_labels[932]))
    print("\nY predicted  1055: {}".format(Y_pred[1055]))
    print("\nY test       1055: {}".format(test_labels[1055]))
    print("\nY predicted  1188: {}".format(Y_pred[1188]))
    print("\nY test       1188: {}".format(test_labels[1188]))                     
    
       
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
    

if __name__ == "__main__":
    dataset, labels, class_weight = load_dataset()
    train(dataset, labels, class_weight)
