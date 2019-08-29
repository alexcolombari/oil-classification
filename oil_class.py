import sys
import time
import random

from pycm import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split

import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
from keras.constraints import max_norm
from keras import metrics, regularizers, initializers
from keras.models import load_model, Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten

seed = 7
np.random.seed(seed)

data_folder = "/opt/data_repository/oil_samples/"
file_to_open = data_folder + "laminas.pkl"
#samples = 4999    # 7799
df = pd.read_pickle(file_to_open)

def cnn(input_img):
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape = input_img, activation = 'relu', padding = 'same', kernel_regularizer = regularizers.l2(1e-7)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(13, (2, 2), activation = 'relu', padding = 'same', kernel_regularizer = regularizers.l2(1e-7)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(13, activation = 'sigmoid', kernel_regularizer = regularizers.l2(1e-7)))

    return model

def loadFilesTrain(df):
    images = df.loc[: , "lamina"]
    labels = df.loc[: , "classificacao"]

    '''
    # -------------------- CROSS VALIDATION --------------------
    test = df.loc[7499:8799 , :]
    images_test = test.loc[: , "lamina"]
    labels_test = test.loc[: , "classificacao"]

    X = images
    Y = labels
    k_folds = 10
    kfold = KFold(n_splits = k_folds, shuffle = True, random_state = seed)
    kfold.get_n_splits(images)
    for train_index, test_index in kfold.split(X):
        trainData = X[train_index]
        testData = X[test_index]
        trainLabels = Y[train_index]
        testLabels = Y[test_index]
    '''

    img2array = []
    labels2array = []
    for i in range(len(images)):
        # IMAGE ARRAY
        imgarr = np.array(images[i])
        img_resize = np.resize(imgarr, (75, 100, 3))
        img2array.append(img_resize)

        # LABEL ARRAY
        labelsarr = np.array(labels[i])
        labels2array.append(labelsarr)

    img_array = np.asarray(img2array)
    labels_array = np.asarray(labels2array)

    img_array = img_array / 255.

    return img_array, labels_array

def main(epoch, DEBUG):
    # -------------------- TRAINING --------------------
    if DEBUG == 1:
    # -------------------- CLASS WEIGHT --------------------
        def class_weight():
            weights = np.zeros(13)
            for idx,row in df.iterrows():
                weights += np.array(row['classificacao'])
            weights /= len(df)
            class_weight = dict(enumerate(weights))
            return class_weight

        # -------------------- GET LAYER WEIGHT --------------------
        def get_weights():
            for layer in model.layers: print(layer.get_config(), layer.get_weights())

        img_array, labels_array = loadFilesTrain(df)
        trainData, testData, trainLabels, testLabels = train_test_split(img_array, labels_array, test_size = 0.25, random_state = None)


        print("[INFO] trainData shape: {}\n       testData shape: {}\n\n\
        trainLabels shape: {}\n       testLabels shape: {}\n".format(trainData.shape, testData.shape,
            trainLabels.shape, testLabels.shape))

        time.sleep(3)

        '''
        # Autoencoder
        # AUTOENCODER MODEL LOAD
        autoencoder_directory = "model-save/"
        autoencoder_path = autoencoder_directory + "3_auto_encoder_model.hdf5"
        autoencoder_model = load_model(autoencoder_path)

        # GET AUTOENCODED MODEL LAYER
        encoder = Model(autoencoder_model.input, autoencoder_model.layers[-6].output)
        print("\n\n[INFO] Autoencoder model summary")
        print(encoder.summary())
        print("\n[INFO] Starting autoencoder prediction!")
        time.sleep(2)

        # Normalization
        x_train_predict = encoder.predict(trainDataArray)
        x_train_predict = x_train_predict.astype('uint8') / 255.

        x_test_predict = encoder.predict(testDataArray)
        x_test_predict = x_test_predict.astype('uint8') / 255.

        print("\n[INFO] Prediction successful!")
        print("\n[INFO] x_train_predict shape: {}\n\
        x_test_predict shape: {}".format(x_train_predict.shape, x_test_predict.shape))
        time.sleep(5)
        '''

        # DEFINE MODEL PARAMETERS
        input_img = (trainData.shape[1], trainData.shape[2], trainData.shape[3])
        model = cnn(input_img)
        time.sleep(4)
        directory = "model-save/"
        filepath = directory + "5_trained_model.hdf5"
        #model = load_model(filepath)
        print("\n[INFO] Final model summary")
        print(model.summary())

        # MODEL SETTINGS
        BATCH_SIZE = 32
        EPOCHS = epoch
        patience = (EPOCHS * 15) / 100      # 10%


        # CHECKPOINT
        checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True,
            mode='max')
        early = EarlyStopping(monitor='val_acc', min_delta = 0, patience = patience, verbose = 1,
            mode = 'max', restore_best_weights = True)
        tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 2, batch_size = BATCH_SIZE,
            write_graph = True, write_images = False, embeddings_layer_names = None, update_freq = 'epoch') 
        callbacks_list = [checkpoint, early, tensor_board]

        # DATA AUGMENTATION
        aug = ImageDataGenerator(rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest")

        # TRAINING SETTINGS
        learning_rate = 1e-3
        decay_rate = learning_rate / EPOCHS
        momentum = 0.2
        sgd = SGD(lr = learning_rate, momentum = momentum, decay = decay_rate, nesterov = False)
        #adam = Adam(lr = learning_rate, beta_1 = 0.1, beta_2 = 0.999, epsilon = 1e-05, decay = 0.0)
        adam = Adam(lr = 0.0001)

        model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        class_weight = class_weight()
                
        '''
        history = model.fit_generator(
            aug.flow(trainDataArray, trainLabelArray,
            batch_size = BATCH_SIZE),
            epochs = EPOCHS, verbose = 1,
            validation_data = (testDataArray, testLabelArray),
            steps_per_epoch = len(trainDataArray) // BATCH_SIZE,
            shuffle = True,
            callbacks = callbacks_list,
            class_weight = class_weight)
        '''
        # Model Fit
        history = model.fit(trainData, trainLabels,
            batch_size = BATCH_SIZE,
            epochs = EPOCHS, verbose = 1,
            validation_data = (testData, testLabels),
            class_weight = class_weight,
            shuffle = True,
            callbacks = callbacks_list)
        
        
        # Model evaluate
        model = load_model(filepath)

        scores = model.evaluate(testData, testLabels)
        print("\nAccuracy: %.2f%%" % (scores[1]*100))

        predict = model.predict(testData)
        threshold = 0.5
        predict[predict > threshold] = 1
        predict[predict <= threshold] = 0

        #ab = np.argmax(testLabels, axis = 1)
        #bc = np.argmax(predict, axis = 1)

        y_test_non_category = [ np.argmax(t) for t in testLabels ]
        y_predict_non_category = [ np.argmax(t) for t in predict ]

        cm = ConfusionMatrix(actual_vector=y_test_non_category, predict_vector=y_predict_non_category)
        cm.save_html("classification_report")
        print('\nConfusion Matrix: \n', cm, '\n')

        #target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

        #cr = classification_report(y_test_non_category, y_predict_non_category) # , labels=target_names
        #print('\nClassification Report: \n', cr)


        # Train loss curve plot
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy Progress')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train_acc', 'val_acc'], loc='lower right')
        plt.savefig('accuracy_curve.png') 

        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss Progress')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train_loss', 'val_loss'], loc='upper left')
        plt.savefig('loss_curve.png')


    # -------------------- TEST --------------------
    if DEBUG == 2:
        pass

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("[INFO] Uso: python oil_class.py EPOCHS DEBUG")
        print("[INFO] DEBUG\n1 -> treinar a rede neural\n2 -> testar a rede neural")

    epoch = int(sys.argv[1])
    debug = int(sys.argv[2])
    main(epoch, debug)

'''
tensorboard --logdir logs/1 --port 6006

http://localhost:16006/

https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
'''
