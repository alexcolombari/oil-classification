import os
import sys
import time
import random

from models import cnn
from save_plot import save_plot
from dataset import load_dataset
from model_callbacks import model_callbacks

from pycm import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
np.random.seed(14)

initial_time = time.time()

MODEL_PATH = "model-save/"
MODEL_SAVED = MODEL_PATH + "model.h5"


# PESOS
# {0: 0.4127812194326704, 1: 0.0, 2: 0.0014128899032713835, 3: 0.789588088251277,
# 4: 0.03771329203347462, 5: 0.00097815454841865, 6: 0.010542332355178785, 7: 0.6375393978915336,
# 8: 0.7877404629931529, 9: 0.0, 10: 0.25279860884686445, 11: 0.3882186718834909, 12: 0.0026084121291164004}

def main(epoch):
    img_array, labels_array, class_weight = load_dataset()
    trainData, testData, trainLabels, testLabels = train_test_split(img_array, labels_array, test_size = 0.3, shuffle = True, random_state = None)

    print("\n[INFO] trainData shape: {}\n       testData shape: {}\n\n\
       trainLabels shape: {}\n       testLabels shape: {}\n".format(trainData.shape, testData.shape,
        trainLabels.shape, testLabels.shape))

    time.sleep(2)

    # MODEL SETTINGS
    BATCH_SIZE = 10
    EPOCHS = epoch

    class_weight = class_weight
    calls = model_callbacks(EPOCHS, BATCH_SIZE)

    # -------------------- TRAINING --------------------

    # -------------------- GET LAYER WEIGHT --------------------
    def get_weights():
        for layer in model.layers: print(layer.get_config(), layer.get_weights())

    try:
        # DEFINE MODEL PARAMETERS
        input_img = (trainData.shape[1], trainData.shape[2], 3)
        model = cnn(input_img)
        time.sleep(4)
        print(model.summary())

        adam = Adam(lr = 0.001)
        model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Model Fit
        history = model.fit(trainData, trainLabels,
            batch_size = BATCH_SIZE,
            epochs = EPOCHS, verbose = 1,
            validation_data = (testData, testLabels),
            class_weight = class_weight,
            shuffle = True,
            callbacks = calls)

        save_plot(history)

    except KeyboardInterrupt:
        print("\n\nTraining stopped!")

    final_time = time.time() - initial_time
    print("Elapsed time: {:.2f}s".format(final_time))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[INFO] Uso: python oil_class.py EPOCHS")

    epoch = int(sys.argv[1])
    main(epoch)

'''
tensorboard --logdir logs/ --port 6006

http://localhost:16006/

https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
'''

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
