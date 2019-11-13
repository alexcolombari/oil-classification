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
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
from keras.constraints import max_norm
from keras import metrics, regularizers, initializers
np.random.seed(14)

MODEL_PATH = "model-save/"
MODEL_SAVED = MODEL_PATH + "model.h5"

def main(epoch):
    img_array, labels_array, class_weight = load_dataset()
    trainData, testData, trainLabels, testLabels = train_test_split(img_array, labels_array, test_size = 0.3, shuffle = True, random_state = None)

    print("\n[INFO] trainData shape: {}\n       testData shape: {}\n\n\
        trainLabels shape: {}\n       testLabels shape: {}\n".format(trainData.shape, testData.shape,
        trainLabels.shape, testLabels.shape))
    
    # DEFINE MODEL PARAMETERS
    input_img = (trainData.shape[1], trainData.shape[2], 3)
    model = load_model(MODEL_SAVED, compile = True)
    print("Loaded model from disk")
    print(model.summary())
    #print(model.get_weights())
    #print(model.optimizer)
    
    time.sleep(2)

    # MODEL SETTINGS
    BATCH_SIZE = 10
    EPOCHS = epoch
    calls = model_callbacks(EPOCHS, BATCH_SIZE)

    # Model Fit
    history = model.fit(trainData, trainLabels,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS, verbose = 1,
        validation_data = (testData, testLabels),
        class_weight = class_weight,
        shuffle = True,
        callbacks = calls)

    # model.save(MODEL_SAVED)
    
    # Model evaluate
    # Model open
    # loaded_model = load_model(MODEL_SAVED)
    print("Loaded model from disk to evaluate")

    scores = model.evaluate(testData, testLabels)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

    save_plot(history)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[INFO] Uso: python oil_class.py EPOCHS")

    epoch = int(sys.argv[1])
    main(epoch)

# ultima epoca melhorada: 
