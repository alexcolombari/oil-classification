'''
    Algorithm for testing trained model
'''
import numpy as np
import pandas as pd
from PIL import Image
from generate import *
from keras.layers import Input
from keras import backend as K
from keras.models import load_model, Model

def test_model():
    dataframe = createImage()
    
    # LOAD IMAGE
    imagem = Image.open('test_image.jpg')
    imagem = np.array(imagem)
    imagem = np.resize(imagem, (1, 100, 200, 1))

    imagem = imagem / 255.0

    
    # MODEL LOAD
    model_directory = "model-save/"
    model_path = model_directory + "auto_encoder_model.hdf5"
    autoencoder_model = load_model(model_path)

    # INPUT SHAPE AND MODEL SETTINGS
    x, y = 25, 50
    inChannel = 3
    input_img = Input(shape = (x, y, inChannel))
    
    # GET AUTOENCODED MODEL LAYERS
    encoder = Model(autoencoder_model.input, autoencoder_model.layers[-6].output)

    x_train_predict = encoder.predict(imagem)
    x_train_predict = np.resize(x_train_predict, (len(x_train_predict), 25, 50, 3)) 


    # LOAD MODEL
    model_filepath = "model-save/"
    model_name = model_filepath + "trained_model.hdf5"
    model = load_model(model_name)

    # MODEL PREDICT
    pred = model.predict(x_train_predict)
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0

    vetor = getVetor()
    vetor[vetor>=0.5] = 1
    vetor[vetor<0.5] = 0

    print("Predict Label:  {}".format(pred))
    print("Original Label: {}".format(vetor))

    return pred, vetor

if __name__ == "__main__":
    test_model()
