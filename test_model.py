'''
    Algorithm for testing trained model
'''
import numpy as np
import pandas as pd
from PIL import Image
from generate import *
from keras import backend as K
from keras.models import load_model

def test_model():
    dataframe = createImage()
    # LOAD IMAGE
    imagem = Image.open('test_image.jpg')
    imagem = np.array(imagem)
    imagem = np.resize(imagem, (1, 100, 200, 3))

    imagem = imagem / 255.0
    
    array_imagem = []
    array_imagem.append(imagem)
    array_imagem = np.expand_dims(array_imagem, axis = 1)

    # LOAD MODEL
    model_filepath = "model-save/"
    model_name = model_filepath + "5_val_loss_model.hdf5"
    model = load_model(model_name)

    # MODEL PREDICT
    pred = model.predict(imagem)
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0

    vetor = getVetor()
    vetor[vetor>=0.5] = 1
    vetor[vetor<0.5] = 0

    print("Predict Label:  {}".format(pred))
    print("Original Label: {}".format(vetor))


test_model()
