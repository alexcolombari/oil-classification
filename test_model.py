'''
    Algorithm for testing trained model
'''
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model

def test_model():
    # LOAD IMAGE
    imagem = Image.open('test_image.jpg')
    imagem = np.array(imagem)
    imagem = np.resize(imagem, (1, 100,200,3))
    
    array_imagem = []
    array_imagem.append(imagem)
    array_imagem = np.expand_dims(array_imagem, axis = 1)
    print(array_imagem)

    # LOAD MODEL
    model_filepath = "model-save/"
    model_name = model_filepath + "2_val_loss_model.h5"
    model = load_model(model_name)

    # MODEL PREDICT
    pred = model.predict(imagem)
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0

    print("Predict Label: {}".format(pred))

test_model()
