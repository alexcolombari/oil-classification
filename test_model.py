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
    imagem = np.resize(imagem, (1, 75, 100, 8))
    imagem = imagem / 255.0
    # DATASET CONVERTION TO ARRAY TYPE
    img2array = []

    for i in range(len(imagem)):
        # IMAGE ARRAY
        imgarr = np.array(imagem[i])
        #img2arr = np.resize(imgarr, (300, 400, 3))
        img2array.append(imgarr)


    # LOAD MODEL
    model_filepath = "model-save/"
    model_name = model_filepath + "trained_model.hdf5"
    model = load_model(model_name)

    # MODEL PREDICT
    pred = model.predict(imagem)
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0

    vetor = getVetor()
    vetor[vetor>=0.5] = 1
    vetor[vetor<0.5] = 0

    print("\nPredict Label {}:  {}".format(amostra, pred))
    print("Original Label {}: {}".format(amostra, vetor))

    erro = np.mean(pred != vetor)
    
    if (pred == vetor).all():
        print("[INFO] Acertou 100%")
    else:
        print("[INFO] Erro de %.2f%%" % (erro*100))

    return pred, vetor

if __name__ == "__main__":
    test_model()
