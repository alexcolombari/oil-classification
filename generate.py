import pandas as pd
import numpy as np
from PIL import Image
import random

# GENERATE PICKLE ARCHIVE
data_folder = "/opt/data_repository/oil_samples/"
#file_to_open = data_folder + "laminas.pkl"
file_to_open = "laminas-dez.pkl"
df = pd.read_pickle(file_to_open)

def getNumero():
    for i in df:
        numero = random.randint(0, len(df))
    return numero

amostra = getNumero()

def createImage():
    aa = df.loc[amostra, "lamina"]
    aa.save('test_image.jpg')
    print("[INFO] DONE!")

def getVetor():
    ab = df.loc[amostra, "classificacao"]
    vetor = np.zeros((1, 13))
    vetor += np.array(ab)
    return vetor

#aa.to_pickle(data_folder + "7500-samples.pkl")
#createImage()
