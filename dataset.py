import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

def load_dataset():
    #data_folder = "/opt/data_repository/oil_samples/"
    #file_to_open = data_folder + "half-samples.pkl"

    file_to_open = "laminas-dez.pkl"
    df = pd.read_pickle(file_to_open)

    imagens = df.loc[: , "lamina"]
    labels = df.loc[: , "classificacao"]

    # Conversao do dataset em array
    img2array = []
    labels2array = []

    #Autoencoder

    for i in range(len(imagens)):
        imgarr = np.array(imagens[i])
        img_resize = np.resize(imgarr, (300, 400, 3))  #100, 113, 3
        img2array.append(img_resize)

        labelsarr = np.asarray(labels[i])
        labels2array.append(labelsarr)

    img_array = np.array(img2array)
    labels_array = np.array(labels2array)

    return img_array, labels_array

'''
    labels = [("Particulas corrosivas"),
              ("Fibras"),
              ("Esferas"),
              ("Oxido preto"),
              ("Oxido vermelho"),
              ("Arrancamento"),
              ("Mancal"),
              ("Desgaste deslizamento"),
              ("Atrito normal"),
              ("Esferas contaminante"),
              ("Areia / sujeira"),
              ("Engrenagem"),
              ("Degradacao do lubrificante")]


# Class weight
def class_weight_dataset(y_train, y_test):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoder.fit(y_test)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    y_train = y_train.flatten()
    class_weight_list = compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weight = dict(enumerate(np.unique(y_train), class_weight_list))
    y_train = keras.utils.to_categorical(y_train, num_classes = len(np.unique(y_train)))
    
    return class_weight, y_train, y_test
'''

