import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

    for i in range(229):
        imgarr = np.array(imagens[i])
        img_resize = np.resize(imgarr, (10, 10, 3))  #100, 113, 3
        img2array.append(img_resize)

        labelsarr = np.asarray(labels[i])
        labels2array.append(labelsarr)

    img_array = np.array(img2array)
    labels_array = np.array(labels2array)

    return img_array, labels_array

'''
def label_encoder(df):
    labels = ['Particulas corrosivas',
              'Fibras',
              'Esferas',
              'Oxido preto',
              'Oxido vermelho',
              'Arrancamento',
              'Mancal',
              'Desgaste deslizamento',
              'Atrito normal',
              'Esferas contaminante',
              'Areia / sujeira',
              'Engrenagem',
              'Degradacao do lubrificante']

    encoder = LabelEncoder()
    encoder.fit(labels)

    for i, item in enumerate(encoder.classes_):
        print(item, '=>', i)
    
    

    #le.fit(df['classificacao'])
    

    return df
    '''

# Class weight
def class_weight_dataset(y):
    y = y.flatten()
    class_weight_list = compute_class_weight('balanced', np.unique(y), y)
    class_weight = dict(zip(np.unique(y), class_weight_list))

    return class_weight
