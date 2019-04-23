import numpy as np
import pandas as pd
np.random.seed(42)

def load_dataset():
    data_folder = "/opt/data_repository/oil_samples/"
    #file_to_open = data_folder + "5000-samples.pkl"
    file_to_open = data_folder + "laminas.pkl"

    #file_to_open = "laminas-dez.pkl"
    df = pd.read_pickle(file_to_open)

    imagens = df.loc[: , "lamina"]
    labels = df.loc[: , "classificacao"]

    # CLASS WEIGHT MEAN FOR EACH COLUMN IN "CLASSIFICACAO"
    soma = np.zeros(13)
    for idx,row in df.iterrows():
        soma+=np.array(row['classificacao'])
        
    soma/=len(df)
    class_weight = dict(enumerate(soma))


    # DATASET CONVERTION TO ARRAY TYPE
    img2array = []
    labels2array = []

    for i in range(len(imagens)):
        # IMAGE ARRAY
        imgarr = np.array(imagens[i])
        img_resize = np.resize(imgarr, (100, 200, 3))  #100, 113, 3
        img2array.append(img_resize)

        # LABEL ARRAY
        labelsarr = np.array(labels[i])
        labels2array.append(labelsarr)

    img_array = np.asarray(img2array)
    labels_array = np.asarray(labels2array)

    return img_array, labels_array, class_weight

load_dataset()
