import numpy as np
import pandas as pd

def load_dataset():
    data_folder = "/opt/data_repository/oil_samples/"
    file_to_open = data_folder + "half-samples.pkl"
    df = pd.read_pickle(file_to_open)

    imagens = df.loc[: , "lamina"]
    labels = df.loc[: , "classificacao"]

    # Conversao do dataset em array
    img2array = []
    labels2array = []

    for i in range(4000):
        imgarr = np.array(imagens[i])
        img_resize = np.resize(imgarr, (100, 133, 3))
        img2array.append(img_resize)

        labelsarr = np.asarray(labels[i])
        labels2array.append(labelsarr)

    img_array = np.array(img2array)
    labels_array = np.array(labels2array)

    return img_array, labels_array