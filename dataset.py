import numpy as np
import pandas as pd
np.random.seed(14)

# data_folder = "/opt/data_repository/oil_samples/"

def load_dataset():
    # PATH = "/opt/data_repository/laminas_full_1.pkl"
    PATH = "/opt/data_repository/oil_samples/laminas.pkl"
    df = pd.read_pickle(PATH)
    
    # parametro de carregamento do dataset
    x = 0
    y = 99
    df = df.iloc[x:y, :]
    images = df['lamina']
    labels = df['classificacao']

    # class weight
    weights = np.zeros(13)
    for idx,row in df.iterrows():
        weights += np.array(row['classificacao'])
    
    weights /= len(df)
    class_weight = dict(enumerate(weights))


    # converte o dataset em arrays
    img2array = []
    labels2array = []
    #for i in range(x, y):
    for i in range(len(labels)):
        # IMAGE ARRAY
        imgarr = np.array(images[i])
        imgarr = imgarr / 255
        img2array.append(imgarr)

        # LABEL ARRAY
        labelsarr = np.array(labels[i])
        labels2array.append(labelsarr)

    img_array = np.asarray(img2array)
    labels_array = np.asarray(labels2array)

    return img_array, labels_array
    #return img_array, labels_array, class_weight


if __name__ == '__main__':
    load_dataset()


'''
# Retirado classe 1 e 9
df['0'] = df['classificacao'].str[0]
df['1'] = df['classificacao'].str[2]
df['2'] = df['classificacao'].str[3]
df['3'] = df['classificacao'].str[4]
df['4'] = df['classificacao'].str[5]
df['5'] = df['classificacao'].str[6]
df['6'] = df['classificacao'].str[7]
df['7'] = df['classificacao'].str[8]
df['8'] = df['classificacao'].str[10]
df['9'] = df['classificacao'].str[11]
df['10'] = df['classificacao'].str[12]

del df['classificacao']
del df['id']

images = df.loc[: , "lamina"]
labels = df.loc[: , "0":]
'''
'''
# -------------------- CROSS VALIDATION --------------------
test = df.loc[7499:8799 , :]
images_test = test.loc[: , "lamina"]
labels_test = test.loc[: , "classificacao"]

X = images
Y = labels
k_folds = 10
kfold = KFold(n_splits = k_folds, shuffle = True, random_state = seed)
kfold.get_n_splits(images)
for train_index, test_index in kfold.split(X):
    trainData = X[train_index]
    testData = X[test_index]
    trainLabels = Y[train_index]
    testLabels = Y[test_index]
    '''