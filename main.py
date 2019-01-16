import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
np.random.seed(42)

data_folder = "/opt/data_repository/oil_samples/"
file_to_open = data_folder + "half-samples.pkl"
df = pd.read_pickle(file_to_open)

imagens = df.loc[: , "lamina"]
labels = df.loc[: , "classificacao"]

# Conversão do dataset em array
img2array = []
labels2array = []

for i in range(4000):
    imgarr = np.array(imagens[i])
    img_resize = np.resize(imgarr, (300, 400, 3))
    img2array.append(img_resize)

    labelsarr = np.asarray(labels[i])
    labels2array.append(labelsarr)

img_array = np.array(img2array)
labels_array = np.array(labels2array)

# Split de dados de treino / teste
X_train, X_test, Y_train, Y_test = train_test_split(img_array, labels_array)

# Normalização
X_train = X_train / 255.0
X_test = X_test / 255.0

# Data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.15,
    height_shift_range=0.15, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


print("X_train shape: {}\nY_train shape: {}".format(X_train.shape, Y_train.shape))
print("X_test shape: {}\nY_test shape: {}".format(X_test.shape, Y_test.shape))
exit()

# Definição do input shape
inputShape = (300, 400, 3)

# Inicio da rede neural
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape = inputShape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='sigmoid'))

    return model

# Modelo
trainmodel = createModel()
batch_size = 10
epochs = 100
trainmodel.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
 
# Fit do modelo com data aug
history = trainmodel.fit_generator(
    aug.flow(X_train, Y_train, batch_size=batch_size),
    epochs=epochs, verbose=1, steps_per_epoch=len(X_train)//batch_size,
    validation_data=(X_test, Y_test))
 
scores = trainmodel.evaluate(X_test, Y_test)
print("\nAccuracy: %.2f%%" % (scores[1]*100))


#Y_pred = trainmodel.predict(X_test)
#print("Previsao baseado em X de teste: {}".format(abcd))
#print(Y_test)
'''
print('\nConfusion Matrix')
matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
print(matrix)'''

y_test = np.argmax(Y_test, axis=1)
y_pred = trainmodel.predict(X_test)
print(classification_report(y_test, y_pred))

'''
# salva um arquivo do modelo
trainmodel.save("model")'''