import numpy as np
from models import createModel
from dataset import load_dataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(42)

# search Under and Over sampling or Class weight

# (400 / 300) x 100 = new height

def train(img_array, labels_array):
    # Split de dados de treino / teste
    X_train, X_test, Y_train, Y_test = train_test_split(img_array, labels_array)

    # Normalizacao
    X_train = X_train / 255.0
    X_test = X_test / 255.0


    # Data augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.15,
        height_shift_range=0.15, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


    print("X_train shape: {}\nY_train shape: {}".format(X_train.shape, Y_train.shape))
    print("X_test shape: {}\nY_test shape: {}".format(X_test.shape, Y_test.shape))


    # Definicao do input shape
    inputShape = (100, 133, 3)

    # Modelo
    trainmodel = createModel(inputShape)
    batch_size = 32
    epochs = 100
    trainmodel.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
 

    # Fit do modelo com data aug
    history = trainmodel.fit_generator(
        aug.flow(X_train, Y_train, batch_size=batch_size),
        epochs=epochs, verbose=1, steps_per_epoch=len(X_train)//batch_size,
        validation_data=(X_test, Y_test))
 
    scores = trainmodel.evaluate(X_test, Y_test)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))


    Y_pred = trainmodel.predict(X_test)
    #print("Previsao baseado em X de teste: {}".format(abcd))
    #print(Y_test)

    print('\nConfusion Matrix')
    matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
    print(matrix)

'''
y_test = np.argmax(Y_test, axis=1)
y_pred = trainmodel.predict(X_test)
print(classification_report(y_test, y_pred))


# salva um arquivo do modelo
trainmodel.save("model")'''

dataset, labels = load_dataset()
train(dataset, labels)
