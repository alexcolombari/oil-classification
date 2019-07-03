import time
import random
import numpy as np
import pandas as pd
from models import *
from keras import metrics
from keras.layers import Input
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.model_selection import KFold
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad
seed = 7
np.random.seed(seed)

def train_model():
    # -------------------- CLASS WEIGHT --------------------
    def class_weight():
        soma = np.zeros(13)
        for idx,row in df.iterrows():
            soma += np.array(row['classificacao'])
        soma /= len(df)
        class_weight = dict(enumerate(soma))
        return class_weight

    # -------------------- GET LAYER WEIGHT --------------------
    def get_weights():
        for layer in output_model.layers: print(layer.get_config(), layer.get_weights())

    # -------------------- DATA MANIPULATION --------------------
    data_folder = "/opt/data_repository/oil_samples/"
    file_to_open = data_folder + "laminas.pkl"
    valor = 6899    # 7799
    df = pd.read_pickle(file_to_open)
    df = df.loc[:valor , :]
    imagens = df.loc[: , "lamina"]
    labels = df.loc[: , "classificacao"]

    # -------------------- CROSS VALIDATION --------------------
    X = imagens
    Y = labels
    k_folds = 10
    kfold = KFold(n_splits = k_folds, shuffle = False, random_state = None)
    kfold.get_n_splits(imagens)
    for train_index, test_index, in kfold.split(X):
        trainData = X[train_index]
        testData = X[test_index]
        trainLabels = Y[train_index]
        testLabels = Y[test_index]

    train_length = len(trainData)
    test_length = len(testData)

    trainData = np.asarray(trainData)
    testData = np.asarray(testData)
    trainLabels = np.asarray(trainLabels)
    testLabels = np.asarray(testLabels)

    print("\n[INFO] Cross validation successful!")
    
    time.sleep(3)

    
    # Image object convertion to array
    trainDataArray = []
    for i in range(train_length):
        trainx = np.array(trainData[i])
        trainx = np.resize(trainx, (300, 400, 1))
        trainDataArray.append(trainx)
    trainDataArray = np.asarray(trainDataArray)
    trainDataArray = trainDataArray / 255.
    print("[INFO] Images appended successful!")
    
    testDataArray = []
    for j in range(len(testData)):
        testx = np.array(testData[j])
        testx = np.resize(testx, (300, 400, 1))
        testDataArray.append(testx)
    testDataArray = np.asarray(testDataArray)
    testDataArray = testDataArray / 255.
    print("[INFO] Images appended successful!")

    
    # Label object convertion to array
    trainLabelArray = []
    for x in range(train_length):
        labelsarrx = np.array(trainLabels[x])
        trainLabelArray.append(labelsarrx)
    trainLabelArray = np.asarray(trainLabelArray)
    print("[INFO] Train Labels appended successful!\n")

    testLabelArray = []
    for z in range(test_length):
        labelsarry = np.array(testLabels[z])
        testLabelArray.append(labelsarry)
    testLabelArray = np.asarray(testLabelArray)
    print("[INFO] Test Labels appended successful!\n")


    print("[INFO] trainData shape: {}\n       testData shape: {}\n\n\
       trainLabels shape: {}\n       testLabels shape: {}\n".format(trainDataArray.shape, testDataArray.shape,
        trainLabelArray.shape, testLabelArray.shape))

    time.sleep(2)

    
    # -------------------- TRAINING --------------------

    '''
    # AUTOENCODER MODEL LOAD
    autoencoder_directory = "model-save/"
    autoencoder_path = autoencoder_directory + "3_auto_encoder_model.hdf5"
    autoencoder_model = load_model(autoencoder_path)

    # GET AUTOENCODED MODEL LAYER
    encoder = Model(autoencoder_model.input, autoencoder_model.layers[-6].output)
    print("\n\n[INFO] Autoencoder model summary")
    print(encoder.summary())
    print("\n[INFO] Starting autoencoder prediction!")
    time.sleep(2)

    # Normalization
    x_train_predict = encoder.predict(trainDataArray)
    x_train_predict = x_train_predict.astype('uint8') / 255.

    x_test_predict = encoder.predict(testDataArray)
    x_test_predict = x_test_predict.astype('uint8') / 255.

    print("\n[INFO] Prediction successful!")
    print("\n[INFO] x_train_predict shape: {}\n\
       x_test_predict shape: {}".format(x_train_predict.shape, x_test_predict.shape))
    time.sleep(5)
    '''

    # INPUT SHAPE AND MODEL SETTINGS
    x, y = trainDataArray.shape[1], trainDataArray.shape[2]  # 75, 100
    inChannel = trainDataArray.shape[3]
    input_img = Input(shape = (x, y, inChannel))
    BATCH_SIZE = 64
    EPOCHS = 10000
    patience = (EPOCHS * 10) / 100

    # DEFINE OUTPUT MODEL
    final_model = cnn(input_img)
    output_model = Model(input_img, final_model)
    print("\n[INFO] Final model summary")
    print(output_model.summary())
    time.sleep(4)
    directory = "model-save/"
    filepath = directory + "3_trained_model.hdf5"

    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True,
        mode='max')
    early = EarlyStopping(monitor='val_acc', min_delta = 0, patience = patience, verbose = 1,
        mode = 'max', restore_best_weights = True)
    #tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 2, batch_size = BATCH_SIZE,
        #write_graph = False, write_images = False, embeddings_layer_names = None, update_freq = 'epoch')    
    callbacks_list = [checkpoint, early]

    # SETTINGS
    learning_rate = 1e-5
    decay_rate = learning_rate / EPOCHS
    momentum = 0.7
    sgd = SGD(lr = learning_rate, momentum = momentum, decay = decay_rate, nesterov = False)
    adam = Adam(lr = learning_rate, beta_1 = 0.1, beta_2 = 0.999, epsilon = 1e-05, decay = 0.0)

    output_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # DATA AUGMENTATION
    aug = ImageDataGenerator(rotation_range= 60, width_shift_range=0.37, 
            height_shift_range=0.50, shear_range=0.56, zoom_range=0.50,
                horizontal_flip=True, fill_mode="nearest")
    
    '''
    history = output_model.fit_generator(
        aug.flow(x_train_predict, trainLabels,
        batch_size = BATCH_SIZE),
        epochs = EPOCHS, verbose = 1,
        validation_data = (x_test_predict, testLabels),
        shuffle = False,
        callbacks = callbacks_list)
        #class_weight = class_weight(),
    '''
    # Model Fit
    history = output_model.fit(trainDataArray, trainLabelArray,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS, verbose = 1,
        validation_data = (testDataArray, testLabelArray),
        class_weight = class_weight(),
        shuffle = False,
        callbacks = callbacks_list)
    
    
    # Model evaluate
    output_model = load_model(filepath)

    cvscores = []
    scores = output_model.evaluate(testDataArray, testLabelArray)
    print("%s: %.2f%%" % (output_model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    #print("\nAccuracy: %.2f%%" % (scores[1]*100))

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    predict = output_model.predict(testDataArray)
    predict[predict>=0.5] = 1
    predict[predict<0.5] = 0

    # Train loss curve plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy Progress')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train_acc', 'val_acc'], loc='lower right')
    plt.savefig('accuracy_curve.png') 

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss Progress')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.savefig('loss_curve.png')  

if __name__ == "__main__":
    train_model()

'''
----- name: trained_model -----
Trained with 101426 epochs
    Acc: 90.40%
    Val loss: 0.2552
    
----- name: 3_trained_model -----
Trained with 1074 epochs
    Acc: 89.22%
    Val loss: 0.2539
'''
