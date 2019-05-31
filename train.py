import numpy as np
import pandas as pd
from models import *
from keras.layers import Input
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
np.random.seed(7)

def train_model():
    data_folder = "/opt/data_repository/oil_samples/"
    file_to_open = data_folder + "laminas.pkl"

    df = pd.read_pickle(file_to_open)
    df = df.loc[:6299 , :]

    imagens = df.loc[: , "lamina"]
    labels = df.loc[: , "classificacao"]


    # CLASS WEIGHT MEAN FOR EACH COLUMN IN "CLASSIFICACAO"
    soma = np.zeros(13)
    for idx,row in df.iterrows():
        soma += np.array(row['classificacao'])
        
    soma/=len(df)
    class_weight = dict(enumerate(soma))

    
    # DATASET CONVERTION TO ARRAY TYPE
    img2array = []
    labels2array = []

    for i in range(len(imagens)):
        # IMAGE ARRAY
        imgarr = np.array(imagens[i])
        img2arr = np.resize(imgarr, (300, 400, 3))
        img2array.append(img2arr)

        # LABEL ARRAY
        labelsarr = np.array(labels[i])
        labels2array.append(labelsarr)

    img_array = np.asarray(img2array)
    labels_array = np.asarray(labels2array)    
    

    # SPLIT
    train_images, test_images, train_labels, test_labels = train_test_split(img_array, labels_array)


    # MODEL LOAD
    model_directory = "model-save/"
    model_path = model_directory + "auto_encoder_model.hdf5"
    autoencoder_model = load_model(model_path)

    # GET AUTOENCODED MODEL LAYER
    encoder = Model(autoencoder_model.input, autoencoder_model.layers[-6].output)

    x_train_predict = encoder.predict(train_images)
    x_train_predict = x_train_predict.astype('uint8') / 355
    #x_train_predict = np.resize(x_train_predict, (len(x_train_predict), 300, 400, 3))

    x_test_predict = encoder.predict(test_images)
    x_test_predict = x_test_predict.astype('uint8') / 355
    #x_test_predict = np.resize(x_test_predict, (len(x_test_predict), 300, 400, 3))



    # INPUT SHAPE AND MODEL SETTINGS
    x, y = 75, 100
    inChannel = 8
    input_img = Input(shape = (x, y, inChannel))
    BATCH_SIZE = 512
    EPOCHS = 300
    patience = (EPOCHS * 20) / 100
    

    final_model = model_1(input_img)
    output_model = Model(input_img, final_model)  
    print(output_model.summary()) 
    directory = "model-save/"
    filepath = directory + "trained_model.hdf5"


    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True,
        mode='max')
    early = EarlyStopping(monitor='val_acc', min_delta = 0, patience = patience, verbose = 1,
        mode = 'max', restore_best_weights = True)
    #tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 2, batch_size = BATCH_SIZE,
        #write_graph = False, write_images = False, embeddings_layer_names = None, update_freq = 'epoch')    
    callbacks_list = [checkpoint, early]

    output_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    

    history = output_model.fit(x_train_predict, train_labels,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS, verbose = 1,
        validation_data = (x_test_predict, test_labels),
        class_weight = class_weight,
        shuffle = False,
        callbacks = callbacks_list)

    output_model = load_model(filepath)

    scores = output_model.evaluate(x_test_predict, test_labels)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

    predict = output_model.predict(x_test_predict)
    predict[predict>=0.5] = 1
    predict[predict<0.5] = 0

    print("Predict  35: {}".format(predict[35]))
    print("Original 35: {}".format(test_labels[35]))  


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('train_curve.png')  


if __name__ == "__main__":
    train_model()
