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
    file_to_open = data_folder + "5000-samples.pkl"

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
        img_resize = np.resize(imgarr, (100, 200, 1))
        img2array.append(img_resize)

        # LABEL ARRAY
        labelsarr = np.array(labels[i])
        labels2array.append(labelsarr)

    img_array = np.asarray(img2array)
    labels_array = np.asarray(labels2array)    
    
    

    # SPLIT
    train_images, test_images, train_labels, test_labels = train_test_split(img_array, labels_array)

    # NORMALIZATION
    train_images = train_images / 255.0
    test_images = test_images / 255.0   

    # MODEL LOAD
    model_directory = "model-save/"
    model_path = model_directory + "auto_encoder_model.hdf5"
    autoencoder_model = load_model(model_path)

    # INPUT SHAPE AND MODEL SETTINGS
    x, y = 25, 50
    inChannel = 3
    input_img = Input(shape = (x, y, inChannel))
    BATCH_SIZE = 256
    EPOCHS = 1300
    
    # GET AUTOENCODED MODEL LAYERS
    encoder = Model(autoencoder_model.input, autoencoder_model.layers[-6].output)

    x_train_predict = encoder.predict(train_images)
    x_train_predict = np.resize(x_train_predict, (len(x_train_predict), 25, 50, 3))    
    x_test_predict = encoder.predict(test_images)
    x_test_predict = np.resize(x_test_predict, (len(x_test_predict), 25, 50, 3))

    final_model = model_1(input_img)
    output_model = Model(input_img, final_model)  
    print(output_model.summary()) 
    directory = "model-save/"
    filepath = directory + "trained_model.hdf5"


    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'accuracy', verbose = 1, save_best_only = True,
        mode='max')
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 950, verbose = 1,
        mode = 'min', restore_best_weights = True)
    #tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 2, batch_size = BATCH_SIZE,
        #write_graph = False, write_images = False, embeddings_layer_names = None, update_freq = 'epoch')    
    callbacks_list = [checkpoint]

    output_model.compile(optimizer = "adagrad", loss = "binary_crossentropy", metrics = ['accuracy'])
    
    test_images = np.resize(test_images, (len(test_images), 25, 50, 3))

    output_model.fit(x_train_predict, train_labels, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1,
        validation_data = (test_images, test_labels), class_weight = class_weight, callbacks = callbacks_list)

    output_model = load_model(filepath)

    scores = output_model.evaluate(x_test_predict, test_labels)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

    predict = output_model.predict(x_test_predict)
    predict[predict>=0.5] = 1
    predict[predict<0.5] = 0

    print("Predict  25: {}".format(predict[25]))
    print("Original 25: {}".format(test_labels[25]))    


if __name__ == "__main__":
    train_model()
