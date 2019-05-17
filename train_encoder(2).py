import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
import matplotlib.pyplot as plt
from dataset import load_dataset
from keras.utils import plot_model
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, BatchNormalization
np.random.seed(7)

def train_autoencoder():

    data_folder = "/opt/data_repository/oil_samples/"
    file_to_open = data_folder + "laminas.pkl"
    
    df = pd.read_pickle(file_to_open)
    imagens = df.loc[0:3000 , "lamina"]

    input_img = Input(shape=(300, 396, 1))

    x = Conv2D(12, (3, 3), activation=K.relu, padding='same')(input_img) # 300 x 396
    x = MaxPooling2D((2, 2), padding='same')(x) # 300 x 396
    x = Conv2D(6, (3, 3), activation=K.relu,, padding='same')(x) # 150 x 198
    x = MaxPooling2D((2, 2), padding='same')(x) # 150 x 198
    x = Conv2D(6, (3, 3), activation=K.relu,, padding='same')(x) # 75 x 99
    encoded = MaxPooling2D((2, 2), padding='same')(x) # 37.5 x 49.5   

    x = Conv2D(6, (3, 3), activation=K.relu,, padding='same')(encoded) # 37.5 x 49.5
    x = UpSampling2D((2, 2))(x) # 37.5 x 49.5
    x = Conv2D(6, (3, 3), activation=K.relu,, padding='same')(x) # 75 x 99
    x = UpSampling2D((2, 2))(x) # 75 x 99
    x = Conv2D(12, (3, 3), activation=K.relu,)(x) # 150 x 198
    x = UpSampling2D((2, 2))(x) # 150 x 198
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x) # 300 x 396


    img2array = []
    for i in range(len(imagens)):
            imgarr = np.array(imagens[i])
    img_array = np.asarray(imgarr)


    x_train, x_test = train_test_split(img_array, test_size = 0.33)

    x_train = x_train.astype('uint8') / 255.
    x_test = x_test.astype('uint8') / 255.
    x_train = np.resize(x_train, (len(x_train), 300, 396, 1))
    x_test = np.resize(x_test, (len(x_test), 300, 396, 1))

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')
    #plot_model(autoencoder, to_file='model_encoder.png', show_shapes = True)

    directory = "model-save/"
    filepath = directory + "2_auto_encoder_model.hdf5"

    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True,
        mode='min')
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 1300, verbose = 1,
        mode = 'min', restore_best_weights = True)
    callbacks_list = [checkpoint, early]
    

    history = autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks = callbacks_list)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('2_encoder_loss_curve.png')

if __name__ == "__main__":
    train_autoencoder()

    # val loss (adadelta / mse) = 0.0127
