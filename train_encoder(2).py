import numpy as np
import pandas as pd
from PIL import Image
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
    imagens = df.loc[0:1000 , "lamina"]

    input_img = Input(shape=(300, 396, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)    

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


    img2array = []
    for i in range(len(imagens)):
            imgarr = np.array(imagens[i])
    img_array = np.asarray(imgarr)


    x_train, x_test = train_test_split(img_array, test_size = 0.33)

    x_train = x_train.astype('float') / 255.
    x_test = x_test.astype('float') / 255.
    x_train = np.resize(x_train, (len(x_train), 300, 396, 1))
    x_test = np.resize(x_test, (len(x_test), 300, 396, 1))

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')
    plot_model(autoencoder, to_file='model_encoder.png', show_shapes = True)

    directory = "model-save/"
    filepath = directory + "auto_encoder_model.hdf5"

    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True,
        mode='min')
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 100, verbose = 1,
        mode = 'min', restore_best_weights = True)
    callbacks_list = [checkpoint, early]
    

    autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks = callbacks_list)

    autoencoder = load_model(filepath)

    decoded_imgs = autoencoder.predict(x_test)
    print(decoded_imgs)
    exit()
    x_test = np.resize(x_test, (300, 396, 1))
    decoded_imgs = np.resize(decoded_imgs, (300, 396, 1))
    decoded_imgs = decoded_imgs * 255.0

    imgs = Image.fromarray(decoded_imgs[1], 'RGB')
    imgs.save('reconstructed.png')


if __name__ == "__main__":
    train_autoencoder()
