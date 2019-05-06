import numpy as np
from models import encoder_model
from keras.layers import Input
import matplotlib.pyplot as plt
from dataset import load_dataset
from keras.utils import plot_model
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
np.random.seed(7)

def train_encoder(img_array, labels_array, class_weight):
    # SPLIT TRAIN AND TEST DATA
    train_X, valid_X, train_ground, valid_ground = train_test_split(img_array, img_array)

    # NORMALIZATION
    train_X = train_X / 255.0
    valid_X = valid_X / 255.0
    train_ground = train_ground / 255.0
    valid_ground = valid_ground / 255.0

    # DATA AUGMENTATION
    aug = ImageDataGenerator(rotation_range=50, width_shift_range=0.5, 
            height_shift_range=0.5, shear_range=0.60, zoom_range=0.55,
                horizontal_flip=True, fill_mode="nearest")

    # INPUT SHAPE AND MODEL SETTINGS
    x, y = 100, 200
    inChannel = 1
    input_img = Input(shape = (x, y, inChannel))
    
    directory = "model-save/"
    filepath = directory + "auto_encoder_model.hdf5"
    encoder = encoder_model(input_img)
    autoencoder = Model(input_img, encoder)
    BATCH_SIZE = 256
    EPOCHS = 8000

    # MODEL COMPILE
    autoencoder.compile(optimizer = 'adam', loss = 'mae')

    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True,
        mode='min')
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 950, verbose = 1,
        mode = 'min', restore_best_weights = True)
    callbacks_list = [checkpoint, early]

    # ENCODER FIT WITH DATA AUG
    history = autoencoder.fit_generator(
        aug.flow(train_X, train_ground, batch_size=BATCH_SIZE),
        epochs=EPOCHS, verbose=1, validation_data=(valid_X, valid_ground),
        callbacks = callbacks_list)
    
    '''
    history = autoencoder.fit(train_X, train_ground, epochs=5, batch_size=BATCH_SIZE, verbose=1,
        callbacks = callbacks_list, validation_data=(valid_X, valid_ground))
    '''

if __name__ == "__main__":
    dataset, labels, class_weight = load_dataset()
    train_encoder(dataset, labels, class_weight)
