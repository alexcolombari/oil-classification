import numpy as np
from PIL import Image
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
    #autoencoder = load_model(filepath)
    BATCH_SIZE = 128
    EPOCHS = 700

    # MODEL COMPILE
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    print(autoencoder.summary())
    plot_model(autoencoder, to_file='model_encoder.png', show_shapes = True)

    # CHECKPOINT
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True,
        mode='min')
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 950, verbose = 1,
        mode = 'min', restore_best_weights = True)
    callbacks_list = [checkpoint, early]


    # ENCODER FIT WITH DATA AUG
    history = autoencoder.fit_generator(
        aug.flow(train_X, train_X, batch_size=BATCH_SIZE),
        epochs=EPOCHS, verbose=1, validation_data=(valid_X, valid_ground), #valid_X, valid_ground
        callbacks = callbacks_list)


    # Loss curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('encoder_loss_curve.png')


    # val loss = 0.08128
    # val loss = 0.01527

if __name__ == "__main__":
    dataset, labels, class_weight = load_dataset()
    train_encoder(dataset, labels, class_weight)
