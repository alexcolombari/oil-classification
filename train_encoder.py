import numpy as np
import pandas as pd
from PIL import Image
from keras import optimizers
import matplotlib.pyplot as plt
from dataset import load_dataset
from keras.utils import plot_model
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, BatchNormalization
np.random.seed(7)

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img) #176 x 176 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #88 x 88 x 32
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1) #88 x 88 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #44 x 44 x 64
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2) #44 x 44 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv3) #44 x 44 x 128
    up1 = UpSampling2D((2,2))(conv4) # 88 x 88 x 128
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1) # 88 x 88 x 64
    up2 = UpSampling2D((2,2))(conv5) # 176 x 176 x 64
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(up2) # 176 x 176 x 1

    return decoded


# LOAD DATASET
data_folder = "/opt/data_repository/oil_samples/"
file_to_open = data_folder + "laminas.pkl"

df = pd.read_pickle(file_to_open)

imagens = df.loc[0:99 , "lamina"]
#imagens = df.loc[0:9199 , "lamina"]

# CONVERSION TO ARRAY
for i in range(len(imagens)):
        imgarr = np.array(imagens[i])
img_array = np.asarray(imgarr)

# SPLIT
train_data, test_data = train_test_split(img_array)

train_data = np.resize(train_data, (len(train_data), 300, 400, 3))
test_data = np.resize(test_data, (len(test_data), 300, 400, 3))

train_data = train_data.astype('uint8') / 255.0
test_data = test_data.astype('uint8') / 255.0


input_img = Input(shape=(300, 400, 3)) # 176, 176, 1

autoencoder = Model(input_img, autoencoder(input_img))

autoencoder.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
#plot_model(autoencoder, to_file='model_encoder.png', show_shapes = True)
print(autoencoder.summary())

directory = "model-save/"
filepath = directory + "auto_encoder_model.hdf5"

epochs = 1000
batch_size = 64
patience = (epochs * 10) / 100

# CHECKPOINT
checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True,
    mode='min')
early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = patience, verbose = 1,
    mode = 'min', restore_best_weights = True)
callbacks_list = [checkpoint, early]

history = autoencoder.fit(train_data, train_data,
            epochs = epochs,
            batch_size = batch_size,
            shuffle = False,
            validation_data = (test_data, test_data),
            callbacks = callbacks_list)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Autoencoder Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('encoder_loss_curve.png')


# val loss (adadelta / mse) = 0.0127
# val loss (adam / mse) = 0.00118
# ----------------------------------
# val loss (adam / mse) = 0.03445 // accuracy = 0.9994 (1000 epochs)
