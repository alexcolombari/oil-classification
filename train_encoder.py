import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO  
from keras import optimizers
from keras import backend as K
import matplotlib.pyplot as plt
from dataset import load_dataset
from keras.utils import plot_model
from sklearn.model_selection import KFold
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D
seed = 7
np.random.seed(seed)

def autoencoder(input_img):
    x1 = Conv2D(8, (3, 3), activation='tanh', padding='same')(input_img)
    x2 = MaxPooling2D((2, 2), padding='same')(x1)
    x3 = Dropout(0.3)(x2)
    x4 = Conv2D(8, (3, 3), activation='tanh', padding='same')(x3)
    encoded = MaxPooling2D((2, 2), padding='same')(x4)

    print("shape of encoded", K.int_shape(encoded))

    x5 = Conv2D(8, (3, 3), activation='tanh', padding='same')(encoded)
    x6 = UpSampling2D((2, 2))(x5)
    x7 = Conv2D(8, (3, 3), activation='tanh', padding='same')(x6)
    x8 = UpSampling2D((2, 2))(x7)
    decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x8)
    print("shape of decoded", K.int_shape(decoded))
    return decoded

 
# LOAD DATASET
data_folder = "/opt/data_repository/oil_samples/"
file_to_open = data_folder + "laminas.pkl"
df = pd.read_pickle(file_to_open)
valor = 6699
df = df.loc[:valor , :]
imagens = df.loc[: , "lamina"]
length = len(df)

# CONVERSION TO ARRAY
img = []
for i in range(length):
    imgarr = np.array(imagens[i])
    imgarr = np.resize(imgarr, (300, 400, 1))
    img.append(imgarr)
img = np.asarray(img)

# SPLIT
trainData, testData = train_test_split(img)
#train_data = np.resize(train_data, (len(train_data), 300, 400, 1))
#test_data = np.resize(test_data, (len(test_data), 300, 400))

trainData = trainData.astype('uint8') / 255.0
testData = testData.astype('uint8') / 255.0

x, y = trainData.shape[1], trainData.shape[2]
inChannel = 1

input_img = Input(shape=(x, y, inChannel))  # input_img = Input(shape=(300, 400, 3))
autoencoder = Model(input_img, autoencoder(input_img))

autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
#plot_model(autoencoder, to_file='model_encoder.png', show_shapes = True)
print(autoencoder.summary())

directory = "model-save/"
filepath = directory + "3_auto_encoder_model.hdf5"

epochs = 1500
batch_size = 64
patience = (epochs * 10) / 100

# CHECKPOINT
checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True,
    mode='min')
early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = patience, verbose = 1,
    mode = 'min', restore_best_weights = True)
callbacks_list = [checkpoint, early]

history = autoencoder.fit(trainData, trainData,
            epochs = epochs,
            batch_size = batch_size,
            shuffle = False,
            validation_data = (testData, testData),
            callbacks = callbacks_list)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Autoencoder Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('encoder_loss_curve.png')


'''
val loss (adadelta / mse) = 0.0127
val loss (adam / mse) = 0.00118

 ----------------------------------

val loss (adam / mae) = 0.03445 // accuracy = 0.9994 (1000 epochs)

30/05 val loss (adadelta / mse) = 0.00968 // accuracy = 0.9712 (15000 epochs)
07/06 val loss (adam / mse) = 0.00265 (4500 epochs)

21/06 val loss (adam / mse) = 0.00212 (250 epochs)
'''
