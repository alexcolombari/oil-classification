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

def train_model(img_array, labels_array, class_weight):
    # SPLIT
    train_images, test_images, train_labels, test_labels = train_test_split(img_array, labels_array) 

    # NORMALIZATION
    train_images = train_images / 255
    test_images = test_images / 255   

    # MODEL LOAD
    model_directory = "model-save/"
    model_path = model_directory + "auto_encoder_model.hdf5"
    autoencoder_model = load_model(model_path)
    
    # GET LAYERS OF AUTOENCODED MODEL
    encoder = Model(autoencoder_model.input, autoencoder_model.layers[-6].output)
    print(encoder.summary())

    predict = encoder.predict(test_images)
    print("Predict: {}".format(predict))


if __name__ == "__main__":
    dataset, labels, class_weight = load_dataset()
    train_model(dataset, labels, class_weight)
