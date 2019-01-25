import numpy as np
from models import createModel
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dataset import load_dataset, class_weight_dataset
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(42)

# search Under and Over sampling or Class weight

# (400 / 300) x 100 = new height      (100 x 113, 3)
'''
def sensitivity(Y_test, Y_pred):
    true_positives = K.sum(K.round(K.clip(Y_test * Y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(Y_test, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(Y_test, Y_pred):
    true_negatives = K.sum(K.round(K.clip((1-Y_test) * (1-Y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-Y_test, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
'''

def train(img_array, labels_array):
    # Spliting train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(img_array, labels_array)

    # Normalization
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    class_weight = class_weight_dataset(Y_train)
    

    # Data augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.15,
        height_shift_range=0.15, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


    print("X_train shape: {}\nY_train shape: {}".format(X_train.shape, Y_train.shape))
    print("X_test shape: {}\nY_test shape: {}".format(X_test.shape, Y_test.shape))

    # INPUT SHAPE AND MODEL SETTINGS
    IMG_DIMS = (10, 10, 3)

    trainmodel = createModel(IMG_DIMS)
    BATCH_SIZE = 3
    EPOCHS = 100
    INIT_LR = 1e-3
    #opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    trainmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    '''
    # CHECKPOINT
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
        mode='max')
    callbacks_list = [checkpoint]
    '''

    # MODEL FIT WITH DATA AUG
    history = trainmodel.fit_generator(
        aug.flow(X_train, Y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS, verbose=1, steps_per_epoch=len(X_train)//BATCH_SIZE,
        validation_data=(X_test, Y_test), class_weight = class_weight)

    scores = trainmodel.evaluate(X_test, Y_test)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

    y_pred = trainmodel.predict(X_test)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    print('\nConfusion Matrix')
    matrix = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)

'''
y_test = np.argmax(Y_test, axis=1)
y_pred = trainmodel.predict(X_test)

# salva um arquivo do modelo
trainmodel.save("model")'''

dataset, labels = load_dataset()


#if __name__ == '__main__':
train(dataset, labels)
