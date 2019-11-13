'''
    Algorithm for testing trained model
'''
import numpy as np
import pandas as pd
from keras.layers import Input
from keras import backend as K
from keras.models import load_model, Model

from dataset import load_dataset

def test_model():
    testData, testLabels = load_dataset()

    # DEFINE MODEL PARAMETERS
    model = load_model(MODEL_PATH)
    print(model.summary())

    predict = model.predict(testData)
    threshold = 0.5
    predict[predict > threshold] = 1
    predict[predict <= threshold] = 0

    y_test_non_category = [ np.argmax(t) for t in testLabels ]
    y_predict_non_category = [ np.argmax(z) for z in predict ]

    cm = ConfusionMatrix(actual_vector=y_test_non_category, predict_vector=y_predict_non_category)
    cm.save_html("classification_report")
    # print('\nConfusion Matrix: \n', cm, '\n')

    print(os.system("scp classification_report.html alex@192.168.0.180:~/Documents/git/oil_class"))

    # target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    
    # cm = confusion_matrix(y_test_non_category, y_predict_non_category)
    # print('\nConfusion Matrix: \n', cm)

    cr = classification_report(y_test_non_category, y_predict_non_category)
    print('\nClassification Report: \n', cr)

    scores = model.evaluate(testData, testLabels)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))

if __name__ == "__main__":
    test_model()
