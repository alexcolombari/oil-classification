from keras import backend as K

def recall_m(test_labels, Y_pred):
        true_positives = K.sum(K.round(K.clip(test_labels * Y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(test_labels, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(test_labels, Y_pred):
        true_positives = K.sum(K.round(K.clip(test_labels * Y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(test_labels, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(test_labels, Y_pred):
    precision = precision_m(test_labels, Y_pred)
    recall = recall_m(test_labels, Y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
