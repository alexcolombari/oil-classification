from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

MODEL_PATH = "model-save/"
MODEL_SAVED = MODEL_PATH + "model.h5"


def model_callbacks(EPOCHS, BATCH_SIZE):
    patience = (EPOCHS * 10) / 100      # 10%
    # CHECKPOINT
    checkpoint = ModelCheckpoint(MODEL_SAVED, monitor = 'val_loss', verbose = 1, save_best_only = True,
        mode='min', save_weights_only = False)
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 300, verbose = 1,
        mode = 'min', restore_best_weights = True)
    tensor_board = TensorBoard(log_dir = './logs', histogram_freq = 1, batch_size = BATCH_SIZE,
        write_graph = True, write_grads = False, write_images = False, embeddings_layer_names = None,
        update_freq = 3) 
    callbacks_list = [checkpoint, early]

    return callbacks_list
