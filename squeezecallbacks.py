from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def getCallbacks(model_name, patience=12):
    checkpoint = ModelCheckpoint(
        model_name,
        monitor='loss',
        verbose=0,
        save_best_only=True)

    tb = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    early = EarlyStopping(patience=patience, verbose=1)

    return [checkpoint]
