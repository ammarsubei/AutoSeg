from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
        model_name,
        monitor='val_loss',
        verbose=0,
        save_best_only=True)

tb = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=True)

early = EarlyStopping(patience=batch_size, verbose=1)

callbacks = [checkpoint, tb, early]