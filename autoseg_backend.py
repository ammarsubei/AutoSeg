import os

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import cv2

def pixelwise_crossentropy(target, output):
    """
    Keras' default crossentropy function wasn't working, not sure why.
    """
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    return -tf.reduce_sum(target * tf.log(output))

def pixelwise_accuracy(y_true, y_pred):
    """
    Same as Keras' default accuracy function, but with axis=-1 changed to axis=2.
    These should be equivalent, I'm not sure why they result in different values.
    """
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=2),
                                 K.argmax(y_pred, axis=2)),
                         K.floatx()))

def get_callbacks(model_name='test.h5', logdir='./logs/DEFAULT'):
    """
    Returns a standard set of callbacks.
    Kept here mainly to avoid clutter in train_model.py
    """
    checkpoint = ModelCheckpoint(
        model_name,
        monitor='val_loss',
        verbose=0,
        save_best_only=True)

    tb = TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        write_graph=True,
        write_grads=True,
        write_images=True)

    return [checkpoint, tb]

def get_file_list(directories):
    """
    For each directory in a list, recursively find all files in it.
    Return a list of lists of files of the same length as directories.
    """
    file_list = []
    for directory in directories:
        contents = []
        for path, subdirs, files in os.walk(os.getcwd() + directory):
            for f in files:
                contents.append(os.path.join(path, f))
        contents.sort()
        file_list.append(contents)

    return file_list

def merge_file_lists(input_files, output_files):
    inputs = []
    for i in range(len(input_files[0])):
        inp = []
        for j in range(len(input_files)):
            inp.append(input_files[j][i])
        inputs.append(inp)
    outputs = []
    for i in range(len(output_files[0])):
        outp = []
        for j in range(len(output_files)):
            outp.append(output_files[j][i])
        outputs.append(outp)

    data = []
    for i in range(len(inputs[0])):
        data.append((inputs[i], outputs[i]))
    return data

class Cityscapes(object):
    """
    Contains the parameters of a particular dataset, noteably the file lists.
    """
    def __init__(self):
        self.num_classes = 34
        self.colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 74, 111], [81, 0, 81], [128, 64, 128], [232, 35, 244],
                       [160, 170, 250], [140, 150, 230], [70, 70, 70],
                       [156, 102, 102], [153, 153, 190], [180, 165, 180],
                       [100, 100, 150], [90, 120, 150], [153, 153, 153],
                       [153, 153, 153], [30, 170, 250], [0, 220, 220],
                       [35, 142, 107], [152, 251, 152], [180, 130, 70],
                       [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
                       [100, 60, 0], [90, 0, 0], [110, 0, 0], [100, 80, 0],
                       [230, 0, 0], [32, 11, 119]]
        self.data_dir = 'Cityscapes/'

    def generate_data(self, batch_size, augment_data=False, validating=True):
        """
        Replaces Keras' native ImageDataGenerator.
        """
        print(10)
        if not validating:
            training_input_files = get_file_list(['/cityscapes_768/images_left/train',
                                                 '/cityscapes_768/images_right/train'])
            training_output_files = get_file_list(['/cityscapes_768/labels_fine/train'])
            data = merge_file_lists(training_input_files, training_output_files)
        else:
            validation_input_files = get_file_list(['/cityscapes_768/images_left/val',
                                                    '/cityscapes_768/images_right/val'])
            validation_output_files = get_file_list(['/cityscapes_768/labels_fine/val'])
            data = merge_file_lists(validation_input_files, validation_output_files)


        random.shuffle(data)
        print(data)

        i = 0
        while True:
            input_batch = []
            output_batch = []
            for b in range(batch_size):
                if i == len(data):
                    i = 0
                    random.shuffle(data)
                sample = data[i]
                i += 1

                inputs = []
                for inp in sample[0]:
                    inputs.append((cv2.imread(inp).astype(float) - 128) / 128)

                outputs = []
                for outp in sample[1]:
                    outputs.append(label_to_onehot(cv2.imread(outp, 0), 34))

                input_batch.append(inputs)
                print(inputs)
                print(outputs)
                output_batch.append(outputs)


            yield (input_batch, output_batch)

if __name__ == "__main__":
    cityscapes = Cityscapes()
    cityscapes.generate_data(10)
