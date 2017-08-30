from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf
import os, random, math, string
import pickle
import numpy as np
import cv2
from PIL import Image

INPUT_SHAPE = (1024, 768)

# Class weights. Classes with weight 0 do not contribute to the loss.

IGNORE_CLASSES = [0, 0, 0, 0, 0, 0, 0,  # ignore 'void' class
                  1, 1, 0, 0,           # ignore 'parking', 'rail track'
                  1, 1, 1, 0, 0, 0,     # ignore 'guard rail', 'bridge', 'tunnel'
                  1, 0,                 # ignore 'polegroup'
                  1, 1, 1, 1,           # 'object' class
                  1, 1, 1,              # 'nature' and 'sky' classes
                  1, 1,                 # 'human' class
                  1, 0, 0, 1, 1, 1]     # ignore 'caravan', 'trailer'

MAPILLARY_COLORS = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180], [102, 102, 156], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60], [255, 0, 0], [255, 0, 0], [255, 0, 0], [200, 128, 128], [255, 255, 255], [64, 170, 64], [128, 64, 64], [70, 130, 180], [255, 255, 255], [152,
    251, 152], [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [0, 0, 0], [220, 220, 220], [170, 170, 170], [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33], [170, 170, 170], [0, 0, 142], [170, 170, 170], [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 142], [250, 170, 30], [192, 192, 192], [220, 220, 0], [180, 165, 180], [119, 11, 32], [0, 0, 142], [0, 60, 100], [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0,
        70], [0, 0, 192], [32, 32, 32], [0, 0, 0], [0, 0, 0]]

print(len(MAPILLARY_COLORS))


def label_to_onehot(label, num_classes):
    """Converts labels (e.g. 2) to one-hot vectors (e.g. [0,0,1,0]).
    Accepts and returns a numpy array."""
    return np.eye(num_classes)[label]

# Accepts and returns a numpy array.
def one_hot_to_label(one_hot):
    """Converts one-hot vectors (e.g. [0,0,1,0]) to labels (e.g. 2).
    Accepts and returns a numpy array."""
    return one_hot.argmax(2).astype('uint8')

def remap_class(arr):
    """Remaps CityScapes classes as explained below."""
    return arr

    arr[arr == 12] = 11 # walls -> buildings
    arr[arr == 13] = 11 # fences -> buildings

    arr[arr == 25] = 24 # cyclists -> people

    arr[arr == 27] = 26 # truck -> car
    arr[arr == 28] = 26 # bus -> car
    arr[arr == 32] = 26 # motorcycle -> car

    return arr

def deprettify_mapillary(pretty):
    print(pretty)
    label = np.zeros((pretty.shape[0], pretty.shape[1], 1))
    for i in range(len(MAPILLARY_COLORS)):
        print(((pretty == [180,130,70]).shape))
        label[np.where((pretty == MAPILLARY_COLORS[i]))] = i
    print(label)
    return label

def pixelwise_crossentropy(target, output):
    """Keras' default crossentropy function wasn't working, not sure why."""
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    return -tf.reduce_sum(target * tf.log(output))

def class_weighted_pixelwise_crossentropy(target, output):
    """As above. IGNORE_CLASSES is already used in calculating class_weights."""
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    with open('class_weights.pickle', 'rb') as f:
        weights = pickle.load(f)
    return -tf.reduce_sum(target * weights * tf.log(output))

def pixelwise_accuracy(y_true, y_pred):
    """Same as Keras' default accuracy function, but with axis=-1 changed to axis=2.
        These should be equivalent, I'm not sure why they result in different values."""
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=2),
                          K.argmax(y_pred, axis=2)),
                  K.floatx()))

class VisualizeResult(Callback):
    """Custom callback that shows the model's prediction on a sample image during training."""
    def __init__(self, num_classes, image_path, label_path, validation_file_list):
        self.num_classes = num_classes
        self.image_path = image_path
        self.label_path = label_path
        self.validation_file_list = validation_file_list
        self.colors = MAPILLARY_COLORS
        i = random.choice(self.validation_file_list)
        print(i)
        self.image = cv2.resize(cv2.imread(i[0]), INPUT_SHAPE)
        cv2.imshow('Sample Image', cv2.resize(self.image, (600, 450)))
        cv2.moveWindow('Sample Image', 10, 10)
        self.ground_truth = cv2.resize(cv2.imread(i[1]), INPUT_SHAPE)
        print(self.ground_truth)
        cv2.imshow('Ground Truth', cv2.resize(self.ground_truth, (600, 450)))
        cv2.moveWindow('Ground Truth', 850, 10)
        cv2.waitKey(1)

    # Accepts and returns a numpy array.
    def make_label_pretty(self, label):
        """Maps integer labels to RGB color values, for visualization."""
        pretty_label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
        if self.colors is None:
            # Saved as example.
            '''
            self.colors = [
            [0,0,0],        # 0: void
            [128,64,128],   # 1: road
            [  0,  0,142],  # 3: vehicle
            [220, 20, 60],  # 2: person
            ]
            '''
            with open('cityscapes_color_mappings.pickle', 'rb') as f:
                self.colors = pickle.load(f)

            assert self.num_classes <= len(self.colors)

        for i in range(self.num_classes):
            pretty_label[np.where((label == [i]))] = self.colors[i]

        return pretty_label

    def on_batch_end(self, batch, logs={}):
        """Updates the displayed prediction after every 10th batch."""
        if batch % 10 == 0 or batch < 5:
            seg_result = self.model.predict(np.array([self.image]))
            main = self.make_label_pretty(one_hot_to_label(seg_result.squeeze(0)))
            cv2.imshow('Segmentation Result', cv2.resize(main, (600, 450)))
            cv2.moveWindow('Segmentation Result', 850, 500)
            cv2.waitKey(1)

    def on_epoch_end(self, epoch, logs={}):
        """Changes the image used as an example at the end of each epoch."""
        new_img = random.choice(self.validation_file_list)
        self.image = cv2.resize(cv2.imread(new_img[0]), (800, 600))
        self.ground_truth = cv2.resize(cv2.imread(new_img[1]), INPUT_SHAPE)
        cv2.imshow('Sample Image', cv2.resize(self.image, (600, 450)))
        cv2.imshow('Ground Truth', cv2.resize(self.ground_truth, (600, 450)))

    def on_train_end(self, logs={}):
        """Currently does nothing useful, saved here as a TODO."""
        print("Training ended!")

class BackendHandler(object):
    """Handles data generation and hides callback creation."""
    def __init__(self, data_dir, num_classes, visualize_while_training=False):
        self.data_dir = os.getcwd() + data_dir
        self.num_classes = num_classes
        self.visualize_while_training = visualize_while_training
        self.image_path = self.data_dir + 'images/'
        self.label_path = self.data_dir + 'labels/'
        self.cwd_contents = os.listdir(os.getcwd())
        self.training_file_list = self.get_file_list('training/')
        self.validation_file_list = self.get_file_list('validation/')
        self.get_class_weights()

    def get_file_list(self, category='train/'):
        """Recursively get raw images and annotations in given dir.
            Return as tuple where first element is the path to the image
            and the second element is the path to the corresponding annotation."""
        file_dir = self.image_path + category
        allfiles = []
        for path, subdirs, files in os.walk(file_dir):
            for f in files:
                #f = f.split('_')[0] + '/' + f
                allfiles.append(f)
        file_list = []
        for f in allfiles:
            input_output = (self.image_path + category + f,
                            self.label_path + category + f.replace('.jpg', '.png'))
            file_list.append(input_output)
        return file_list

    def get_class_weights(self):
        """Iterates over the entire dataset and for each class calculates the
            instances of that class divided by the total number of instances.
            Since this is time-consuming for large datasets, save the result
            to a pickle file for reuse."""
        if 'class_weights.pickle' not in self.cwd_contents:
            file_list = self.training_file_list + self.validation_file_list
            print("Calculating class weights for " + str(len(file_list)) + \
                  " images, this may take a while...")
            classcounts = [1]*self.num_classes
            count = 0
            for f in file_list:
                lbl = cv2.imread(f[1], 0)
                show = lbl*int(255/self.num_classes)
                cv2.putText(show, str(count) + '/' + str(len(file_list)),
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                cv2.imshow('Processing...', show)
                cv2.waitKey(1)
                for i in range(self.num_classes):
                    classcounts[i] += len(np.where(lbl == i)[0])
                count += 1
            total = sum(classcounts)
            class_weights = [0]*self.num_classes
            for i in range(self.num_classes):
                class_weights[i] = IGNORE_CLASSES[i] * total / classcounts[i]
            self.class_weights = [100 * float(w) / max(class_weights) for w in class_weights]
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            with open('class_weights.pickle', 'wb') as f:
                pickle.dump(self.class_weights, f, protocol=0)
        else:
            with open('class_weights.pickle', 'rb') as f:
                self.class_weights = pickle.load(f)
        #print(self.class_weights)


    def generate_data(self, batch_size, validating=False,
                      horizontal_flip=True, vertical_flip=False,
                      adjust_brightness=0.1, rotate=5, zoom=0.1):
        """Replaces Keras' native ImageDataGenerator."""
        if validating:
            data = self.validation_file_list
        else:
            data = self.training_file_list
        random.shuffle(data)

        i = 0
        while True:
            image_batch = []
            label_batch = []
            for batch in range(batch_size):
                if i == len(data):
                    i = 0
                    random.shuffle(data)
                sample = data[i]
                i += 1
                image = cv2.resize(cv2.imread(sample[0]), INPUT_SHAPE)
                label = cv2.resize(cv2.imread(sample[1],0), INPUT_SHAPE)

                # Data Augmentation
                if not validating:
                    # Horizontal flip.
                    if horizontal_flip and random.randint(0, 1):
                        cv2.flip(image, 1)
                        cv2.flip(label, 1)
                    # Vertical flip.
                    if vertical_flip and random.randint(0, 1):
                        cv2.flip(image, 0)
                        cv2.flip(label, 0)
                    # Randomly make image brighter or darker.
                    if adjust_brightness:
                        factor = 1 + abs(random.gauss(mu=0, sigma=adjust_brightness))
                        if random.randint(0, 1):
                            factor = 1 / factor
                        image = 255*((image/255)**factor)
                        image = np.array(image, dtype='uint8')
                    # Randomly rotate image within a certain range.
                    if rotate:
                        angle = random.gauss(mu=0, sigma=rotate)
                    else:
                        angle = 0
                    # Randomly zoom image.
                    if zoom:
                        scale = random.gauss(mu=1, sigma=zoom)
                    else:
                        scale = 1
                    # Apply rotation/zoom calculated above.
                    if rotate or zoom:
                        rows, cols = label.shape
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                        image = cv2.warpAffine(image, M, (cols, rows))
                        label = cv2.warpAffine(label, M, (cols, rows))

                one_hot = label_to_onehot(label, self.num_classes)
                image_batch.append((image.astype(float) - 128) / 128)
                label_batch.append(one_hot)
            image_batch = np.array(image_batch)
            label_batch = np.array(label_batch)

            yield (image_batch, label_batch)

    def get_callbacks(self, model_name='test.h5', patience=500, logdir='./logs/default'):
        """Returns a standard set of callbacks.
            Kept here mainly to avoid clutter in main.py"""
        checkpoint = ModelCheckpoint(
            model_name,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True)

        tb = TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            write_images=True)

        early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

        if self.visualize_while_training:
            vis = VisualizeResult(self.num_classes, self.image_path,
                                  self.label_path, self.validation_file_list)
            return [checkpoint, tb, vis]
        else:
            return [checkpoint, tb]
