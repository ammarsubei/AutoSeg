from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf
import os, random, math, string
import numpy as np
import cv2
import pickle

ignore_classes = [0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1]

# Get lists: train_img, train_label, etc.
# Create a method that batch-process the categorical labels to a one-hot matrix and saves them in a database along with the images.
# Create a generator method that yields a tuple of two tensors of shape (batch_size, img_height, img_width, num_channels).
# This method should also imshow() the segmentation of the sample image after each batch.

random.seed(1) # For reproducability.

# Accepts and returns a numpy array.
def labelToOneHot(label, num_classes):
    return np.eye(num_classes)[label]

# Accepts and returns a numpy array.
def oneHotToLabel(one_hot):
    return one_hot.argmax(2).astype('uint8')

# Reduces the 34 CityScapes classes to 9.
def remapClass(arr):
    o = arr.copy()
    arr[arr<=6] = 0
    arr[arr==7] = 1

    arr[arr==8] = 2
    arr[arr==9] = 2
    arr[arr==10] = 2

    arr[arr==11] = 3
    arr[arr==12] = 3
    arr[arr==13] = 3
    arr[arr==14] = 3
    arr[arr==15] = 3
    arr[arr==16] = 3

    arr[arr==17] = 4
    arr[arr==18] = 4
    arr[arr==19] = 4
    arr[arr==20] = 4

    arr[arr==21] = 5
    arr[arr==22] = 6
    arr[arr==23] = 7

    arr[arr==24] = 8
    arr[arr==25] = 8

    arr[arr>=26] = 9

    return o

def getID(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def pixelwise_crossentropy(target, output):
    # scale preds so that the class probas of each sample sum to 1
    '''
    output /= tf.reduce_sum(output,
                            reduction_indices=len(output.get_shape()) - 1,
                            keep_dims=True)
    '''
    # manual computation of crossentropy
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    return -tf.reduce_sum(target * tf.log(output))

def class_weighted_pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    with open('class_weights.pickle', 'rb') as f:
        weight = pickle.load(f)
    return -tf.reduce_sum(target * weight * tf.log(output))


def pixelwise_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=2),
                          K.argmax(y_pred, axis=2)),
                  K.floatx())

def mIoU(y_true, y_pred):
    false_negatives = tf.metrics.false_negatives(y_true, y_pred)[0]
    false_positives = tf.metrics.false_positives(y_true, y_pred)[0]
    true_positives = tf.metrics.true_positives(y_true, y_pred)[0]

    iou = true_positives / (true_positives + false_positives + false_negatives)

    return iou

class VisualizeResult(Callback):
    def __init__(self, num_classes, image_path, label_path, validation_file_list):
        self.num_classes = num_classes
        self.image_path = image_path
        self.label_path = label_path
        self.validation_file_list = validation_file_list
        self.colors = None
        i = random.choice(self.validation_file_list)
        self.image = cv2.imread(i[0])
        cv2.imshow('Sample Image', cv2.resize(self.image, (800,400)) )
        cv2.moveWindow('Sample Image', 10, 10)
        self.ground_truth = remapClass( cv2.imread(i[1], 0) )
        self.ground_truth = self.makeLabelPretty(self.ground_truth)
        cv2.imshow('Ground Truth', cv2.resize(self.ground_truth, (800,400)))
        cv2.moveWindow('Ground Truth', 850, 10)
        cv2.waitKey(1)


    # Accepts and returns a numpy array.
    def makeLabelPretty(self, label):
        prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
        if self.colors is None:
            '''
            self.colors = [
            [0,0,0],        # 0: void
            [128,64,128],   # 1: road
            [244, 35,232],  # 2: sidewalk
            [70,70,70],     # 3: construction
            [153,153,153],  # 4: object
            [107,142, 35],  # 5: vegetation
            [152,251,152],  # 6: terrain
            [ 70,130,180],  # 7: sky
            [220, 20, 60],  # 8: person
            [  0,  0,142]   # 9: vehicle
            ]
            '''
            with open('cityscapes_color_mappings.pickle', 'rb') as f:
                self.colors =  pickle.load(f)

            assert self.num_classes <= len(self.colors)

        for i in range(self.num_classes):
            prettyLabel[np.where( (label==[i]) )] = self.colors[i]

        return prettyLabel

    def on_batch_end(self, batch, logs={}):
        if batch % 5 == 0 or batch < 10:
            seg_result = self.model.predict( np.array( [self.image] ) )
            main = self.makeLabelPretty(oneHotToLabel(seg_result.squeeze(0)))
            cv2.imshow('Segmentation Result', cv2.resize(main, (800,400)))
            cv2.moveWindow('Segmentation Result', 850, 500)
            cv2.waitKey(1)

    def on_epoch_end(self, epoch, logs={}):
        new_img = random.choice(self.validation_file_list)
        self.image = cv2.imread(new_img[0])
        self.ground_truth = self.makeLabelPretty( remapClass( cv2.imread(new_img[1], 0) ) )
        cv2.imshow('Sample Image', cv2.resize(self.image, (800,400)))
        cv2.imshow('Ground Truth', cv2.resize(self.ground_truth, (800,400)))

    def on_train_end(self, logs={}):
        print("Training ended!")
        seg_result = oneHotToLabel( self.model.predict( np.array( [self.image] ) ).squeeze(0) )
        pl = self.makeLabelPretty(seg_result)
        cv2.imwrite('sample_output.png', pl)
        # TODO: Run predict over test set and write results to files in "results" dir.

class BackendHandler(object):

    def __init__(self, data_dir, num_classes, visualize_while_training=False):
        self.data_dir = os.getcwd() + data_dir
        self.num_classes = num_classes
        self.visualize_while_training = visualize_while_training
        self.image_path = self.data_dir + 'images/'
        self.label_path = self.data_dir + 'labels_fine/'
        self.cwd_contents = os.listdir(os.getcwd())
        self.training_file_list = self.getFileList('train/')
        self.validation_file_list = self.getFileList('val/')
        self.getClassWeights()

    # Sort the data into training and validation sets, or load already sorted sets.
    def getFileList(self, category='train/'):
        # Make training file list.
        file_dir = self.image_path + category
        allfiles = []
        for path, subdirs, files in os.walk(file_dir):
            for f in files:
                f = f.split('_')[0] + '/' + f
                allfiles.append(f)
        file_list = []
        for f in allfiles:
            input_output = (self.image_path + category + f, self.label_path + category + f.replace('leftImg8bit', 'gtFine_labelIds'))
            file_list.append(input_output)
        return file_list

    def getClassWeights(self):
        if 'class_weights.pickle' not in self.cwd_contents:
            file_list = self.training_file_list + self.validation_file_list
            print("Calculating class weights for " + str(len(file_list)) + " images, this may take a while...")
            classcounts = [1]*self.num_classes
            c = 0
            for f in file_list:
                lbl = cv2.imread(f[1], 0)
                show = lbl*int(255/self.num_classes)
                cv2.putText(show, str(c) + '/' + str(len(file_list)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                cv2.imshow('Processing...', show)
                cv2.waitKey(1)
                for i in range(self.num_classes):
                    classcounts[i] += len(np.where(lbl == i)[0])
                c += 1
            total = sum(classcounts)
            class_weights = [0]*self.num_classes
            for i in range(self.num_classes):
                class_weights[i] = ignore_classes[i] * total / classcounts[i]
            self.class_weights = [100 * float(w) / max(class_weights) for w in class_weights]
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            with open('class_weights.pickle', 'wb') as f:
                pickle.dump(self.class_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('class_weights.pickle', 'rb') as f:
                self.class_weights = pickle.load(f)
        print(self.class_weights)


    def generateData(self, batch_size, validating=False, horizontal_flip=True, vertical_flip=False, adjust_brightness=0.1, rotate=5, zoom=0.1):
        if validating:
            data = self.validation_file_list
        else:
            data = self.training_file_list
        random.shuffle(data)

        i = 0
        while True:
            image_batch = []
            label_batch = []
            small_label_batch = []
            for b in range(batch_size):
                if i == len(data):
                    i = 0
                    random.shuffle(data)
                sample = data[i]
                i += 1
                image = cv2.imread(sample[0]) / 255
                label = remapClass( cv2.imread(sample[1], 0) )

                # Data Augmentation
                if not validating:
                    if horizontal_flip and random.randint(0,1):
                        cv2.flip(image, 1)
                        cv2.flip(label, 1)
                    if vertical_flip and random.randint(0,1):
                        cv2.flip(image, 0)
                        cv2.flip(label, 0)
                    if adjust_brightness:
                        factor = 1 + abs(random.gauss(mu=0, sigma=adjust_brightness))
                        if random.randint(0,1):
                            factor = 1 / factor
                        image = 255*( (image/255)**factor )
                        image = np.array(image, dtype='uint8')
                    if rotate:
                        angle = random.gauss(mu=0, sigma=rotate)
                    else:
                        angle = 0
                    if zoom:
                        scale = random.gauss(mu=1, sigma=zoom)
                    else:
                        scale = 1
                    if rotate or zoom:
                        rows,cols = label.shape
                        M = cv2.getRotationMatrix2D( (cols/2, rows/2), angle, scale)
                        image = cv2.warpAffine(image, M, (cols, rows))
                        label = cv2.warpAffine(label, M, (cols, rows))

                one_hot = labelToOneHot(label, self.num_classes)
                image_batch.append(image)
                label_batch.append(one_hot)
            image_batch = np.array(image_batch)
            label_batch = np.array(label_batch)
            yield (image_batch, label_batch)

    def getCallbacks(self, model_name='test.h5', patience=12):
        checkpoint = ModelCheckpoint(
            model_name,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True)

        tb = TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            write_images=True)

        early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

        if self.visualize_while_training:
            vis = VisualizeResult(self.num_classes, self.image_path, self.label_path, self.validation_file_list)
            return [checkpoint, tb, vis]
        else:
            return [checkpoint, tb]

#sg = SegGen('/data/', 11)
#print(next(sg.trainingGenerator(11)))
