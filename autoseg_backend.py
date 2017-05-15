from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf
import os, random, math, string
import numpy as np
import cv2
import pickle

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
    return - tf.reduce_mean(target * tf.log(output))

def pixelwise_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=2),
                          K.argmax(y_pred, axis=2)),
                  K.floatx())

class VisualizeResult(Callback):
    def __init__(self, num_classes, image_path, label_path, validation_file_list):
        self.num_classes = num_classes
        self.image_path = image_path
        self.label_path = label_path
        self.validation_file_list = validation_file_list
        self.colors = None
        i = random.choice(self.validation_file_list)
        self.image = cv2.imread(self.image_path + i)
        cv2.imshow('Sample Image', self.image)
        cv2.moveWindow('Sample Image', 10, 10)
        self.ground_truth = self.makeLabelPretty( cv2.imread(self.label_path + i, 0) )
        cv2.imshow('Ground Truth', self.ground_truth)
        cv2.imwrite('sample_ground_truth.png', self.ground_truth)
        cv2.moveWindow('Ground Truth', 510, 10)
        #cv2.imshow('Auxiliary Ground Truth', cv2.resize(self.ground_truth, (0,0), fx=0.125, fy=0.125))
        #cv2.moveWindow('Auxiliary Ground Truth', 510, 410)
        cv2.waitKey(1)


        self.activity_by_layer = []


    # Accepts and returns a numpy array.
    def makeLabelPretty(self, label):
        prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
        if self.colors is None:

            self.colors = [
            [255,102,102],  # 0: light red
            [255,255,102],  # 1: light yellow
            [102,255,102],  # 2: light green
            [102,255,255],  # 3: light blue
            [102,102,255],  # 4: light indigo
            [255,102,255],  # 5: light pink
            [255,178,102],  # 6: light orange
            [153,51,255],   # 7: violet
            [153,0,0],      # 8: dark red
            [153,153,0],    # 9: dark yellow
            [0,102,0],      # 10: dark green
            [0,76,153],     # 11: dark blue
            [102,0,51],     # 12: dark pink
            ]
            '''
            with open('cityscapes_color_mappings.pickle', 'rb') as f:
                self.colors =  pickle.load(f)
            '''
            assert self.num_classes <= len(self.colors)

        for i in range(self.num_classes):
            prettyLabel[np.where( (label==[i]) )] = self.colors[i]

        return prettyLabel

    def calculateActivityByLayer(self):
        current_weights = self.model.get_weights()
        avg_diff = []
        for i in range(len(current_weights)):
            avg_diff.append( np.mean(current_weights[i] - self.previous_epoch_weights[i]) )
        self.activity_by_layer += avg_diff
        histogram = np.zeros((500,1000,3))
        h_width = int(1000 / len(self.model.layers) - 1)
        x = 0
        for i in range(len(avg_diff)):
            cv2.rectangle(histogram, (x,0), (x+h_width, int(self.activity_by_layer[i] / max(self.activity_by_layer))*400), (255,0,0), -1)
            x = x + h_width
        cv2.imshow('Activity By Layer', histogram)
        cv2.moveWindow('Activity By Layer', 10, 510)
        cv2.waitKey(1)



    def on_batch_end(self, batch, logs={}):
        seg_result = self.model.predict( np.array( [self.image] ) )
        main = self.makeLabelPretty(oneHotToLabel(seg_result[0].squeeze(0)))
        aux = self.makeLabelPretty(oneHotToLabel(seg_result[1].squeeze(0)))
        cv2.imshow('Segmentation Result', main)
        cv2.moveWindow('Segmentation Result', 1010, 10)
        aux_result = oneHotToLabel( self.model.predict( np.array( [self.image] ) )[1].squeeze(0) )
        cv2.imshow('Scaled Auxiliary Result', cv2.resize(aux, (0,0), fx=8, fy=8))
        cv2.moveWindow('Scaled Auxiliary Result', 1010, 410)
        cv2.waitKey(1)

    def on_epoch_begin(self, epoch, logs={}):
        self.previous_epoch_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs={}):
        new_img = random.choice(self.validation_file_list)
        self.image = cv2.imread(self.image_path + new_img)
        self.ground_truth = self.makeLabelPretty( cv2.imread(self.label_path + new_img, 0) )
        cv2.imshow('Sample Image', self.image)
        cv2.imshow('Ground Truth', self.ground_truth)
        #cv2.imshow('Auxiliary Ground Truth', cv2.resize(self.ground_truth, (0,0), fx=0.125, fy=0.125))
        #self.calculateActivityByLayer()

    def on_train_end(self, logs={}):
        print("Training ended!")
        seg_result = oneHotToLabel( self.model.predict( np.array( [self.image] ) ).squeeze(0) )
        pl = self.makeLabelPretty(seg_result)
        cv2.imwrite('sample_output.png', pl)
        # TODO: Run predict over test set and write results to files in "results" dir.

class BackendHandler(object):

    def __init__(self, data_dir, num_classes, reinitialize=False):
        self.data_dir = os.getcwd() + data_dir
        # TODO: Calculate num_classes automatically by iterating over the entire dataset and calculating len(uniques).
        self.num_classes = num_classes
        self.image_path = self.data_dir + 'images/'
        self.label_path = self.data_dir + 'labels/'
        self.file_list = os.listdir(self.data_dir + 'images/')
        self.cwd_contents = os.listdir(os.getcwd())
        self.splitData(reinitialize)
        self.getClassWeights(reinitialize)

    # Sort the data into training and validation sets, or load already sorted sets.
    def splitData(self, reinitialize):
        if reinitialize or 'training_file_list.pickle' not in self.cwd_contents or 'validation_file_list.pickle' not in self.cwd_contents:
            print("Splitting training/validation data 80/20...")
            random.shuffle(self.file_list)
            breakpoint = int(len(self.file_list) * 0.8)
            self.training_file_list, self.validation_file_list = self.file_list[:breakpoint], self.file_list[breakpoint:]
            # pickle the files
            with open('training_file_list.pickle', 'wb') as f:
                pickle.dump(self.training_file_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('validation_file_list.pickle', 'wb') as f:
                pickle.dump(self.validation_file_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading training/validation split...")
            with open('training_file_list.pickle', 'rb') as f:
                self.training_file_list =  pickle.load(f)
            with open('validation_file_list.pickle', 'rb') as f:
                self.validation_file_list = pickle.load(f)
        print("Train on " + str(len(self.training_file_list)) + " images, validate on " + str(len(self.validation_file_list)) + " images.")

    def getClassWeights(self, reinitialize):
        if reinitialize or 'class_weights.pickle' not in self.cwd_contents:
            print("Calculating class weights for " + str(len(self.file_list)) + " images, this may take a while...")
            classcounts = [0]*self.num_classes
            for f in self.file_list:
                lbl = cv2.imread(self.label_path + f, 0)
                for i in range(self.num_classes):
                    classcounts[i] += len(np.where(lbl == i)[0])
            total = sum(classcounts)
            self.class_weights = {}
            for i in range(self.num_classes):
                self.class_weights.update( {i : float(math.log( total / classcounts[i] ))} )
            with open('class_weights.pickle', 'wb') as f:
                pickle.dump(self.class_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Class weights found, loading...")
            with open('class_weights.pickle', 'rb') as f:
                self.class_weights = pickle.load(f)
        #print("Class weights: ", end="")
        print(self.class_weights)

    def generateData(self, batch_size, validating=False):
        if validating:
            data = self.validation_file_list
        else:
            data = self.training_file_list
        i = 0
        while True:
            image_batch = []
            label_batch = []
            small_label_batch = []
            for b in range(batch_size):
                if i == len(data):
                    i = 0
                    random.shuffle(self.training_file_list)
                sample = data[i]
                i += 1
                image = cv2.imread(self.image_path + sample) / 255
                label = cv2.imread(self.label_path + sample, 0)
                image = image
                label = label
                small_label = cv2.resize(label, (0,0), fx=0.125, fy=0.125)
                one_hot = labelToOneHot(label, self.num_classes)
                small_one_hot = labelToOneHot(small_label, self.num_classes)
                image_batch.append(image)
                label_batch.append(one_hot)
                small_label_batch.append(small_one_hot)
            image_batch = np.array(image_batch)
            label_batch = np.array(label_batch)
            small_label_batch = np.array(small_label_batch)
            yield (image_batch, [label_batch, small_label_batch])

    def getCallbacks(self, model_name='test.h5', num_classes=12, patience=12):
        checkpoint = ModelCheckpoint(
            model_name,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True)

        tb = TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            write_graph=True,
            write_images=True)

        early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

        vis = VisualizeResult(self.num_classes, self.image_path, self.label_path, self.validation_file_list)

        return [checkpoint, tb, vis]

#sg = SegGen('/data/', 11)
#print(next(sg.trainingGenerator(11)))
