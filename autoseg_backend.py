from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
import os, random
import numpy as np
import cv2
import pickle

# Get lists: train_img, train_label, etc.
# Create a method that batch-process the categorical labels to a one-hot matrix and saves them in a database along with the images.
# Create a generator method that yields a tuple of two tensors of shape (batch_size, img_height, img_width, num_channels).
# This method should also imshow() the segmentation of the sample image after each batch.

# Accepts and returns a numpy array.
def labelToOneHot(label, num_classes):
    return np.eye(num_classes)[label]

# Accepts and returns a numpy array.
def oneHotToLabel(one_hot):
    return one_hot.argmax(2).astype('uint8')


class VisualizeResult(Callback):
    def __init__(self, num_classes, image_path, label_path, validation_file_list):
        self.num_classes = num_classes
        # TODO: Randomly select and create these if not present.
        self.image = cv2.imread(os.getcwd() + '/sample_image.png')
        cv2.imshow('Sample Image', self.image)
        cv2.moveWindow('Sample Image', 10, 10)
        self.ground_truth = self.makeLabelPretty( cv2.imread(os.getcwd() + '/sample_label.png', 0) )
        cv2.imshow('Ground Truth', self.ground_truth)
        cv2.moveWindow('Ground Truth', 510, 10)
        cv2.waitKey(1)
        self.image_path = image_path
        self.label_path = label_path
        self.validation_file_list = validation_file_list


    # Accepts and returns a numpy array.
    def makeLabelPretty(self, label):
        prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
        colors = [
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
        [0,153,153],    # 13: dark turquoise
        ]

        assert self.num_classes <= len(colors)
        print

        for i in range(self.num_classes):
            prettyLabel[np.where( (label==[i]) )] = colors[i]

        return prettyLabel

    def on_batch_end(self, batch, logs={}):
        seg_result = oneHotToLabel( self.model.predict( np.array( [self.image] ) ).squeeze(0) )
        pl = self.makeLabelPretty(seg_result)
        cv2.imshow('Segmentation Result', pl)
        cv2.moveWindow('Segmentation Result', 1010, 10)
        cv2.waitKey(1)

    #def on_train_end(self):

class BackendHandler(object):

    def __init__(self, data_dir, num_classes, reinitialize=False):
        self.data_dir = os.getcwd() + data_dir
        self.num_classes = num_classes
        self.image_path = self.data_dir + 'images/'
        self.label_path = self.data_dir + 'labels/'
        self.file_list = os.listdir(self.data_dir + 'images/')
        cwd_contents = os.listdir(os.getcwd())
        # Sort the data into training and validation sets, or load already sorted sets.
        if reinitialize or 'training_file_list.pickle' not in cwd_contents or 'validation_file_list.pickle' not in cwd_contents:
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

    def generateData(self, batch_size, validating=False):
        if validating:
            data = self.validation_file_list
        else:
            data = self.training_file_list
        i = 0
        while True:
            image_batch = []
            label_batch = []
            for b in range(batch_size):
                if i == len(data):
                    i = 0
                    random.shuffle(training_file_list)
                    random.shuffle(validation_file_list)
                sample = data[i]
                i += 1
                image = cv2.imread(self.image_path + sample) / 255
                label = cv2.imread(self.label_path + sample, 0)
                image = image
                label = label
                one_hot = labelToOneHot(label, self.num_classes)
                image_batch.append(image)
                label_batch.append(one_hot)
            image_batch = np.array(image_batch)
            label_batch = np.array(label_batch)
            yield (image_batch, label_batch)

    def getCallbacks(self, model_name='test.h5', num_classes=12, patience=12):
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

        early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

        vis = VisualizeResult(self.num_classes, self.image_path, self.label_path, self.validation_file_list)

        return [checkpoint, early, vis]

#sg = SegGen('/data/', 11)
#print(next(sg.trainingGenerator(11)))
