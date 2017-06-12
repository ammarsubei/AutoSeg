import tensorflow as tf
import cv2
import os, pickle
import numpy as np
from autoseg_backend import BackendHandler, oneHotToLabel

os.environ["CUDA_VISIBLE_DEVICES"]="1"

num_classes = 34
data_dir = '/cityscapes_800/'

def makeLabelPretty(label):
    prettyLabel = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
    with open('cityscapes_color_mappings.pickle', 'rb') as f:
        colors = pickle.load(f)
    for i in range(num_classes):
        prettyLabel[np.where( (label==[i]) )] = colors[i]

    return prettyLabel

sess = tf.Session()
saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
print(sess.graph.get_operations())

last_op = sess.graph.get_operation_by_name('main/truediv')
main = last_op.values()[0]

backend = BackendHandler(data_dir=data_dir, num_classes=num_classes)

x = sess.graph.get_tensor_by_name('input_1:0')
d1 = sess.graph.get_tensor_by_name('dropout_1/keras_learning_phase:0')

for batch in backend.generateData(batch_size=1, validating=True):
    print(batch[0].shape)
    predictions = main.eval(feed_dict={
        x: batch[0], d1: False}, session=sess)
    for i in range(len(predictions)):
        cv2.imshow('Image', batch[0][i])
        cv2.moveWindow('Image', 10, 10)
        cv2.imshow('Ground Truth', makeLabelPretty( oneHotToLabel( batch[1][i] ) ))
        cv2.moveWindow('Ground Truth', 850, 10)
        cv2.imshow('Model Output', makeLabelPretty( oneHotToLabel(predictions[i]) ))
        cv2.moveWindow('Model Output', 850, 500)
        cv2.waitKey(5000)
